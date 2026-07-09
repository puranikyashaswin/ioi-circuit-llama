import torch
import json
import time
import random
import os
from statistics import mean

from .utils import (safe_div, jaccard, logit_diff, name_tokens, set_seed,
                     get_layers, n_layers, mem_mb, cleanup)
from .dataset import make_pairs
from .attribution import score_all_layers, pick_top_layers


# patch just the delta (output - input), full hidden-state patching makes almost anything look good.


def _get_contribs(model, tokenizer, prompt, layers, device):
    """Extract per-layer residual stream contributions (output - input).
    LLaMA adds the residual inside the block, so this hooks the whole
    block and subtracts pre from post.
    """
    pre, post = {}, {}
    handles = []

    def _pre(li):
        def h(mod, args):
            pre[li] = (args[0] if isinstance(args, tuple) else args).detach().cpu()
        return h

    def _post(li):
        def h(mod, inp, out):
            post[li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
        return h

    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(_pre(li)))
        handles.append(layer.register_forward_hook(_post(li)))

    toks = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**toks)
    for h in handles:
        h.remove()

    return {li: post[li] - pre[li] for li in pre}


def patch_and_measure(model, tokenizer, pairs, circuit_layers, device):
    """Run corrupt input through the model while injecting contribution
    corrections at circuit layers. Returns clean, corrupt, and patched
    logit-diff margins.
    """
    layers = get_layers(model)
    mdtype = next(model.parameters()).dtype

    cm_sum, xm_sum, pm_sum = 0.0, 0.0, 0.0

    for pair in pairs:
        cid = name_tokens(tokenizer, pair["correct_name"])
        wid = name_tokens(tokenizer, pair["wrong_name"])

        clean_c = _get_contribs(model, tokenizer, pair["clean"], layers, device)
        corrupt_c = _get_contribs(model, tokenizer, pair["corrupt"], layers, device)

        toks_c = tokenizer(pair["clean"], return_tensors="pt").to(device)
        with torch.no_grad():
            cm = logit_diff(model(**toks_c).logits, cid, wid)

        toks_x = tokenizer(pair["corrupt"], return_tensors="pt").to(device)
        with torch.no_grad():
            xm = logit_diff(model(**toks_x).logits, cid, wid)

        # add back clean contribution deltas for the circuit layers only
        corrections = {}
        for li in circuit_layers:
            corrections[li] = (clean_c[li] - corrupt_c[li]).to(device).to(mdtype)

        handles = []
        for li in circuit_layers:
            handles.append(layers[li].register_forward_hook(_correction_hook(corrections[li])))

        with torch.no_grad():
            pm = logit_diff(model(**toks_x).logits, cid, wid)
        for h in handles:
            h.remove()

        cm_sum += cm
        xm_sum += xm
        pm_sum += pm
        del clean_c, corrupt_c, corrections, toks_c, toks_x

    n = len(pairs)
    return {"clean": cm_sum/n, "corrupt": xm_sum/n, "patched": pm_sum/n}


def _correction_hook(corr):
    def h(mod, inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            fixed = hs + corr[:, :hs.shape[1], :]
            return (fixed,) + out[1:]
        return out + corr[:, :out.shape[1], :]
    return h


def run(model, tokenizer, cfg, device, smoke=False):
    seeds = cfg["seed_list"]
    if smoke:
        seeds = [seeds[0]]

    # seed global RNGs for reproducibility of torch/numpy ops. Per-seed
    # data and baseline sampling use explicit seeds on top of this.
    set_seed(int(cfg.get("seed", seeds[0])))

    ds_size = cfg.get("dataset_size", 24)
    disc_n = cfg.get("discovery_examples", 12)
    ig_steps = cfg.get("ig_steps", 4)
    sparsity = cfg.get("sparsity", 0.15)
    n_rand = cfg.get("n_random_baselines", 10)
    corr_base = cfg.get("corruption_base", "mild")
    corr_rob = cfg.get("corruption_robustness", ["mild", "strong"])
    out_dir = cfg.get("output_root", "runs")

    if smoke:
        ds_size, disc_n, ig_steps, n_rand = 4, 2, 2, 2
        corr_rob = []

    os.makedirs(out_dir, exist_ok=True)

    nl = n_layers(model)
    results = []
    circuits = []
    t0 = time.time()
    peak = mem_mb()

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"seed={seed}")
        print(f"{'='*50}")

        # separate discovery and evaluation sets to avoid overfitting
        disc_pairs = make_pairs(disc_n, corruption=corr_base, seed=seed)
        eval_pairs = make_pairs(ds_size, corruption=corr_base, seed=seed + 1000)

        # --- attribution ---
        print(f"\nattribution (ig_steps={ig_steps})")
        td = time.time()
        scores = score_all_layers(model, tokenizer, disc_pairs, ig_steps=ig_steps, device=device)
        selected = pick_top_layers(scores, sparsity)
        print(f"took {time.time()-td:.1f}s")

        circuits.append(selected)

        # --- faithfulness ---
        print(f"\nevaluating circuit {selected}")
        m = patch_and_measure(model, tokenizer, eval_pairs, selected, device)

        faith = safe_div(m["patched"] - m["corrupt"], m["clean"] - m["corrupt"])

        # sanity check: if this is near perfect, we probably patched full state by accident
        if faith >= 0.99:
            raise ValueError(
                f"faith={faith:.4f} is too high; check that this is contribution patching"
            )

        print(f"  clean={m['clean']:.4f} corrupt={m['corrupt']:.4f} patched={m['patched']:.4f}")
        print(f"  faithfulness={faith:.4f}")

        # --- random baselines ---
        # random masks can accidentally hit useful layers, so average a few trials
        print(f"\nrandom baselines (n={n_rand})")
        rng = random.Random(seed + 5000)
        rfaiths = []
        rlayers = []
        k = len(selected)

        for t in range(n_rand):
            rl = sorted(rng.sample(range(nl), k=k))
            rm = patch_and_measure(model, tokenizer, eval_pairs, rl, device)
            rf = safe_div(rm["patched"] - rm["corrupt"], rm["clean"] - rm["corrupt"])
            rfaiths.append(rf)
            rlayers.append(rl)
            print(f"  trial {t}: {rl} -> {rf:.4f}")
            cleanup()

        ravg = mean(rfaiths) if rfaiths else 0.0
        print(f"  avg random faith={ravg:.4f}")

        # a discovered circuit only means something if it beats a same-sparsity
        # random mask. Flag it loudly when it does not, so weak seeds are not
        # hidden inside an aggregate average.
        beats_random = faith > ravg
        if not beats_random:
            print(f"  WARNING: circuit faith {faith:.4f} <= random avg {ravg:.4f}; "
                  f"this circuit is not better than random")

        # --- robustness ---
        rob = {}
        for ct in corr_rob:
            print(f"\nrobustness ({ct})")
            rp = make_pairs(ds_size, corruption=ct, seed=seed + 2000)
            rm = patch_and_measure(model, tokenizer, rp, selected, device)
            rf = safe_div(rm["patched"] - rm["corrupt"], rm["clean"] - rm["corrupt"])
            rob[ct] = {"faith": rf, "clean": rm["clean"], "corrupt": rm["corrupt"], "patched": rm["patched"]}
            print(f"  faith={rf:.4f}")

        peak = max(peak, mem_mb())

        sr = {
            "seed": seed,
            "circuit": selected,
            "scores": scores.tolist(),
            "clean": m["clean"], "corrupt": m["corrupt"], "patched": m["patched"],
            "faith": faith,
            "beats_random": beats_random,
            "rand_faiths": rfaiths, "rand_layers": rlayers, "rand_avg": ravg,
            "robustness": rob,
            "disc_time": time.time() - td,
        }
        results.append(sr)

        with open(os.path.join(out_dir, f"seed_{seed}.json"), "w") as f:
            json.dump(sr, f, indent=2)
        print(f"\nsaved seed_{seed}.json")
        cleanup()

    # --- aggregate ---
    total = time.time() - t0
    print(f"\n{'='*50}")
    print(f"done ({len(seeds)} seeds, {total:.1f}s)")
    print(f"{'='*50}")

    avg_f = mean([r["faith"] for r in results])
    avg_r = mean([r["rand_avg"] for r in results])

    stab = []
    for i in range(len(circuits)):
        for j in range(i+1, len(circuits)):
            stab.append(jaccard(circuits[i], circuits[j]))
    avg_stab = mean(stab) if stab else 1.0

    rob_agg = {}
    for ct in corr_rob:
        fs = [r["robustness"][ct]["faith"] for r in results if ct in r.get("robustness", {})]
        if fs:
            rob_agg[ct] = mean(fs)

    agg = {
        "config": {
            "model": cfg["model_name"], "dtype": cfg["dtype"],
            "batch_size": cfg["batch_size"], "dataset_size": ds_size,
            "disc_examples": disc_n, "ig_steps": ig_steps,
            "sparsity": sparsity, "n_random": n_rand, "seeds": seeds,
        },
        "circuits": {r["seed"]: r["circuit"] for r in results},
        "avg_faith": avg_f, "avg_rand_faith": avg_r,
        "gap": avg_f - avg_r, "stability": avg_stab,
        "robustness": rob_agg,
        "time_s": total, "peak_mb": peak,
        "per_seed": results,
    }

    with open(os.path.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\ncircuit faith: {avg_f:.4f}")
    print(f"random faith:  {avg_r:.4f}")
    print(f"gap:           {avg_f - avg_r:.4f}")
    print(f"stability:     {avg_stab:.4f}")
    for ct, v in rob_agg.items():
        print(f"robustness({ct}): {v:.4f}")
    print(f"peak mem:      {peak:.0f} MB")
    print(f"total:         {total:.1f}s")

    print("\nrun 'python scripts/plot_results.py' to generate attribution plot")

    return agg
