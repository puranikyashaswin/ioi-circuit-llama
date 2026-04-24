import torch
import numpy as np
from .utils import get_layers, n_layers, cleanup


def score_all_layers(model, tokenizer, pairs, ig_steps=4, device="mps"):
    """Score each block by integrating gradients over its residual contribution.
    Uses post minus pre so discovery ranks the same thing patching
    later swaps back in.
    """
    nl = n_layers(model)
    layers = get_layers(model)
    mdtype = next(model.parameters()).dtype
    scores = np.zeros(nl)

    # cache contributions up front so discovery and eval use the same delta
    print(f"caching contributions ({nl} layers)...")
    clean_contribs = {li: [] for li in range(nl)}
    corrupt_contribs = {li: [] for li in range(nl)}

    for pair in pairs:
        ca = _grab_contribs(model, tokenizer, pair["clean"], layers, device)
        xa = _grab_contribs(model, tokenizer, pair["corrupt"], layers, device)
        for li in range(nl):
            clean_contribs[li].append(ca[li])
            corrupt_contribs[li].append(xa[li])

    cids, wids = [], []
    for pair in pairs:
        cids.append(tokenizer.encode(" " + pair["correct_name"], add_special_tokens=False)[0])
        wids.append(tokenizer.encode(" " + pair["wrong_name"], add_special_tokens=False)[0])

    alphas = np.linspace(0, 1, ig_steps)

    # score one layer at a time, hooking all at once breaks gradients
    for li in range(nl):
        print(f"  layer {li}/{nl-1}", end=" ", flush=True)
        layer_score = 0.0

        for pi, pair in enumerate(pairs):
            ca_li = clean_contribs[li][pi]
            xa_li = corrupt_contribs[li][pi]
            cid = torch.tensor([cids[pi]], device=device)
            wid = torch.tensor([wids[pi]], device=device)

            pair_score = 0.0
            for alpha in alphas:
                holder = []
                hook = _ig_hook_fn(xa_li, ca_li, float(alpha), holder, mdtype)
                handle = layers[li].register_forward_hook(hook)

                toks = tokenizer(pair["corrupt"], return_tensors="pt").to(device)
                with torch.enable_grad():
                    out = model(**toks)
                    last = out.logits[:, -1, :]
                    diff = (last[:, cid[0]] - last[:, wid[0]]).sum()
                    diff.backward()

                handle.remove()

                if holder and holder[0].grad is not None:
                    g = holder[0].grad.detach().cpu().float()
                    d = ca_li.cpu().float() - xa_li.cpu().float()
                    # keep sign, layers moving the wrong way should not rank high
                    pair_score += (g * d).sum().item()
                else:
                    # this was usually an MPS float16 grad issue while debugging
                    print(f"[no grad l={li} a={alpha:.1f}]", end=" ")

                del holder, toks, out
                model.zero_grad(set_to_none=True)

            layer_score += pair_score / ig_steps

        scores[li] = layer_score / len(pairs)
        print(f"-> {scores[li]:.4f}")
        cleanup()

    return scores


def _ig_hook_fn(xa, ca, alpha, holder, mdtype):
    """Hook factory for IG over a single block contribution interpolation.
    Rebuilds output as pre plus interpolated contribution so the residual
    stream stays intact.
    """
    def hook(mod, inp, out):
        pre = (inp[0] if isinstance(inp, tuple) else inp).detach()
        hs = out[0] if isinstance(out, tuple) else out
        dev = hs.device

        interp = xa.float() + alpha * (ca.float() - xa.float())

        # MPS plus float16 autograd silently gave zero grads, so keep IG in float32
        interp = interp.detach().clone().float().requires_grad_(True)
        interp.retain_grad()
        holder.append(interp)

        # rebuild normal block output from frozen pre-activation plus IG delta
        fixed = pre.float() + interp.to(dev).float()[:, :pre.shape[1], :]
        fixed = fixed.to(mdtype)
        if isinstance(out, tuple):
            return (fixed,) + out[1:]
        return fixed
    return hook


def _grab_contribs(model, tokenizer, prompt, layers, device):
    # grab block delta, discovery needs the same object as patching
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


def pick_top_layers(scores, sparsity):
    k = max(1, round(len(scores) * sparsity))
    candidates = [(i, s) for i, s in enumerate(scores) if s > 0]
    if len(candidates) >= k:
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = sorted([i for i, _ in candidates[:k]])
    else:
        top = sorted(np.argsort(scores)[-k:].tolist())
    print(f"selected layers {top} (k={k})")
    return top
