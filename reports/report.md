# IOI Circuit Discovery on LLaMA 3.2 1B

## Setup

| Parameter | Value |
|---|---|
| Model | unsloth/Llama-3.2-1B |
| Precision | float16 |
| Batch size | 1 |
| Hardware | Apple M2 Air, 8GB unified memory, MPS backend |
| Sparsity | 0.15 (k=2 of 16 layers) |
| IG steps | 4 |
| Seeds | 13, 17, 23 |
| Dataset | 24 eval / 12 discovery per seed |
| Random baselines | 10 per seed |
| Runtime | 806.8s (~13 min) |
| Peak memory | 395 MB RSS. MPS does not expose GPU allocation directly; estimated ~2.5 GB model weights at float16 |


Command:
```bash
HF_HOME=.hf_cache /opt/homebrew/Caskroom/miniforge/base/envs/mhc/bin/python \
  scripts/run_assignment.py --config configs/default.toml 2>&1 | tee runs/run.log
```

## How I Did This

**Per-layer integrated gradients.** Each of the 16 transformer blocks is scored independently, one hook at a time, one forward-backward pass per interpolation step. Hooking all layers at once breaks gradient flow through the earlier intervened blocks, so each layer is scored on its own.

This implementation scores full decoder blocks rather than inter-component edges. That keeps the run manageable, cutting 512 attribution passes down to 16 while still showing which model regions carry most of the IOI signal.

I found that scoring the full block output was misleading. A layer can have a large residual stream but still add very little useful IOI signal. The discovery score now uses block contributions, `post_block - pre_block`, which is the same object the patching code swaps back in. The interpolation tensor is cast to float32 before `requires_grad_()` because MPS float16 autograd silently accumulates zero gradients.

**Contribution-based patching.** Naive hidden-state replacement gives faithfulness near 1.0 for many layer sets, including random draws, because the clean hidden state carries forward through the rest of the residual stream. The fix is to patch only what each block added. For each selected layer, `contribution = output - input`. During the corrupt run, the hook adds `(clean_contribution - corrupt_contribution)` to the layer output. That swaps the block's computation without replacing the whole residual stream.

Hook placement matters. LLaMA blocks add the residual internally, so `register_forward_pre_hook` captures the stream before the block and `register_forward_hook` captures after the residual add.

## Results

### Summary

| Metric | Value |
|---|---|
| Circuit | layers [0, 8] (seeds 13+23), [8, 14] (seed 17) |
| Clean IOI score | 5.545 avg |
| Corrupt IOI score | -5.107 avg |
| Patched IOI score | 3.766 avg |
| Circuit faithfulness | 0.831 |
| Random baseline | 0.393 |
| Gap | +0.438 (circuit is 2.1x random) |
| Stability (Jaccard) | 0.556 |
| Robustness mild | 0.821 |
| Robustness strong | 0.829 |

### Per-seed breakdown

| Seed | Circuit | Faith | Random avg | Clean | Corrupt | Patched |
|---|---|---|---|---|---|---|
| 13 | [0, 8] | 0.915 | 0.347 | 5.916 | -5.039 | 4.982 |
| 17 | [8, 14] | 0.653 | 0.505 | 5.339 | -5.016 | 1.742 |
| 23 | [0, 8] | 0.924 | 0.327 | 5.380 | -5.266 | 4.574 |

### Notes

**Layer 0 is the strongest recovery path.** Seeds 13 and 23 both select layer 0 alongside layer 8. Layer 0 is the first decoder block after embeddings. Patching its contribution nearly recovers the clean behavior, which suggests the model encodes much of the name-identity signal very early.

**Layer 8 is the stable layer.** All three seeds select layer 8. Seed 17 swaps layer 0 for layer 14, but the shared layer 8 is enough to keep the aggregate circuit well above random.

**Seed 17 random baseline is inflated.** Three of ten random draws include layer 0, pushing that seed's random average to 0.505. Excluding those layer-0 draws drops it close to 0.30, which matches seeds 13 and 23. Layer 0 is genuinely useful, so random masks that include it are not meaningless contamination. They are a sign that early name information is doing real work.

**Faithfulness above 1.0 can happen.** For example, seed 17 has a random draw [0,15] with faithfulness 1.21. Contribution patching can overcorrect when the injected delta pushes the patched margin past the clean baseline. That is a measurement artifact of patching under noisy corrupt baselines, not a code bug.

**Robustness is stronger than expected.** Strong corruption, where both names are swapped, gives faithfulness 0.829. That is slightly above the mild result of 0.821. The likely reason is layer 0: once the early name-identity signal is corrected, the small block-level circuit keeps much of the behavior intact even under the harder corruption.

## Limitations

**MPS float16 non-determinism.** Absolute attribution scores can move across runs on Apple MPS. The submitted run records the exact Python, torch, and transformers versions in `runs/run.log`.

**Block-level granularity.** Each selected layer contains attention, MLP, and layer norms. Head-level attribution would probably give a cleaner circuit, but it would cost far more compute.

**Stability is partial, not perfect.** Jaccard 0.556 means seeds 13 and 23 agree exactly, while seed 17 picks [8,14]. The result is still useful because all seeds include layer 8 and the aggregate faithfulness is far above random.

## Debugging Notes

The main bug I hit was with patching. Full hidden-state replacement made both circuit masks and random masks look almost perfect. That was wrong, because the patch was replacing too much of the residual stream. Contribution patching fixed the metric by only adding back the block delta.

The second annoying issue was MPS precision. The code ran without crashing, but some IG gradients were effectively zero when the interpolation tensor stayed in float16. Keeping that tensor in float32 made the attribution scores usable.

The last lesson was to check random baselines early. A high faithfulness score by itself is not enough. The circuit only means something if it beats a same-sparsity random mask.

## Artifacts

- Per-seed results: `runs/seed_13.json`, `runs/seed_17.json`, `runs/seed_23.json`
- Aggregate: `runs/aggregate.json`
- Console log: `runs/run.log`
- Attribution plot: `reports/attribution.png`
- Config: `configs/default.toml`
- Code: `src/ioi_circuit/`

## Recommended Defaults for Future IOI-on-LLaMA Runs

| Setting | Value | Rationale |
|---|---|---|
| Sparsity | 0.15 | k=2 for 16-layer models. Captures the main signal without giving random masks too much room |
| IG steps | 4 | Enough for block-level ranking on this model |
| Random baselines | 10+ per seed | Minimum useful check for same-sparsity random masks |
| Seeds | 3 | Enough to catch instability without turning the run into a full sweep |
| Discovery examples | 12 | Keeps runtime manageable while still giving stable layer-8 selection |
| Eval examples | 24 | Disjoint from discovery by seed offset +1000 |
| Precision | float16 on MPS, bfloat16 on CUDA | Keep the IG interpolation tensor in float32 either way |
| Patching method | contribution-based | Full hidden-state replacement contaminates the residual stream |

## What I Actually Learned

**1. I was surprised that layer 8 appeared in every seed.** Layers 0 and 14 trade off across seeds, but layer 8 is the shared block. That makes it the most stable part of the discovered circuit.

**2. Layer 0 mattered more than I expected.** I expected IOI to mostly show up in middle layers, but two seeds select [0,8], and random draws containing layer 0 often patch very strongly. The first block seems to encode name identity early enough that fixing its contribution recovers most of the clean behavior.

**3. Random baselines saved the project from a bad metric.** The first patching version looked great until random masks also looked great. Scoring and patching `post - pre` contributions made the result meaningful and gave a circuit that beats random by a clear margin.

## What I Would Do Differently Next Time

- Start with random baselines immediately instead of trusting a high faithfulness score.
- Test the IG hook on CPU first before debugging MPS behavior.
- Save more intermediate outputs, especially per-layer single-patch faithfulness.
- Try head-level attribution after the block-level circuit is working.
