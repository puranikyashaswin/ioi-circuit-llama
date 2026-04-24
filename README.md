# IOI Circuit Discovery on LLaMA 3.2 1B

Per-layer EAP-IG circuit discovery for indirect object identification on `unsloth/Llama-3.2-1B`, evaluated with contribution-based patching.

## Results

| Metric | Value |
|---|---|
| Circuit | layers [0, 8] (seeds 13, 23) / [8, 14] (seed 17) |
| Circuit faithfulness | 0.831 |
| Random baseline | 0.393 |
| Gap | +0.438 (2.1x random) |
| Stability (Jaccard) | 0.556 |
| Robustness mild | 0.821 |
| Robustness strong | 0.829 |
| Runtime | ~13 min on Apple M2 8GB |
| Peak memory | ~395 MB RSS |

Full report: [`reports/report.md`](reports/report.md)

## What I Learned

The discovery code was not the hardest part. The patching metric was.

The first version used full hidden-state replacement and gave faithfulness near 1.0 even for random layer masks. That looked good for about five minutes, then it became obvious the metric was broken: replacing the whole hidden state lets the clean residual stream carry forward, so the corrupt prompt barely matters after the patched layer.

The fix was to patch only the block contribution, `output - input`. After that, random masks dropped to 0.393 while the discovered circuit reached 0.831.

MPS also caused one annoying issue: float16 autograd could silently give zero gradients during IG. The interpolation tensor is kept in float32 before `requires_grad_()` because of that.

The result that surprised me most was layer 0. I expected middle layers to dominate IOI, but layer 0 appears in two of three seeds and random masks containing it often patch strongly. Layer 8 was the most stable layer across all seeds.

Requirements: Python 3.11+, ~3 GB RAM for model weights.

## Setup

```bash
pip install -e ".[plot]"
```

## Usage

```bash
# smoke test (~2 min)
python scripts/run_assignment.py --config configs/default.toml --smoke-test

# full 3-seed run (~13 min on M2)
HF_HOME=.hf_cache python scripts/run_assignment.py --config configs/default.toml 2>&1 | tee runs/run.log

# attribution plot
python scripts/plot_results.py
```

Results go in `runs/`. Plot goes in `reports/`.
