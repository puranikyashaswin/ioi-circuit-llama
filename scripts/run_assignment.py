#!/usr/bin/env python3

import sys
import os
import argparse
import time
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_cfg(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    import torch
    import transformers

    print(f"model: {cfg['model_name']}")
    print(f"dtype: {cfg['dtype']}, seeds: {cfg['seed_list']}")
    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}, transformers: {transformers.__version__}")
    print(f"smoke={args.smoke_test}")
    print()

    from ioi_circuit.utils import load_model, mem_mb
    model, tokenizer, device, dtype = load_model(
        cfg["model_name"], cfg["device"], cfg["dtype"]
    )
    print(f"mem after load: {mem_mb():.0f} MB\n")

    from ioi_circuit.evaluation import run
    t0 = time.time()
    agg = run(model, tokenizer, cfg, device, smoke=args.smoke_test)

    print(f"\nfinished in {time.time()-t0:.1f}s")
    if args.smoke_test:
        print("smoke test ok, patching logic looks sane")


if __name__ == "__main__":
    main()
