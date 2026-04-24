#!/usr/bin/env python3
# make the attribution plot from saved results
# run this after the main pipeline finishes

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agg_path = os.path.join(base, "runs", "aggregate.json")

    if not os.path.exists(agg_path):
        print(f"no aggregate.json at {agg_path}")
        sys.exit(1)

    with open(agg_path) as f:
        agg = json.load(f)

    results = agg["per_seed"]
    nl = len(results[0]["scores"])

    # use first seed's circuit for the red guide lines
    circuit = results[0]["circuit"]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2563eb", "#d97706", "#059669"]
    markers = ["o", "s", "^"]

    for i, r in enumerate(results):
        sc = r["scores"]
        ax.plot(range(nl), sc, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                label=f"seed {r['seed']}", linewidth=1.8, markersize=6, alpha=0.9)

    # mark circuit layers
    for li in circuit:
        ax.axvline(x=li, color="#dc2626", alpha=0.35, linestyle="--", linewidth=1.5)
        ax.text(li + 0.15, max(max(r["scores"]) for r in results) + 0.5,
                f"L{li}", color="#dc2626", fontsize=9, fontweight="bold")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Attribution Score (EAP-IG)", fontsize=12)
    ax.set_title("Per-Layer Attribution Scores for LLaMA 3.2 1B", fontsize=13, fontweight="bold")
    ax.set_xticks(range(nl))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.5, nl - 0.5)

    out = os.path.join(base, "reports", "attribution.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
