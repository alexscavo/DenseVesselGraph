"""Plot held-out patch distributions from an existing evaluate.py export."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def values(rows, name):
    result = np.asarray([float(row[name]) if row[name] else np.nan for row in rows], dtype=float)
    return result[np.isfinite(result)]


def fixed_f1(rows, prefix):
    tp = values(rows, prefix + "_tp")
    fp = values(rows, prefix + "_fp")
    fn = values(rows, prefix + "_fn")
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def count_scatter(axis, rows, kind):
    target = values(rows, "target_" + kind)
    predicted = values(rows, "predicted_" + kind)
    axis.scatter(target, predicted, s=7, alpha=0.15, color="#1f77b4", rasterized=True)
    limit = max(float(target.max()), float(predicted.max())) * 1.03
    axis.plot([0, limit], [0, limit], "k--", lw=1, label="exact count")
    axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="GT count", ylabel="Predicted count")
    axis.set_title(kind.replace("_", " ").capitalize())
    axis.legend(loc="upper left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.evaluation_dir / "summary.json").read_text())
    with (args.evaluation_dir / "per-patch-metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != summary["samples"]:
        raise ValueError("Per-patch CSV and summary sample counts differ")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")
    axes[0, 0].hist(fixed_f1(rows, "node"), bins=30, alpha=0.65,
                    label=f"Node (micro {summary['node_f1']:.3f})")
    axes[0, 0].hist(fixed_f1(rows, "edge"), bins=30, alpha=0.65,
                    label=f"Edge (micro {summary['edge_f1']:.3f})")
    axes[0, 0].set(title="Patch node / edge F1", xlabel="F1", ylabel="Patches")
    axes[0, 0].legend()

    axes[0, 1].hist(100 * values(rows, "branch_f1"), bins=30, color="#d62728")
    axes[0, 1].set(title=f"Branch F1 (micro {100*summary['branch_f1']:.3f}%)",
                   xlabel="Patch Branch F1 (%)", ylabel="Patches")

    for name, color in (("node_mAP", "#1f77b4"), ("edge_mAP", "#ff7f0e"),
                        ("node_mAR", "#2ca02c"), ("edge_mAR", "#9467bd")):
        axes[0, 2].hist(values(rows, name), bins=30, histtype="step", lw=1.5,
                        color=color, label=f"{name} ({summary[name]:.3f})")
    axes[0, 2].set(title="Detection AP / AR", xlabel="Per-patch value", ylabel="Patches")
    axes[0, 2].legend()

    for axis, name, title in (
        (axes[1, 0], "smd", "Street mover distance"),
        (axes[1, 1], "beta0_absolute_error", "Component count error |β₀|"),
        (axes[1, 2], "beta1_absolute_error", "Cycle count error |β₁|"),
    ):
        axis.hist(values(rows, name), bins=40, color="#4682b4")
        axis.set(title=f"{title} (mean {summary[name]:.3f})", xlabel="Per-patch value", ylabel="Patches")

    count_scatter(axes[2, 0], rows, "nodes")
    count_scatter(axes[2, 1], rows, "edges")
    count_scatter(axes[2, 2], rows, "beta1")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(f"IXI held-out test • {summary['samples']:,} patches • validation-selected checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
