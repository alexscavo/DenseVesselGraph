"""Plot the IXI 300-epoch continuation from its logs and JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESUME_RUN = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_q256_100ep_20260923_resume")
EXTEND_RUN = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_q256_extend200_20260923")
LOGS = (
    Path("/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/gnbm-train-93539.out"),
    Path("/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/gnbm-train-101450.out"),
    Path("/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/gnbm-train-113928.out"),
)
OUT = Path(__file__).resolve().parent


def rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    run_dirs = (RESUME_RUN, EXTEND_RUN)
    performance = sorted(
        {int(row["epoch"]): row for run in run_dirs
         for row in rows(run / "performance.jsonl")}.values(),
        key=lambda row: row["epoch"],
    )
    validation = sorted(
        {int(row["epoch"]): row for run in run_dirs
         for row in rows(run / "validation-metrics.jsonl")}.values(),
        key=lambda row: row["epoch"],
    )
    log = "\n".join(path.read_text() for path in LOGS if path.is_file())
    training_loss = sorted({int(epoch): float(value) for epoch, value in
                            re.findall(r"^epoch=(\d+)/\d+ total=([0-9.eE+-]+)", log, re.M)}.items())
    validation_loss = sorted({int(epoch): float(value) for epoch, value in
                              re.findall(r"^validation epoch=(\d+) total=([0-9.eE+-]+)", log, re.M)}.items())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for points, label in ((training_loss, "train loss"), (validation_loss, "validation loss")):
        if points:
            axes[0, 0].plot(*zip(*points), marker="o", ms=2, label=label)
    axes[0, 0].set_title("Optimization")
    if training_loss or validation_loss:
        axes[0, 0].legend()
    branch_axis = axes[0, 1].twinx()
    for key, label in (("node_f1", "Node F1"), ("edge_f1", "Edge F1")):
        points = [(row["epoch"], row[key]) for row in validation
                  if row.get(key) is not None]
        if points:
            axes[0, 1].plot(*zip(*points), marker="o", label=label)
    branch_points = [(row["epoch"], 100.0 * row["branch_f1"]) for row in validation
                     if row.get("branch_f1") is not None]
    if branch_points:
        branch_axis.plot(*zip(*branch_points), marker="o", color="tab:red",
                         label="Branch F1 (%)")
        branch_axis.set_ylim(0, max(0.1, max(value for _, value in branch_points) * 1.25))
        branch_axis.set_ylabel("Branch F1 (%)")
    axes[0, 1].set_title("Held-out IXI graph metrics")
    axes[0, 1].set_ylim(0, 1)
    if validation:
        handles, labels = axes[0, 1].get_legend_handles_labels()
        branch_handles, branch_labels = branch_axis.get_legend_handles_labels()
        axes[0, 1].legend(handles + branch_handles, labels + branch_labels)
    if performance:
        axes[1, 0].plot([row["epoch"] for row in performance],
                        [row["samples_per_second"] for row in performance], marker=".")
        axes[1, 1].plot([row["epoch"] for row in performance],
                        [row["peak_allocated_gib"] for row in performance], marker=".")
    axes[1, 0].set_title("Training throughput (samples/s)")
    axes[1, 1].set_title("Peak allocated GPU memory (GiB)")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.25)
    fig.suptitle("IXI-only adaptive-GT training, epochs 1–300 target, 256 queries, two H100s")
    fig.savefig(OUT / "progress.png", dpi=160)
    plt.close(fig)
    (OUT / "progress_summary.json").write_text(json.dumps({
        "training_epochs_logged": len(training_loss),
        "validation_loss_epochs_logged": len(validation_loss),
        "graph_metric_epochs_logged": len(validation),
        "performance_epochs_logged": len(performance),
        "latest_epoch": max((epoch for epoch, _ in training_loss), default=0),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
