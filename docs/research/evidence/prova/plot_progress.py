"""Plot and print progress of the current IXI training run."""

from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN = Path(
    "/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/"
    "ixi_q256_types_300ep_20260924"
)

LOG = Path(
    "/lustre/fswork/projects/rech/vnc/upz25mj/logs/"
    "graph-native-betti-matching/gnbm-train-145201.out"
)

OUT = Path("/lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching/docs/research/evidence/prova")


def rows(path: Path):
    """Read a JSONL file."""
    if not path.is_file():
        return []

    result = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    return result


def main():

    # ---------------------------------------------------------
    # Check paths
    # ---------------------------------------------------------

    if not RUN.is_dir():
        raise FileNotFoundError(f"Run directory not found: {RUN}")

    if not LOG.is_file():
        raise FileNotFoundError(f"Log file not found: {LOG}")

    # ---------------------------------------------------------
    # JSONL metrics
    # ---------------------------------------------------------

    performance = sorted(
        rows(RUN / "performance.jsonl"),
        key=lambda row: int(row["epoch"]),
    )

    validation = sorted(
        rows(RUN / "validation-metrics.jsonl"),
        key=lambda row: int(row["epoch"]),
    )

    # ---------------------------------------------------------
    # Training / validation losses from Slurm log
    # ---------------------------------------------------------

    log = LOG.read_text(errors="replace")

    training_loss = sorted(
        {
            int(epoch): float(value)
            for epoch, value in re.findall(
                r"epoch=(\d+)/\d+.*?\btotal=([0-9.eE+-]+)",
                log,
            )
        }.items()
    )

    validation_loss = sorted(
        {
            int(epoch): float(value)
            for epoch, value in re.findall(
                r"validation\s+epoch=(\d+).*?\btotal=([0-9.eE+-]+)",
                log,
                re.IGNORECASE,
            )
        }.items()
    )

    # ---------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("CURRENT RUN")
    print("=" * 72)

    print(f"Run directory: {RUN}")
    print(f"Training log:  {LOG}")

    print()
    print("Data found:")
    print(f"  Training-loss epochs:   {len(training_loss)}")
    print(f"  Validation-loss epochs: {len(validation_loss)}")
    print(f"  Graph-metric epochs:    {len(validation)}")
    print(f"  Performance epochs:     {len(performance)}")

    if training_loss:
        epoch, value = training_loss[-1]

        print()
        print("Latest training loss:")
        print(f"  epoch {epoch}: {value:.6f}")
    else:
        print()
        print("WARNING: no training-loss entries were detected in the log.")

    if validation_loss:
        epoch, value = validation_loss[-1]

        print()
        print("Latest validation loss:")
        print(f"  epoch {epoch}: {value:.6f}")

    if validation:
        latest = validation[-1]

        print()
        print(f"Latest graph metrics (epoch {latest['epoch']}):")

        for key in (
            "node_f1",
            "edge_f1",
            "branch_f1",
        ):
            if latest.get(key) is not None:
                value = float(latest[key])

                if key == "branch_f1":
                    print(f"  {key:12s}: {100.0 * value:.4f}%")
                else:
                    print(f"  {key:12s}: {value:.6f}")

    if performance:
        latest = performance[-1]

        print()
        print(f"Latest performance (epoch {latest['epoch']}):")

        if latest.get("samples_per_second") is not None:
            print(
                f"  throughput: "
                f"{float(latest['samples_per_second']):.2f} samples/s"
            )

        if latest.get("peak_allocated_gib") is not None:
            print(
                f"  GPU memory: "
                f"{float(latest['peak_allocated_gib']):.2f} GiB"
            )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
        layout="constrained",
    )

    # ---------------------------------------------------------
    # 1. Optimization
    # ---------------------------------------------------------

    if training_loss:
        epochs, values = zip(*training_loss)

        axes[0, 0].plot(
            epochs,
            values,
            marker="o",
            markersize=2,
            color="tab:blue",
            label="Train loss",
        )

    if validation_loss:
        epochs, values = zip(*validation_loss)

        axes[0, 0].plot(
            epochs,
            values,
            marker="o",
            markersize=3,
            color="tab:orange",
            label="Validation loss",
        )

    axes[0, 0].set_title("Optimization")

    if training_loss or validation_loss:
        axes[0, 0].legend()

    # ---------------------------------------------------------
    # 2. Graph metrics
    # ---------------------------------------------------------

    branch_axis = axes[0, 1].twinx()

    metric_config = (
        ("node_f1", "Node F1", "tab:blue"),
        ("edge_f1", "Edge F1", "tab:orange"),
    )

    for key, label, color in metric_config:

        points = [
            (
                int(row["epoch"]),
                float(row[key]),
            )
            for row in validation
            if row.get(key) is not None
        ]

        if points:
            epochs, values = zip(*points)

            axes[0, 1].plot(
                epochs,
                values,
                marker="o",
                markersize=4,
                color=color,
                label=label,
            )

    branch_points = [
        (
            int(row["epoch"]),
            100.0 * float(row["branch_f1"]),
        )
        for row in validation
        if row.get("branch_f1") is not None
    ]

    if branch_points:
        epochs, values = zip(*branch_points)

        branch_axis.plot(
            epochs,
            values,
            marker="o",
            markersize=4,
            color="tab:red",
            label="Branch F1 (%)",
        )

        branch_axis.set_ylabel("Branch F1 (%)")

        branch_axis.set_ylim(
            0,
            max(
                0.1,
                max(values) * 1.25,
            ),
        )

    axes[0, 1].set_title("Held-out IXI graph metrics")
    axes[0, 1].set_ylabel("Node / Edge F1")
    axes[0, 1].set_ylim(0, 1)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    branch_handles, branch_labels = branch_axis.get_legend_handles_labels()

    if handles or branch_handles:
        axes[0, 1].legend(
            handles + branch_handles,
            labels + branch_labels,
            loc="best",
        )

    # ---------------------------------------------------------
    # 3. Training throughput
    # ---------------------------------------------------------

    throughput = [
        (
            int(row["epoch"]),
            float(row["samples_per_second"]),
        )
        for row in performance
        if row.get("samples_per_second") is not None
    ]

    if throughput:
        epochs, values = zip(*throughput)

        axes[1, 0].plot(
            epochs,
            values,
            marker=".",
            color="tab:blue",
        )

    axes[1, 0].set_title("Training throughput (samples/s)")

    # ---------------------------------------------------------
    # 4. GPU memory
    # ---------------------------------------------------------

    memory = [
        (
            int(row["epoch"]),
            float(row["peak_allocated_gib"]),
        )
        for row in performance
        if row.get("peak_allocated_gib") is not None
    ]

    if memory:
        epochs, values = zip(*memory)

        axes[1, 1].plot(
            epochs,
            values,
            marker=".",
            color="tab:green",
        )

    axes[1, 1].set_title("Peak allocated GPU memory (GiB)")

    # ---------------------------------------------------------
    # Common formatting
    # ---------------------------------------------------------

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)

    fig.suptitle(
        "IXI adaptive-GT training — types, 300 epochs, 256 queries"
    )

    output_plot = OUT / "progress.png"

    fig.savefig(
        output_plot,
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # JSON summary
    # ---------------------------------------------------------

    all_epochs = []

    all_epochs.extend(epoch for epoch, _ in training_loss)
    all_epochs.extend(epoch for epoch, _ in validation_loss)
    all_epochs.extend(
        int(row["epoch"])
        for row in validation
        if row.get("epoch") is not None
    )
    all_epochs.extend(
        int(row["epoch"])
        for row in performance
        if row.get("epoch") is not None
    )

    latest_epoch = max(all_epochs, default=0)

    summary = {
        "run": str(RUN),
        "log": str(LOG),
        "training_epochs_logged": len(training_loss),
        "validation_loss_epochs_logged": len(validation_loss),
        "graph_metric_epochs_logged": len(validation),
        "performance_epochs_logged": len(performance),
        "latest_epoch": latest_epoch,
    }

    output_summary = OUT / "progress_summary.json"

    output_summary.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print()
    print("=" * 72)
    print(f"Latest epoch detected: {latest_epoch}")
    print(f"Plot saved to:         {output_plot}")
    print(f"Summary saved to:      {output_summary}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()