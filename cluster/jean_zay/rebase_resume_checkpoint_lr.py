#!/usr/bin/env python3
"""Prepare a full-state resume checkpoint with a new polynomial LR horizon."""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-epochs", required=True, type=int)
    parser.add_argument("--iterations-per-epoch", required=True, type=int)
    parser.add_argument("--warmup-epochs", required=True, type=float)
    parser.add_argument("--power", required=True, type=float)
    parser.add_argument("--target-lr", required=True, type=float)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error("checkpoint does not exist: {}".format(args.checkpoint))
    if args.output.exists():
        parser.error("output already exists: {}".format(args.output))
    if min(args.total_epochs, args.iterations_per_epoch) <= 0:
        parser.error("total epochs and iterations per epoch must be positive")
    if args.warmup_epochs < 0 or args.power <= 0 or args.target_lr <= 0:
        parser.error("warmup must be nonnegative; power and target LR must be positive")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    epoch = int(checkpoint["epoch"])
    scheduler = checkpoint["scheduler"]
    current_iteration = int(scheduler["last_epoch"])
    warmup_iterations = args.warmup_epochs * args.iterations_per_epoch
    training_iterations = args.total_epochs * args.iterations_per_epoch
    if current_iteration >= training_iterations:
        parser.error("checkpoint scheduler is already at or beyond the requested horizon")
    if current_iteration < warmup_iterations:
        multiplier = (current_iteration / max(1.0, warmup_iterations))
    else:
        progress = (current_iteration - warmup_iterations) / training_iterations
        multiplier = max(0.0, 1.0 - progress) ** args.power
    if multiplier <= 0:
        parser.error("requested schedule gives a zero LR at the checkpoint iteration")

    scheduler_base_lr = args.target_lr / multiplier
    optimizer_groups = checkpoint["optimizer"]["param_groups"]
    scheduler["base_lrs"] = [scheduler_base_lr] * len(optimizer_groups)
    scheduler["_last_lr"] = [args.target_lr] * len(optimizer_groups)
    for group in optimizer_groups:
        group["lr"] = args.target_lr
        group["initial_lr"] = scheduler_base_lr

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        parser.error("temporary output already exists: {}".format(temporary))
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        "Prepared epoch {} checkpoint: scheduler iteration {}, target LR {:.9g}, "
        "polynomial base LR {:.9g}, total horizon {} epochs".format(
            epoch,
            current_iteration,
            args.target_lr,
            scheduler_base_lr,
            args.total_epochs,
        )
    )
    print("Output: {}".format(args.output))


if __name__ == "__main__":
    main()
