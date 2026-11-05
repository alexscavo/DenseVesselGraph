#!/usr/bin/env python3
"""Create and curate vessel patches from completed full-volume graphs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_vessel_graphs import DATASET_DIRECTORIES, graph_policy_directory


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/datasets")


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, "-u", str(REPO / "scripts" / script), *arguments]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_DIRECTORIES), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "docs/research/vedo_original_extraction_manifest.csv")
    parser.add_argument("--centerline-backend", choices=("legacy", "vedo_original"),
                        default="vedo_original")
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--evidence", type=Path,
                        default=REPO / "docs/research/evidence/vedo_production")
    parser.add_argument("--reuse-patches-from", type=Path,
                        help="optional previous patches with identical raw/seg grid")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--maximum-stride", type=int, nargs=3, default=(40, 40, 40))
    parser.add_argument("--intensity-window", type=float, nargs=2, metavar=("LO", "HI"),
                        help="opt-in fixed window normalization (passed to the patch generator)")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    policy = graph_policy_directory(args.centerline_backend)
    base = args.dataset_root / DATASET_DIRECTORIES[args.dataset]
    sources = args.sources or base / "vascular_graph_sources" / policy
    output = args.output or base / "vascular_patches" / policy
    split = args.split_output or output / "patient_split.csv"
    if (output / "training_view.json").is_file():
        print(f"already curated: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.reuse_patches_from:
        old_split = args.reuse_patches_from / "patient_split.csv"
        if not old_split.is_file():
            raise FileNotFoundError(old_split)
        split.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_split, split)

    print(f"[1/4] stage {args.dataset} {policy} graphs", flush=True)
    run("prepare_real_graph_patch_sources.py",
        "--manifest", str(args.manifest), "--dataset-root", str(args.dataset_root),
        "--dataset", args.dataset, "--centerline-backend", args.centerline_backend,
        "--output", str(sources), "--split-output", str(split))
    print("[2/4] generate 64³ patch triplets", flush=True)
    generation = ["--root", str(sources), "--output-dir", str(output),
                  "--split-output", str(split), "--patch-size", *map(str, args.patch_size),
                  "--pad", "0", "0", "0", "--maximum-stride", *map(str, args.maximum_stride),
                  "--workers", str(args.workers), "--resume", "--allow-existing-split-proportions"]
    if args.reuse_patches_from:
        generation.extend(("--reuse-patches-from", str(args.reuse_patches_from)))
    if args.intensity_window:
        generation.extend(("--intensity-window", *map(str, args.intensity_window)))
    run("generate_synthetic_mri_dataset.py", *generation)
    print("[3/4] audit patch eligibility", flush=True)
    run("audit_real_patch_eligibility.py", "--patch-root", str(output),
        "--output", str(args.evidence / args.dataset))
    print("[4/4] archive ineligible train/val/test patches", flush=True)
    run("curate_vessel_training_patches.py", "--patch-root", str(output),
        "--all-splits", "--apply")


if __name__ == "__main__":
    main()
