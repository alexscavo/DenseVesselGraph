#!/usr/bin/env python3
"""Extract full-volume vessel graphs with legacy or original-Vedo centerlines."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ixi_vessel_graph import build_representation_family, write_source_graph


SCHEMA_VERSION = 3
FILES = ("nodes.csv", "edges.csv", "graph.vvg")
DATASET_DIRECTORIES = {
    "ixi": "IXI",
    "topbrain": "TopBrain_Data_Release_Batches1n2_081425",
    # Skull-stripped TopBrain, labels clipped to the brain mask (prepare_topbrain_stripped_20261104.py);
    # one dataset key per modality so that each gets its own patch set and split file.
    "topbrain_stripped_mr": "TopBrain_stripped_20261104",
    "topbrain_stripped_ct": "TopBrain_stripped_20261104",
    # Full, unstripped TopBrain volumes with the complete labels (prepare_topbrain_full_20261104.py); supersedes stripped.
    "topbrain_full_mr": "TopBrain_full_20261104",
    "topbrain_full_ct": "TopBrain_full_20261104",
    # Production: tight TopCoW brain mask (HD-BET / TotalSegmentator), labels clipped to it (5 Nov 2026).
    "topbrain_tight_mr": "TopBrain_tight_20261105",
    "topbrain_tight_ct": "TopBrain_tight_20261105",
}
GRAPH_POLICY_DIRECTORY = "optimal_radius_0p75x_c095"


def graph_policy_directory(centerline_backend: str) -> str:
    if centerline_backend == "legacy":
        return GRAPH_POLICY_DIRECTORY
    if centerline_backend == "vedo_original":
        return GRAPH_POLICY_DIRECTORY + "_vedo_original"
    raise ValueError(f"unsupported production backend: {centerline_backend}")


def configuration(args) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "centerline_backend": args.centerline_backend,
        "rdp_voxels": args.rdp_voxels, "radius_fraction": args.radius_fraction,
        "simplification_method": args.simplification_method,
        "spur_length": args.spur_length, "max_junction_extent_mm": args.max_junction_extent_mm,
        "smooth_iterations": args.smooth_iterations, "smooth_alpha": args.smooth_alpha,
        "minimum_chord_fraction": args.minimum_chord_fraction,
    }


def graph_destination(row: dict[str, str], args) -> Path:
    if args.dataset_root is not None:
        if row["dataset"] not in DATASET_DIRECTORIES:
            raise ValueError(f"unsupported dataset folder for {row['dataset']!r}")
        return (
            args.dataset_root / DATASET_DIRECTORIES[row["dataset"]]
            / "vascular_graphs" / graph_policy_directory(args.centerline_backend)
            / row["modality"] / row["split"] / row["subject"]
        )
    return args.output / row["dataset"] / row["modality"] / row["split"] / row["subject"]


def process(row: dict[str, str], args) -> str:
    destination = graph_destination(row, args)
    marker = destination / "complete.json"
    config = configuration(args)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    if marker.is_file() and not args.force:
        old = json.loads(marker.read_text())
        complete = old.get("configuration_fingerprint") == fingerprint
        complete &= all((destination / "graphs" / name / filename).is_file()
                        for name in ("junction_only", "adaptive", "dense") for filename in FILES)
        if complete: return "skipped"

    started = time.monotonic()
    image, label = nib.load(row["image"]), nib.load(row["label"])
    if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-4):
        raise ValueError(f"{row['subject']}: image/label geometry changed since planning")
    segmentation = np.asanyarray(label.dataobj) > 0
    spacing = np.sqrt((label.affine[:3, :3] ** 2).sum(axis=0))
    graphs, centerline_provenance = build_representation_family(
        segmentation, spacing=spacing,
        affine=label.affine,
        return_provenance=True,
        adaptive_tolerance_mm=args.rdp_voxels * float(spacing.min()),
        adaptive_radius_fraction=args.radius_fraction,
        centerline_backend=args.centerline_backend,
        spur_length=args.spur_length,
        max_junction_extent_mm=args.max_junction_extent_mm,
        smooth_iterations=args.smooth_iterations, smooth_alpha=args.smooth_alpha,
        minimum_chord_fraction=args.minimum_chord_fraction,
        simplification_method=args.simplification_method,
    )
    for name, graph in graphs.items():
        write_source_graph(destination / "graphs" / name, graph, label.affine, label.shape)
    payload = {
        **{key: row[key] for key in ("index", "dataset", "modality", "split", "subject", "image", "label")},
        "configuration": config, "configuration_fingerprint": fingerprint,
        "centerline_provenance": centerline_provenance,
        "shape": list(label.shape), "spacing_mm": spacing.tolist(),
        "foreground_voxels": int(segmentation.sum()),
        "representations": {name: {"nodes": graph.node_count, "edges": graph.edge_count,
            "betti_0": graph.betti()[0], "betti_1": graph.betti()[1]} for name, graph in graphs.items()},
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    return "done"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="legacy single-root layout")
    destination.add_argument("--dataset-root", type=Path, help="write directly inside IXI and TopBrain dataset folders")
    parser.add_argument("--shard", type=int, default=int(os.getenv("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--num-shards", type=int, default=int(os.getenv("REAL_GRAPH_NUM_SHARDS", "1")))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--centerline-backend", choices=("legacy", "vedo_original"), default="vedo_original")
    parser.add_argument("--rdp-voxels", type=float, default=0.0)
    parser.add_argument("--radius-fraction", type=float, default=0.75)
    parser.add_argument("--simplification-method", choices=("rdp", "optimal"), default="optimal")
    parser.add_argument("--spur-length", type=int, default=4)
    parser.add_argument("--max-junction-extent-mm", type=float, default=1.5)
    parser.add_argument("--smooth-iterations", type=int, default=5)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--minimum-chord-fraction", type=float, default=0.95)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error("shard must satisfy 0 <= shard < num-shards")
    with args.manifest.open(newline="") as stream: rows = list(csv.DictReader(stream))
    assigned = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(assigned)} of {len(rows)} volumes", flush=True)
    failures = []
    for position, row in enumerate(assigned, 1):
        try:
            state = process(row, args)
            print(f"[{position}/{len(assigned)}] {row['subject']} {state}", flush=True)
        except Exception as error:
            failures.append(row["subject"]); print(f"{row['subject']} FAILED: {error}", flush=True)
    if failures: raise SystemExit(f"failed ({len(failures)}): {', '.join(failures)}")


if __name__ == "__main__":
    main()
