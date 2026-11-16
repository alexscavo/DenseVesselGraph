#!/usr/bin/env python3
"""Run the whole-volume stitching steps of one entry in one CPU job (12 Nov 2026; the cpu_p1 queue waits ~1 h).

  part A: per-subject check against the stored-patch stitching (stitch_rows_vs_stored_20261112.py), then validation
          stitching at the run's own thresholds for the grids overlap, thin, tile0;
  part B: per grid, threshold selection on the first N TRAINING subjects (--select-thresholds), then validation
          stitching at the selected thresholds (--thresholds-from, tag 'selected').
  part D (13 Nov 2026): rows of another node decoding (predict_volume_grid --flip-average / --nms, rows dir
          rows/<entry>-<run>-<suffix>, --rows-suffix): validation stitching, 'core, hungarian 2 mm' only, at the run's
          thresholds under that decoding (recorded in the rows meta), tag <suffix>; then the stitched graphs of the
          first --dump-subjects validation subjects are dumped (<entry>-<grid>-<suffix>-<patient>.pkl, for the figures).
Each step is a separate process; a failing step is reported and the others still run (a grid whose selection failed
gets no 'selected' validation run). Exit code 1 if any step failed.
usage: stitch_batch_20261112.py <entry> <run> A|B|D [--train-subjects N] [--grids overlap thin tile0] [--rows-suffix S]
       [--dump-subjects N]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = Path(os.environ['SCRATCH']) / 'experiments/gnbm/stitching_20261112/rows'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entry')
    ap.add_argument('run')
    ap.add_argument('part', choices=('A', 'B', 'D'))
    ap.add_argument('--train-subjects', type=int, default=6)
    ap.add_argument('--grids', nargs='+', default=['overlap', 'thin', 'tile0'])
    ap.add_argument('--rows-suffix', default='', help='part D: rows dir rows/<entry>-<run>-<suffix> (e.g. nms5-flipavg)')
    ap.add_argument('--dump-subjects', type=int, default=2, help='part D: dump the stitched graphs of the first N validation subjects')
    a = ap.parse_args()
    if a.part == 'D' and not a.rows_suffix:
        raise SystemExit('part D needs --rows-suffix')
    rows = ROWS / (f'{a.entry}-{a.run}' + (f'-{a.rows_suffix}' if a.rows_suffix else ''))
    stitch = [sys.executable, '-u', str(ROOT / 'scripts/stitch_volume_20261112.py'), a.entry, a.run, '--rows-dir', str(rows)]
    steps = []
    if a.part == 'A':
        steps.append(('compare', [sys.executable, '-u', str(ROOT / 'scripts/stitch_rows_vs_stored_20261112.py'), a.entry, a.run, '--rows-dir', str(rows)]))
        steps += [(f'val {g} own', stitch + ['--grid', g, '--split', 'val']) for g in a.grids]
    elif a.part == 'D':
        for g in a.grids:
            core = stitch + ['--grid', g, '--split', 'val', '--only', 'core, hungarian 2 mm', '--tag', a.rows_suffix]
            steps.append((f'val {g} {a.rows_suffix}', core))
            if a.dump_subjects:
                steps.append((f'dump {g} {a.rows_suffix}', core + ['--subjects', str(a.dump_subjects), '--dump']))
    else:
        for g in a.grids:
            steps.append((f'select {g}', stitch + ['--grid', g, '--split', 'train', '--subjects', str(a.train_subjects), '--select-thresholds']))
            sel = ROOT / 'experiments/stitching_20261112' / f'{a.entry}-{a.run}-{g}-thresholds.json'
            steps.append((f'val {g} selected', stitch + ['--grid', g, '--split', 'val', '--thresholds-from', str(sel), '--tag', 'selected']))
    failed = []
    for name, cmd in steps:
        if name.endswith('selected') and f'select {name.split()[1]}' in failed:
            print(f'BATCH {a.entry} {name}: SKIPPED (selection failed)', flush=True)
            failed.append(name)
            continue
        t0 = time.time()
        print(f'BATCH {a.entry} {name}: start', flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        print(f'BATCH {a.entry} {name}: exit {rc}, {time.time() - t0:.0f} s', flush=True)
        if rc:
            failed.append(name)
    print(f'BATCH {a.entry} part {a.part}: failed steps {failed}', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
