#!/usr/bin/env python3
"""Print saved native-2D results using the standard library only."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'experiments/native2d_20261027'
pointer=out/'worktree.json'
if pointer.exists():out=Path(json.loads(pointer.read_text())['output'])
if (out/'summary.md').exists():print((out/'summary.md').read_text())
else:
    print('Native 2D: CPU checks complete; GPU/tiny-fit results are pending. No full training submitted.')
    if (out/'submission.json').exists():print((out/'submission.json').read_text())
    print('Evidence:',out)
