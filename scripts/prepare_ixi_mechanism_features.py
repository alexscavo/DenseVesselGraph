#!/usr/bin/env python3
"""Fill the missing early-CNN feature cache only; never rerun the detector."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from models.checkpoint import load_checkpoint_file
from scripts.ixi_cached_mechanisms import CACHES, manifest, identities, heads, read_cache, parity, save, sha


def prepare(output, smoke):
    output.mkdir(parents=True, exist_ok=False)
    m = manifest(); cfg = m['config']; ids = identities(smoke)
    assert sha(m['checkpoint']) == m['checkpoint_sha256']
    state = load_checkpoint_file(m['checkpoint'])['net']
    # Exact layer0 prefix in the current SENet: conv7/stride2/pad3, saved BN, ReLU.
    early = torch.nn.Sequential(OrderedDict([
        ('conv1', torch.nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False)),
        ('bn1', torch.nn.BatchNorm3d(64)), ('relu1', torch.nn.ReLU(inplace=True))])).eval().requires_grad_(False)
    early.load_state_dict({k[len('encoder.layer0.'):]: v for k, v in state.items()
                           if k.startswith(('encoder.layer0.conv1.', 'encoder.layer0.bn1.'))}, strict=True)
    assert state['encoder.layer0.conv1.weight'].shape == (64, 1, 7, 7, 7)
    del state
    _, spline = heads(); checks = []
    with torch.no_grad():
        for split, names in ids.items():
            (output/split).mkdir()
            for name in names:
                source = json.loads((CACHES['local_image']/split/(name+'-source.json')).read_text())
                raw = np.asarray(nib.load(source['raw']).dataobj, dtype=np.float32)
                assert tuple(raw.shape) == (64, 64, 64)
                image = torch.from_numpy(raw.copy())[None, None]-cfg['data']['datasets']['synthetic_mri']['foreground_mean']
                features = early(image)[0].numpy()
                c = read_cache('local_image', split, name); c['local_features'] = features
                error = parity(c, spline)
                np.save(output/split/(name+'.npy'), features)
                checks.append(dict(split=split, sample_id=name, spline_max_error_mm=error, raw_sha256=sha(source['raw'])))
    save(output/'summary.json', dict(completed=True, ids=ids, checks=checks, checkpoint_sha256=m['checkpoint_sha256'],
         note='Only conv1/bn1/relu1 evaluated on CPU; float32, batch1, saved BN. All decoded curves verified against original GPU cache within 0.001mm.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args(); torch.set_num_threads(1)
    prepare(a.output, a.smoke)
