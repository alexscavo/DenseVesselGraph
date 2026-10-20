"""Opt-in fixed update budgets and consumed-sample accounting for transfer runs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from data.loaders import dataset_sample_manifest


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if 'domain_discriminator' not in name:
            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class RecordedSampler(torch.utils.data.Sampler):
    """Keep the actual draw order; prefetch is not counted as a presentation."""
    def __init__(self, sampler):
        self.sampler = sampler
        self.num_samples = len(sampler)
        self.indices = []

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self.indices = list(self.sampler)
        return iter(self.indices)


def budget_loader(loader, draws, seed):
    assert draws % loader.batch_size == 0
    generator = torch.Generator().manual_seed(seed)
    if isinstance(loader.sampler, WeightedRandomSampler):
        sampler = WeightedRandomSampler(loader.sampler.weights, draws, replacement=True,
                                        generator=generator)
    else:
        sampler = RandomSampler(loader.dataset, replacement=True, num_samples=draws,
                                generator=generator)
    return DataLoader(loader.dataset, batch_size=loader.batch_size,
        sampler=RecordedSampler(sampler), num_workers=loader.num_workers,
        pin_memory=loader.pin_memory, collate_fn=loader.collate_fn,
        worker_init_fn=loader.worker_init_fn,
        generator=torch.Generator().manual_seed(seed + 1))


class Exposures:
    def __init__(self, loader, state=None):
        self.loader = loader
        self.lookup = []
        for dataset in dataset_sample_manifest(loader.dataset):
            source = 'leaves' if dataset['domain_label'] == 0 else 'synthetic_mri'
            self.lookup.extend((source, name) for name in dataset['sample_ids'])
        assert len(self.lookup) == len(loader.dataset)
        self.counts = Counter((state or {}).get('presentations', {}))
        self.seen = {key: set(value) for key, value in (state or {}).get('encountered_ids', {}).items()}
        self.updates = (state or {}).get('optimizer_updates', 0)

    def consume(self, batch_index, batch):
        size = len(batch[2]); assert size == self.loader.batch_size
        indices = self.loader.sampler.indices[batch_index*size:(batch_index+1)*size]
        assert len(indices) == size
        expected_domains = []
        for index in indices:
            domain, name = self.lookup[index]
            self.counts[domain] += 1
            self.seen.setdefault(domain, set()).add(name)
            expected_domains.append(0 if domain == 'leaves' else 1)
        assert expected_domains == batch[-1].tolist()
        self.updates += 1

    def summary(self):
        return dict(optimizer_updates=self.updates, samples_processed=sum(self.counts.values()),
            presentations={key: self.counts[key] for key in ('synthetic_mri', 'leaves')},
            unique_samples_encountered={key: len(self.seen.get(key, set())) for key in ('synthetic_mri', 'leaves')})

    def state(self):
        return dict(**self.summary(), encountered_ids={key: sorted(ids) for key, ids in self.seen.items()})


def initialize(model, plan, pretrain, smoke):
    """Both scratch arms read the same immutable initializer."""
    path = Path(plan['output_root']) / 'initialization.pt'
    if not path.exists():
        assert smoke, 'The common initializer must be created by verification first'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(net={k: v.detach().cpu() for k, v in model.state_dict().items()
                             if 'domain_discriminator' not in k}), path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if pretrain:
        report = model.load_state_dict(checkpoint['net'], strict=False)
        assert not report.unexpected_keys
        assert all('domain_discriminator' in k for k in report.missing_keys)
    digest = state_digest(model)
    if pretrain:
        expected = hashlib.sha256()
        for name, value in sorted(checkpoint['net'].items()):
            expected.update(name.encode()); expected.update(value.numpy().tobytes())
        assert digest == expected.hexdigest()
    return str(path), digest


def duplicate_diagnostics(prediction_path, root, split):
    """One-to-one 2-mm node matching; extra nodes near recovered GT are duplicates."""
    import numpy as np
    from scripts.diagnose_branch_relations import match_nodes
    from scripts.evaluate_vessel_core import records
    totals = Counter()
    samples = 0
    for row in records(prediction_path):
        samples += 1
        with np.load(Path(root)/split/'spline'/(row['source_sample_id']+'.npz')) as cache:
            metric = cache['metric']
            gt = cache['nodes'] @ metric.T
        pred = np.asarray(row['nodes_dhw'], dtype=float).reshape(-1, 3) @ metric.T
        # Reuse the existing maximum-cardinality, distance-constrained matcher.
        from scipy.spatial.distance import cdist
        distance = cdist(pred, gt)
        if len(pred) and len(gt):
            mapping = match_nodes(pred, gt, 2.)
            matched_pred = set(mapping); recovered_gt = list(mapping.values())
            duplicates = sum(i not in matched_pred and len(recovered_gt) > 0 and
                             distance[i, recovered_gt].min() <= 2. for i in range(len(pred)))
            totals['matched_nodes'] += len(mapping)
            totals['duplicate_nodes'] += int(duplicates)
        totals['predicted_nodes'] += len(pred); totals['target_nodes'] += len(gt)
    result = dict(totals, samples=samples, tolerance_mm=2.,
        duplicate_definition='unmatched predicted node within 2mm of an already matched GT node')
    for key in ('matched_nodes','duplicate_nodes','predicted_nodes','target_nodes'):
        result.setdefault(key,0)
    result['duplicates_per_patch'] = totals['duplicate_nodes']/max(1, samples)
    result['duplicate_fraction'] = totals['duplicate_nodes']/max(1, totals['predicted_nodes'])
    Path(prediction_path).with_name('duplicate-nodes.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
