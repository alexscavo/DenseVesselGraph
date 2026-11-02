"""R07 fixed-budget redistribution of reliable, nearby matched negatives.

No change to inference, the matcher, class balance or the set of positive edges.
All masks below are training supervision, never inference-time inputs.
"""
import numpy as np


def population(c, gt, source, target):
    source, target = np.asarray(source, int), np.asarray(target, int)
    inverse = {int(g): i for i, g in enumerate(target)}
    positive = np.asarray([[inverse[int(a)], inverse[int(b)]] for a, b in gt['edges']
                           if int(a) in inverse and int(b) in inverse], int).reshape(-1, 2)
    positive_set = {tuple(sorted(e)) for e in positive.tolist()}
    all_pairs = np.stack(np.triu_indices(len(source), 1), 1)
    negative = all_pairs[[tuple(e) not in positive_set for e in all_pairs]]
    local = np.concatenate([positive, negative])
    labels = np.r_[np.ones(len(positive), int), np.zeros(len(negative), int)]
    queries = source[local]
    vertices = target[local]
    distances = np.linalg.norm(c['physical'][source, None] - c['truth'][None], axis=-1)
    own = distances[np.arange(len(source)), target]
    nearest = distances.argmin(1)
    other = distances.copy()
    other[np.arange(len(source)), target] = np.inf
    margin = other.min(1) - own
    reliable = (own <= 2.) & (nearest == target) & (margin >= .5)
    separation = np.linalg.norm(c['truth'][vertices[:, 0]] - c['truth'][vertices[:, 1]], axis=-1)
    hard = (labels == 0) & (separation < 5.) & reliable[local].all(1)
    nearest_pairs = nearest[local]
    links = {tuple(sorted(e)) for e in gt['edges'].tolist()}
    alternative_positive = np.array([tuple(sorted(e)) in links for e in nearest_pairs])
    possible_conflict = ((labels == 0) & (distances.min(1)[local] <= 2.).all(1) &
                         (alternative_positive | (nearest_pairs[:, 0] == nearest_pairs[:, 1])))
    return dict(queries=queries, labels=labels, hard=hard, separation=separation,
                reliable=reliable, own_mm=own, endpoint_own_mm=own[local], margin_mm=margin,
                ambiguous=(~reliable)[local].any(1), potential_conflict=possible_conflict,
                source=source, target=target, vertices=vertices)


def redistribution_indices(labels, hard, graph_ids, rng, fraction=.25, multiplier=4):
    """Replace easy negative *slots*, after original batch-wide ratio balancing.

    Every positive slot is unchanged. Each graph retains exactly its previous
    effective negative count. Hard-pair multiplicity increases by at most 4x
    its balanced S0 multiplicity and at most 25% of negative slots are replaced.
    This is resampling with replacement, not discovery of previously unseen pairs.
    """
    labels, hard, graph_ids = np.asarray(labels), np.asarray(hard), np.asarray(graph_ids)
    selected = np.arange(len(labels))
    for g in np.unique(graph_ids):
        negative = np.flatnonzero((graph_ids == g) & (labels == 0))
        difficult = negative[hard[negative]]
        donors = negative[~hard[negative]]
        if not len(difficult) or not len(donors):
            continue
        # Original ratio upsampling may already contain copies. Repeating each
        # original slot at most multiplier-1 times preserves the bound per pair.
        additions = np.tile(difficult, multiplier - 1)
        rng.shuffle(additions)
        count = min(int(np.floor(fraction * len(negative))), len(donors), len(additions))
        if count:
            selected[rng.choice(donors, count, replace=False)] = additions[:count]
    assert np.array_equal(labels[selected], labels)
    assert np.array_equal(selected[labels == 1], np.flatnonzero(labels == 1))
    return selected
