#!/usr/bin/env python3
"""How the pass-through head decides on two-hop chains A-C-B (traced_passthrough, validation; 3 Nov 2026).

CPU, saved validation rows + images, each run at its train-monitor-selected node, edge and
pass-through thresholds. Every chain row (A, B, via C) is labelled from GT (2 mm matching):
  pass-through   C spurious and A-B a GT edge  -> contracting C is correct
  spurious, no   C spurious but A-B not a GT edge (e.g. A or B also spurious, or a branch end)
  real node      C matched to a GT node        -> C must be kept
Outputs (docs/research/figures/passthrough_20261103/): q_distributions.png (pass-through score
per category with the selected threshold), examples.png (before / after the decision, image
MIP crops), and passthrough-decisions.json (decision counts and precision of contractions).
The label above is the strict one, taken before contraction. Because removing C changes the one-to-one node
matching, each contraction is also audited after re-matching the kept nodes (A-B correct in the final graph),
with the distance from C to its nearer endpoint: within the 2 mm matching radius, C and that endpoint are near
duplicates of one GT node, so a "real node" C can be removed with no loss.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from node_heatmap_probe_20261030 import make_dataset, SIZE  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402

D = T.D
RUN = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
OUT = REPO / 'docs/research/figures/passthrough_20261103'
ARM = 'traced_passthrough'
CATS = [('pass-through', 'C spurious, A–B is a GT edge', '#2ca02c'),
        ('spurious-no-edge', 'C spurious, A–B not a GT edge', '#ff7f0e'),
        ('real-node', 'C matched to a GT node', '#d62728')]


def chains(rows, nt, et, tau):
    out = []
    for r in rows:
        if r.get('pass_scores') is None or r['edge_scores'] is None:
            continue
        with np.load(r['cache']) as z:
            M, gn, ge = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2)
        m = D.match(torch.as_tensor(r['nodes'] @ M.T), torch.as_tensor(gn))  # ok: rows only hold nodes >= nt
        gt = {tuple(sorted(map(int, e))) for e in ge}
        final, plain = T.contract(r, nt, et, tau), T.contract(r, nt, et, None)
        done = {tuple(t) for t in final['contracted']}
        mm = r['nodes'] @ M.T
        kept = np.nonzero(final['node_scores'] >= nt)[0]
        m2 = {int(kept[i]): g for i, g in D.match(torch.as_tensor(mm[kept]), torch.as_tensor(gn)).items()}
        accepted = {tuple(sorted(map(int, p))) for p, s in zip(r['pairs'], plain['edge_scores']) if s >= et}
        is_gt = lambda u, v, mp: u in mp and v in mp and tuple(sorted((mp[u], mp[v]))) in gt  # noqa: E731
        for j in np.nonzero(r['via'] >= 0)[0]:
            a, b, c = int(r['pairs'][j, 0]), int(r['pairs'][j, 1]), int(r['via'][j])
            if c not in m:
                cat = 'pass-through' if a in m and b in m and tuple(sorted((m[a], m[b]))) in gt else 'spurious-no-edge'
            else:
                cat = 'real-node'
            sides = [tuple(sorted(x)) for x in ((a, c), (c, b))]
            out.append(dict(sample=r['sample_id'], j=int(j), a=a, b=b, c=c, q=float(r['pass_scores'][j]), cat=cat,
                            contracted=(a, c, b) in done, heat_c=float(r['node_scores'][c]),
                            ac=sides[0] in accepted, cb=sides[1] in accepted,
                            near_mm=float(min(np.linalg.norm(mm[c] - mm[a]), np.linalg.norm(mm[c] - mm[b]))),
                            correct_after=is_gt(a, b, m2),
                            tp_removed=sum(s in accepted and is_gt(*s, m) for s in sides)))
    return out


def summary(ch, tau):
    res = {}
    for k, _, _ in CATS:
        sel = [c for c in ch if c['cat'] == k]
        res[k] = dict(chains=len(sel), contracted=sum(c['contracted'] for c in sel),
                      q_median=float(np.median([c['q'] for c in sel])) if sel else None)
    n = sum(c['contracted'] for c in ch)
    res['contractions'] = n
    res['contraction_precision'] = res['pass-through']['contracted'] / max(1, n)
    res['pass_through_recall'] = res['pass-through']['contracted'] / max(1, res['pass-through']['chains'])
    res['threshold'] = tau
    done = [c for c in ch if c['contracted']]
    frac = lambda f: sum(map(f, done)) / max(1, n)  # noqa: E731
    res['audit_of_contractions'] = dict(
        correct_after_rematching=frac(lambda c: c['correct_after']),
        c_within_2mm_of_endpoint=frac(lambda c: c['near_mm'] <= 2.0),
        near_mm_median=float(np.median([c['near_mm'] for c in done])) if done else None,
        both_sides_accepted_before=frac(lambda c: c['ac'] and c['cb']),
        one_side_accepted_before=frac(lambda c: c['ac'] != c['cb']),
        no_side_accepted_before=frac(lambda c: not c['ac'] and not c['cb']),
        true_edges_removed=sum(c['tp_removed'] for c in done))
    res['above_threshold_not_contracted'] = {k: sum(c['cat'] == k and tau is not None and c['q'] >= tau and not c['contracted'] for c in ch) for k, _, _ in CATS}
    return res


def draw(ax, img, extent, proj, row, nodes, scores, pairs, curves, keep_nodes, accepted, title, gp, names):
    ax.imshow(img, cmap='gray', extent=extent, alpha=.55)
    if gp is not None:
        ax.plot(*proj(gp), '-', color='#2ca02c', lw=5, alpha=.35)
    for j in accepted:
        ax.plot(*proj(curves[j]), '-', color='#1f77b4', lw=2)
    for k, i in names.items():
        if scores[i] < 0:
            ax.plot(*proj(nodes[i:i + 1]), 'x', ms=12, mew=3, color='#d62728')
        else:
            ax.plot(*proj(nodes[i:i + 1]), 'o', ms=9, color={'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#d62728'}[k], mec='k')
        ax.annotate(k, proj(nodes[i:i + 1]), xytext=(5, 5), textcoords='offset points', fontsize=12, weight='bold')
    others = [i for i in keep_nodes if i not in names.values()]
    ax.plot(*proj(nodes[others]), 's', ms=4, color='w', mec='k')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def example(axes, d, rows_by_id, ch, nt, et, tau, heading):
    r = rows_by_id[ch['sample']]
    s = make_dataset(d, [ch['sample']], augment=False)[0]
    img3 = s[0][0][0].numpy()
    nodes, a, b, c = r['nodes'] * SIZE, ch['a'], ch['b'], ch['c']
    with np.load(r['cache']) as z:
        M, gn, ge = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2)
        m = D.match(torch.as_tensor(r['nodes'] @ M.T), torch.as_tensor(gn))
        gp = None
        if a in m and b in m:
            for e, (u, v) in enumerate(ge):
                if {int(u), int(v)} == {m[a], m[b]}:
                    gp = z['full_paths'][z['path_offsets'][e]:z['path_offsets'][e + 1]] * SIZE
    pts = np.stack([nodes[a], nodes[b], nodes[c]])
    axis = int(np.argmin(np.ptp(pts if gp is None else gp, 0)))
    ka = [k for k in range(3) if k != axis]
    lo = np.clip(np.floor(pts.min(0) - 10), 0, SIZE - 1).astype(int)
    hi = np.clip(np.ceil(pts.max(0) + 10), 0, SIZE - 1).astype(int)
    sl = tuple(slice(lo[k], hi[k] + 1) for k in range(3))
    img = img3[sl].max(axis)
    extent = (lo[ka[1]] - .5, hi[ka[1]] + .5, hi[ka[0]] + .5, lo[ka[0]] - .5)
    proj = lambda v: (v[:, ka[1]], v[:, ka[0]])
    names = {'A': a, 'B': b, 'C': c}
    keep_nodes = list(range(len(nodes)))
    before, after = T.contract(r, nt, et, None), T.contract(r, nt, et, tau)
    curves = r['curves'] * SIZE
    near = lambda q: [j for j, (u, v) in enumerate(q['pairs']) if q['edge_scores'][j] >= et and {int(u), int(v)} & {a, b, c}
                      and q['node_scores'][u] >= nt and q['node_scores'][v] >= nt]
    fg = img[img > img.min()]
    axes[0].imshow(img, cmap='gray', extent=extent, vmin=np.percentile(fg, 30) if len(fg) else None, vmax=np.percentile(fg, 99.7) if len(fg) else None)
    if gp is not None:
        axes[0].plot(*proj(gp), '-', color='#2ca02c', lw=2)
    for k, i in names.items():
        axes[0].plot(*proj(nodes[i:i + 1]), 'o', ms=9, color={'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#d62728'}[k], mec='k')
        axes[0].annotate(k, proj(nodes[i:i + 1]), xytext=(5, 5), textcoords='offset points', fontsize=12, weight='bold')
    axes[0].set_xlim(extent[0], extent[1]); axes[0].set_ylim(extent[2], extent[3]); axes[0].set_xticks([]); axes[0].set_yticks([])
    label = dict((k, t) for k, t, _ in CATS)[ch['cat']]
    state = lambda ok: 'accepted' if ok else 'not accepted'  # noqa: E731
    reason = 'q < τ' if ch['q'] < tau else 'guard: C has another accepted edge, or a chain node was already removed'
    axes[0].set_title(f'{heading}\n{d} {ch["sample"]}\n{label}; green = GT edge A–B', fontsize=9)
    draw(axes[1], img, extent, proj, r, nodes, before['node_scores'], r['pairs'], curves, keep_nodes, near(before),
         f'Before: A–C {state(ch["ac"])}, C–B {state(ch["cb"])}; heat(C) = {ch["heat_c"]:.2f}; C to nearer end {ch["near_mm"]:.1f} mm', gp, names)
    draw(axes[2], img, extent, proj, r, nodes, after['node_scores'], r['pairs'], curves, keep_nodes, near(after),
         f'After: pass-through score q(C) = {ch["q"]:.2f} vs τ = {tau}\n' + (f'C removed, A–B accepted; A–B correct after re-matching: {"yes" if ch["correct_after"] else "no"}' if ch['contracted'] else f'C kept ({reason})'), gp, names)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report, cache = {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, d in enumerate(['SyntheticMRI', 'IXI', 'BRAVA']):
        res = json.loads((RUN / d / ARM / 'final.json').read_text())
        nt, et, tau = res['thresholds']['node'], res['thresholds']['edge'], res['passthrough']['threshold']
        rows = pickle.loads((RUN / d / ARM / 'final-validation-rows.pkl').read_bytes())
        ch = chains(rows, nt, et, tau)
        report[d] = summary(ch, tau)
        print('PASS', d, json.dumps(report[d]), flush=True)
        bins = np.linspace(0, 1, 26)
        for key, lab, col in CATS:
            q = [c['q'] for c in ch if c['cat'] == key]
            above = sum(v >= tau for v in q) if tau is not None else 0
            axes[k].hist(q, bins=bins, histtype='step', lw=2, color=col, label=f'{lab} (n={len(q)}, {above} ≥ τ)')
        axes[k].set_yscale('log')
        if tau is not None:
            axes[k].axvline(tau, color='k', ls='--', lw=1, label=f'selected τ = {tau}')
        audit = report[d]['audit_of_contractions']
        axes[k].set_title(f'{d}: {report[d]["contractions"]} contractions\nA–B correct: {report[d]["contraction_precision"]:.2f} strict, {audit["correct_after_rematching"]:.2f} after re-matching', fontsize=10)
        axes[k].set_xlabel('pass-through score q(C)')
        axes[k].legend(fontsize=7)
        cache[d] = (rows, ch, nt, et, tau)
    axes[0].set_ylabel('chains (log scale)')
    fig.suptitle('Pass-through head on validation two-hop chains A–C–B (traced_passthrough)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / 'q_distributions.png', dpi=110)
    # examples: one correct contraction per dataset (median q), a real node the head rejects, a strict-label "error" that is
    # a near duplicate, and a wrong contraction that is not (highest q in each case but the rejection, which is a median)
    picks = []
    for d in ['SyntheticMRI', 'IXI', 'BRAVA']:
        rows, ch, nt, et, tau = cache[d]
        good = sorted([c for c in ch if c['cat'] == 'pass-through' and c['contracted']], key=lambda c: -c['q'])
        if good:
            picks.append((d, good[len(good) // 2], 'Correct contraction (median q among correct ones)'))
    rows, ch, *_ = cache['IXI']
    tau = cache['IXI'][4]
    rejected = sorted([c for c in ch if c['cat'] == 'real-node' and c['q'] < tau and c['ac'] and c['cb']], key=lambda c: c['q'])
    if rejected:
        picks.append(('IXI', rejected[len(rejected) // 2], 'Real node rejected by the head (median q)'))
    rows, ch, *_ = cache['BRAVA']
    wrong = sorted([c for c in ch if c['contracted'] and c['cat'] == 'real-node'], key=lambda c: -c['q'])
    harmless = [c for c in wrong if c['correct_after']]
    harmful = [c for c in wrong if not c['correct_after'] and c['near_mm'] > 2.0]
    if harmless:
        picks.append(('BRAVA', harmless[0], 'Strict label wrong, final graph right (highest q)'))
    if harmful:
        picks.append(('BRAVA', harmful[0], 'Wrong: real node removed, C > 2 mm from both ends (highest q)'))
    fig, axes = plt.subplots(len(picks), 3, figsize=(16, 5.2 * len(picks)), squeeze=False)
    for i, (d, ch, head) in enumerate(picks):
        rows, _, nt, et, tau = cache[d]
        example(axes[i], d, {r['sample_id']: r for r in rows}, ch, nt, et, tau, head)
    fig.suptitle('Pass-through decision: before / after (blue = accepted edges, red x = removed node C)', fontsize=12, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(OUT / 'examples.png', dpi=100)
    (REPO / 'experiments/dense_graph_20261030/passthrough-decisions.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    torch.set_num_threads(8)
    main()
