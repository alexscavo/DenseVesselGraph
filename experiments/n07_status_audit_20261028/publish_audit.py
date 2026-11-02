"""Publish the N07 status audit without changing experiment implementations."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
PROJECTS = WORKTREE.parent
MAIN = PROJECTS / 'Graph-Native-Betti-Matching-Representation'
REPORT = MAIN / 'docs/research/metrics/3-ixi-node-refinement-pilot.md'
REGISTRY = MAIN / 'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md'
MARKER = '<!-- N07 STATUS AUDIT 20261028 -->'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    capacity = json.loads((HERE / 'target-capacity.json').read_text())
    identities = json.loads((HERE / 'checkpoint-identities.json').read_text())
    assert all(x['matches_recorded_plan'] for x in identities.values())
    root = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_local_components_20261026')
    smoke = json.loads((root / 'smoke.json').read_text())
    assert not smoke['completed'] and not smoke['arms']
    assert not any(root.rglob('*.pt')) and not (root / 'analysis.json').exists()
    logs = WORKTREE / 'experiments/ixi_local_components_20261026/logs'
    assert 'AF_UNIX path too long' in (logs / '1583591_4294967294.err').read_text()
    for name in ['1583591_4294967294.out', '1583591_4294967294.err',
                 '1569366_4294967294.out', '1569366_4294967294.err']:
        shutil.copy2(logs / name, HERE / name)
    shutil.copy2(root / 'smoke.json', HERE / 'original-incomplete-smoke.json')
    status = dict(audited_at=datetime.now().astimezone().isoformat(),
                  status='PARTIAL / RECOVERY', scientific_training='NOT RUN',
                  planned_arms=['B0', 'A-xyz-fresh', 'A-xyz-transfer', 'A-cls', 'A', 'A-control'],
                  initial_jobs={'1569366': 'FAILED', '1569367': 'CANCELLED',
                                '1569368': 'CANCELLED', '1569369': 'CANCELLED'},
                  replacement_jobs={'1583591': 'TIMEOUT', '1583592': 'CANCELLED',
                                    '1583593': 'CANCELLED', '1583594': 'CANCELLED'},
                  replacement_smoke_elapsed='00:30:04',
                  replacement_smoke_start='2026-10-26T15:14:48+01:00',
                  replacement_smoke_end='2026-10-26T15:44:52+01:00',
                  last_output='smoke-B0/tiny/caches.zip (0 bytes; not a valid prediction cache)',
                  recorded_error='OSError: AF_UNIX path too long',
                  source_checkpoint_identities=identities,
                  no_active_user_jobs_at_audit=True,
                  historical_bn={'1584593_0': 'FAILED (empty-pair index dtype error)',
                                 '1584593_1': 'COMPLETED (SyntheticMRI raw artifacts exist)',
                                 '1584594': 'CANCELLED (no validated combined collector)'},
                  submitted_jobs=[], recovery_gpu_smoke_passed=False,
                  next_run='Repair short SCRATCH multiprocessing TMPDIR; rerun isolated correctness gate before any matched 256-query N07 pilot. No GPU submission in this CPU/status audit.',
                  capacity_file=str(HERE / 'target-capacity.json'))
    (HERE / 'status.json').write_text(json.dumps(status, indent=2) + '\n')

    table = '| Population | Patches / subjects | p95 / p99 nodes | Max nodes | >120 / >192 / >256 |\n|---|---:|---:|---:|---:|\n'
    for dataset, info in capacity['datasets'].items():
        for split, c in info['splits'].items():
            table += (f"| {dataset} {split} | {c['patches']:,} / {c['subjects']} | "
                      f"{c['p95']:.2f} / {c['p99']:.2f} | {c['max']} | "
                      f"{c['overflow']['120']} / {c['overflow']['192']} / {c['overflow']['256']} |\n")
    section = f'''\n\n{MARKER}
## N07/D04 execution and current-target capacity audit — 28 October 2026

**N07 is prepared but scientifically untested (PARTIAL / RECOVERY).** The modern standalone image-conditioned A-cls comparison did not complete zero-shot evaluation or any planned IXI fine-tuning arm. This is not a negative result for classification refinement. The historical October18 continuation and SyntheticMRI combined-A component ablations do not supply that missing comparison.

### Actual execution, rather than accepted submission

| Chain | Correctness smoke | Zero-shot | Six-arm training | Collector |
|---|---|---|---|---|
| Initial | 1569366 FAILED after 1m46s (cross-batch coordinate drift gate) | 1569367 CANCELLED | 1569368 CANCELLED | 1569369 CANCELLED |
| Replacement | 1583591 TIMEOUT after 30m04s | 1583592 CANCELLED | 1583593 CANCELLED | 1583594 CANCELLED |

All downstream jobs have zero execution time. The replacement smoke ran October26 15:14:48–15:44:52 Paris time. Live `sacct` confirms the states; `squeue` contained no same-user jobs at audit. No planned-arm checkpoint, zero-shot summary or collected result exists in the experiment root. The 0-byte `runs/smoke-B0/tiny/caches.zip` is an interrupted smoke artifact, not reusable inference.

The replacement log contains multiprocessing **`OSError: AF_UNIX path too long`** before the timeout. The job wrapper uses a long experiment-runtime path as TMPDIR; a corresponding multiprocessing listener address is 133 bytes and reproduces that exception in a CPU socket check. The saved partial smoke records `completed=false`, no completed arms, batch1 repeat drift **0**, batch1-versus4 normalized coordinate drift **0.00109038**, and physical-spacing checks for149 IXI subjects (approximately0.46–0.4883,0.46–0.4883,0.80mm). It reached the B0 tiny evaluation, but did not validate the six-arm GPU gate or determine a common training budget. Increasing wall time alone is not a fix.

**Required recovery:** separate a short SCRATCH-backed multiprocessing TMPDIR from the long runtime/cache directory; preserve the same batch1 evaluation policy, original plan and old failure artifacts; pass the full correctness/benchmark gate before launching scientific work. No runtime/model code was changed and no GPU job was submitted in this audit, following registry §11's CPU-first/no-training instruction. The stale coordination claim was corrected to zero reserved/running H100s.

### Current serialized K8 targets (J02)

Read every retained development NPZ's actual `nodes`, `edges` and `boundary`, checking K8 control shapes and valid graph indices. Membership matches both dataset manifests, all counts match `patch_index.csv` anchor/branch counts, and eight representative VTP headers agree. Train/validation subjects are disjoint. No image processing, target regeneration, filtering, inference or test-array access was performed.

{table}
There are **zero degree-2 object anchors** in either audited current target population. Isolated anchors are preserved (IXI train/val76/36; BRAVA2610/584), as are recorded boundary anchors (IXI95940/25691; BRAVA102426/21613). Dense paths are separate spline guides, not extra detector objects. The current generator calls `contract_spline_graph(...max_anchors=256,preserve_nodes=boundary)`; existing IXI development exclusions comprise parallel branches (train1207/val253) and closed branches (217/59), not object-capacity exclusions. BRAVA records no exclusions. Counts describe the existing retained population, not all possible source crops.

**120 is not capacity-safe for unchanged IXI labels.** The fixed N07 population also contains one153-node training patch (`sample_000150_0066`) and one124-node validation patch (`sample_000058_0085`). Its4096/512 patches cover123/26 subjects. A 120-query native zero-shot evaluation may be reported as a source-capacity diagnostic, but it must retain overflow patches/GT in the denominator; it is not a capacity-matched fine-tuning arm. Keep256 for N07 recovery.192 would fit current counts but is an additional untested configuration. BRAVA's current retained development targets fit120; that says nothing by itself about model accuracy.

Historical239-node provenance is independently located in `docs/research/evidence/ixi_capacity_192_256/{{selection,summary}}.json`: older root `IXI/vascular_patches/optimal_radius_0p75x_c095`, sample`sample_000021_0117`,239nodes/261edges. That older root is no longer available at its recorded path, so its degree histogram could not be independently recounted. It must not be substituted for the **current contracted** maximum153. Current zero-degree2 counts and actual overflows are measured directly here.

### Identities, related BN status and decision

The immutable N07 plan/manifests and all three source checkpoint SHA256 values were verified. Source seed364505, step12000: common B0 `91ac46e091bb3f74c5748758235c26dc7c8814a232adda03dd49def0c1c8a3de`; A-xyz `a1210fe958dffe0e588d38222cb62bf48b58e9b238fc57c027ec1c7a795277c9`; combined A `ebb47360f7799801f1ab364f3ef9e60a1b877038d39962b4861d33d1944b9385`. Exact paths and byte sizes are in `checkpoint-identities.json`. The fine-tuning plan copies120 learned object rows, seeds136 additional object rows identically across arms, and moves both learned relation-token rows to256/257. Only A-xyz-transfer receives the learned local-refinement head; the shared trunk is B0 for every arm. A-cls is fresh zero-residual classification refinement using the established27-sample module, not the KNN verifier.

L06 was also checked: IXI1584593_0 FAILED after49s with an empty-pair indexing-dtype error; SyntheticMRI1584593_1 COMPLETED after1m37s and has saved `complete.json`, rows and recalibrated caches; collector1584594 CANCELLED. It remains **partial**, and none of those artifacts establishes modern IXI A-cls efficacy. No BN recovery was launched.

**Decision:** no modern IXI winner or training-duration recommendation can be made. The next permitted bounded recovery is the existing256-query B0/A-xyz/A-cls component study after its socket-path smoke fix, using the same initialization, splits and losses. Its final update budget is still unset (planned cap6000; minimum1500), because the gate did not finish. Do not launch a full-IXI120/256 comparison with silent overflow exclusions. No new architecture or loss experiment is justified by an infrastructure timeout.

Evidence: `{HERE}` (`status.json`, `target-capacity.json`, `per-patch.csv`, `per-subject.json`, `overflow-above120.csv`, `cross-checks.json`, copied smoke/logs, source identities). Target audit took{capacity['seconds']:.1f}s on CPU. Reproduce counts with:

```bash
/lustre/fshomisc/sup/pub/anaconda-py3/h100/2024.06/envs/pytorch-gpu-2.3.1+py3.11.9/bin/python {HERE}/audit_targets.py
```

Read the confirmed state without inference or training:

```bash
python3 {WORKTREE}/scripts/show_ixi_local_components.py
```
'''
    changes = {}
    text = REPORT.read_text()
    assert MARKER not in text, 'Already published; preserve the audit'
    text = text.replace('**Status: submitted; scientific results pending.**',
                        '**Status checked 28 October2026: PARTIAL / RECOVERY; zero-shot and planned fine-tuning NOT RUN.**', 1)
    text = text.replace('### Results pending\n\nCPU routing, physical sampling-grid and query-expansion tests passed. GPU correctness, throughput and scientific comparisons are dependency-gated; no results are claimed before they finish.',
                        '### Scientific results unavailable — execution audited 28 October2026\n\nThe replacement GPU smoke timed out after multiprocessing socket-path failures; all downstream jobs were cancelled before starting. CPU checks passed, but the full GPU gate never completed. See the dated execution/capacity audit below. No A-cls/B0/A-xyz efficacy result is available from this batch.', 1)
    changes[REPORT] = text + section

    text = REGISTRY.read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('| **N07** |'):
            lines[i] = '| **N07** | Pure image-aware node classification `A-cls` on modern IXI or SyntheticMRI | **PARTIAL / RECOVERY — scientific comparison NOT RUN** | Live audit28Oct: smoke1569366 FAILED; replacement1583591 TIMEOUT (multiprocessing AF_UNIX path too long); zero-shot1583592, training1583593 and collector1583594 CANCELLED before execution. No modern A-cls efficacy checkpoint/result. Preserve separation from combined A and KNN verifier. | [Report3 execution audit](../docs/research/metrics/3-ixi-node-refinement-pilot.md#n07d04-execution-and-current-target-capacity-audit--28-october-2026); I01 recovery gated |'
        elif line.startswith('| **D04** |'):
            lines[i] = '| **D04** | Modern SyntheticMRI → IXI fully unfrozen transfer, B0/pretrained-fresh A-xyz, standalone A-cls ± combined A | **PARTIAL / RECOVERY — scientific comparison NOT RUN** | Accepted chains are located and audited; both stopped at correctness smoke. No zero-shot or planned fine-tuning arm ran. Source checkpoint hashes and4096/512 manifests verified; current contracted IXI max153 requires more than120 queries. | N07 audit28Oct; repair socket TMPDIR and pass gate before bounded256-query recovery |'
        elif line.startswith('| **D04/N07 —'):
            lines[i] = '| **D04/N07 — modern IXI B0/A-xyz/A-cls transfer** | **AUDITED28October: scientific comparison NOT RUN.** Smoke timeout and cancelled dependents verified; no planned-arm checkpoint/result. | Repair short SCRATCH multiprocessing TMPDIR, then revalidate the correctness gate. No duplicate or additional training submitted by the audit. |'
        elif line.startswith('| J00 / P0 |'):
            lines[i] = '| J00 / P0 | What is already running or measured on modern IXI? | Live Slurm, checkpoints, logs and collectors checked28Oct; N07/D04 failed before scientific work, L06 remains partial. | No modern fully-unfrozen IXI result; current queue empty. Preserve old failures and source hashes. | **DONE — measured status audit**, N07/D04 require recovery |'
        elif line.startswith('| J02 / P0 |'):
            lines[i] = '| J02 / P0 | Is120 sufficient for current contracted IXI and BRAVA GT? | All12649 IXI and10564 BRAVA train/val K8 caches counted; zero degree2 object anchors; subject splits/index/VTP checks pass. | IXI train/val max153/141, five/five patches >120 (none >192); N07 subset one/one overflow. BRAVA max64/69, no120 overflow. Historical239 is an older root, not current capacity. | **DONE — measured CPU audit28Oct**; retain256 for unchanged IXI pilot |'
        elif line.startswith('| Q120 / P1 conditional |'):
            lines[i] = '| Q120 / P1 conditional | Does120 outperform256 after contraction? | Current J02 audit disproves full-IXI120 capacity safety; do not drop the ten overflow targets. Existing256 study stays unchanged. | BRAVA fits120, but a separate accuracy comparison is not launched. A partial-population IXI study would require a separately declared design. | **DEFERRED for unchanged full IXI**; BRAVA conditional |'
        elif line.startswith('| I01 / P0 |'):
            lines[i] = '| I01 / P0 | Can IXI match SyntheticMRI with the modern method? | J00 confirms D04/N07 never passed smoke. Recover existing256-query component protocol after short SCRATCH socket-path fix and full correctness gate; preserve manifests/losses/source hashes. | No modern IXI efficacy conclusion or winner; no automatic large run from the CPU audit. | **NOT RUN; bounded recovery prerequisite confirmed** |'
        elif line.startswith('| **1** | `J00+J02` |'):
            fields = line.split('|')
            fields[-2] = ' **DONE28Oct — status/capacity audit; no training submitted** '
            lines[i] = '|'.join(fields)
        elif line.startswith('- **IXI modern D04/N07 still STATUS UNKNOWN'):
            lines[i] = '- **IXI modern D04/N07 audited28October: PARTIAL / RECOVERY, scientific comparison NOT RUN.** Replacement smoke1583591 timed out with multiprocessing AF_UNIX path-too-long errors; all dependent scientific jobs cancelled. Source hashes/manifests verified. Recover the existing256-query protocol only after its GPU correctness gate; do not substitute historical IXI outcomes.'
    text = '\n'.join(lines) + '\n'
    text += f'''\n{MARKER}
## J00/J02 completed — N07 execution and current target counts, 28 October 2026

N07/D04 is **prepared, not scientifically tested**: initial smoke1569366 FAILED, replacement1583591 TIMEOUT after30m04s, all scientific dependents CANCELLED before execution. Root blocker is the overlong multiprocessing TMPDIR socket path; the exact exception was reproduced on CPU. No target-domain training budget was finalized and no modern A-cls efficacy result exists. L06 remains partial (IXI failed empty-pair dtype indexing, SyntheticMRI artifact exists, joint collector cancelled). No GPU submission or runtime/model changes in this audit.

{table}
These are **actual serialized post-contraction nodes**, not old VTP counts; zero degree2 objects, exact manifest/index agreement and disjoint subjects verified. N07's fixed4096/512 subset still has one153-node training and one124-node validation target. Thus **120 is not safe for unchanged IXI**, independently of the obsolete239-node capacity argument. Keep256 for recovery; BRAVA development targets fit120. No object-capacity exclusions are recorded in these current manifests; existing IXI parallel/closed-branch exclusions remain disclosed.

One next bounded action: repair short SCRATCH multiprocessing TMPDIR and rerun the existing correctness gate, then recover the matched256-query B0/A-xyz/A-cls study if it passes. No new architecture/loss or120-query overflow filtering. Detailed chronology, initializer/query-copy convention and verified hashes are in the [existing IXI report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#n07d04-execution-and-current-target-capacity-audit--28-october-2026). Artifacts: `{HERE}`.
'''
    changes[REGISTRY] = text
    p = MAIN / 'research/roadmap.md'
    text = p.read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('| Local CNN A-xyz /A-cls /A |'):
            lines[i] = '| Local CNN A-xyz /A-cls /A | SyntheticMRI supports coordinate refinement; modern IXI standalone A-cls remains scientifically untested | **Audited28Oct:**1583591 smoke TIMEOUT (AF_UNIX path too long),1583592–1583594 CANCELLED; no completed planned-arm result. Repair short SCRATCH socket TMPDIR and revalidate gate before bounded recovery. Keep256: current contracted IXI max153, ten development patches exceed120. See Report3 dated audit |'
    text = '\n'.join(lines) + '\n'
    text += f'''\n{MARKER}
## Modern IXI execution/capacity gate — 28 October 2026

N07/D04 status is now **PARTIAL / RECOVERY**, not an active/unknown completed study. Both accepted chains stopped at smoke; the replacement failed on an overlong multiprocessing socket path and timed out. Scientific zero-shot/fine-tuning/collector jobs were cancelled before execution; no A-cls winner or efficacy conclusion is available. L06 remains partial, independently.

J00/J02 CPU audit is complete: current contracted IXI train/val10000/2649 targets max153/141, with five/five above120 and none above192. The fixed4096/512 N07 cohort also contains overflow targets. Zero degree2 objects were found; old239-node history is not the current capacity evidence. BRAVA8768/1796 targets max64/69 fit120. Preserve all targets and keep256 for N07 recovery; do not start the conditional full-IXI120/256 comparison on an invalid120 arm.

Next bounded action is the short SCRATCH TMPDIR repair and full correctness-gate recheck before existing matched component training. **No GPU jobs, architecture changes or new losses in this audit.** [Existing node-refinement report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#n07d04-execution-and-current-target-capacity-audit--28-october-2026); numerical artifacts `{HERE}`.
'''
    changes[p] = text
    p = MAIN / 'research/experiments.md'
    changes[p] = p.read_text() + f'''\n{MARKER}
## N07/D04 and J00/J02 — status/capacity audit, 28 October 2026

Modern IXI B0/A-xyz/A-cls scientific comparison **NOT RUN**: smoke1569366 FAILED; replacement1583591 TIMEOUT (AF_UNIX path too long), dependents1583592/1583593/1583594 CANCELLED before execution. No scientific checkpoint/results and no fixed final-update budget. No GPU work launched by this audit. L06 remains partial.

Actual contracted K8 train/val capacity: IXI10000/2649 patches, max153/141, five/five >120; N07 subset4096/512 includes one/one overflow. BRAVA8768/1796, max64/69, none >120. Zero degree2 object nodes; index/manifest/subject checks pass. Keep256 for unchanged IXI recovery. [Full existing report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#n07d04-execution-and-current-target-capacity-audit--28-october-2026). Evidence: `{HERE}`. No efficacy result should be inferred from the infrastructure failure.
'''

    integrations = []
    for path, after in changes.items():
        backup = HERE / 'before-integration' / path.relative_to(MAIN)
        backup.parent.mkdir(parents=True, exist_ok=True)
        assert not backup.exists(), 'Preserve earlier publication backup'
        before = path.read_bytes()
        backup.write_bytes(before)
        assert path.read_bytes() == before, 'Concurrent edit; integration deferred'
        path.write_text(after)
        integrations.append(dict(path=str(path), before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=sha(path)))
    (HERE / 'publication.json').write_text(json.dumps(integrations, indent=2) + '\n')

    claim = PROJECTS / 'experiment_gpu_coordination/ixi-local-components-20261026.json'
    before = claim.read_bytes()
    (HERE / 'original-gpu-claim.json').write_bytes(before)
    data = json.loads(before)
    assert data['jobs']['smoke']['id'] == '1583591'
    data.update(state='failed_smoke_no_active_jobs', concurrent_h100=0,
                audited_at=status['audited_at'], audit=str(HERE / 'status.json'),
                note='Smoke timeout; downstream cancelled. No GPU reservations remain; scientific experiment requires recovery.')
    assert claim.read_bytes() == before, 'Concurrent claim edit'
    claim.write_text(json.dumps(data, indent=2) + '\n')
    print('Published N07 status and J00/J02 capacity audit to four existing research documents.')


if __name__ == '__main__':
    run()
