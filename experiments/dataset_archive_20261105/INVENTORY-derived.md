# Derived training caches in STORE, 2026-11-05

**Status: DONE.** All 8 archives below are packed and verified. The user approved this follow-up to the originals backup (`$STORE/datasets-originals-20261105/`).

- Location: `$STORE/datasets-derived-20261105/` = `/lustre/fsstor/projects/rech/vnc/upz25mj/datasets-derived-20261105/`.
- Source: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets`, which was only read.
- Why: SCRATCH (fsn1) purges unread files, and it has already deleted IXI test images, IXI loop patches and the IXI split. These caches are expensive or impossible to rebuild exactly.

Terms used here:
- **K8 cache**: a `transfer_splines_k8*` / `transfer_source_k8*` folder made by the transfer-spline generator.
- **Split folders**: its `train/`, `val/` and (where present) `test/` folders, each with `raw/ seg/ vtp/ spline/`. Their entries are symlinks into the cache's `candidates/`, which link on to patch files elsewhere.

## How it was made

- Slurm script: `DenseVesselGraph/cluster/jean_zay/archive_dataset_derived.slurm`, run on partition `archive`, account `vnc@cpu`, with 4 CPUs.
- Packing: `tar --dereference --hard-dereference -T members-derived/<name>.txt | zstd -T4 -3`.
  - Every symlink is followed, so the archive holds the patch **content** at the split-folder paths (for example `IXI/transfer_splines_k8_20261015/train/raw/sample_..._data.nii.gz`).
  - `--hard-dereference` stores a full copy even when two paths reach the same inode, so the archive has no tar hard-link entries.
  - After extraction the split folders are plain files and the loaders can read them directly.
- Members of a K8 cache: its split folders plus all of its top-level regular files: `manifest.json`, `patch_index.csv`, `generation_config.json`, `source_manifest.json`, `READY`, and where present `monitoring.json` (IXI) and `brava_semantics.json` (Brava).
- **`candidates/` is never included**: it would store every patch twice. The script refuses any member list that names it.
- Member lists: `DenseVesselGraph/experiments/dataset_archive_20261105/members-derived/`.
- The two TopBrain_tight cache archives each contain both of their caches, `mr/` and `ct/`.

## Verification (every archive)

1. Before packing: no dangling links in the members (`find -L -type l` = 0).
2. Archived regular files == `find -L` regular files of the members. For the K8 caches this also equals (symlinks in the split folders + top-level files), since all links there point to files. The archive contains no symlink or hard-link entry.
3. `zstd -t` integrity test passed.
4. `tar --diff --dereference` against the source: no difference in content, size, mode or mtime. The one exception is listed under TopBrain_tight below.
5. Spot check: 8 random members were extracted to `$SCRATCH/tmp/dsd-<name>-<job>` and compared with `cmp` against the link-followed source. All were identical. Only that temporary folder was deleted afterwards.
6. sha256 of the archive is recorded here and in `SHA256SUMS`. A separate job re-checked all 8 with `sha256sum -c` after the final rename (job 2003725).

Logs, full `tar -tv` listings and diff outputs: `DenseVesselGraph/experiments/dataset_archive_20261105/logs/`.

## Archives

Source GB = dereferenced bytes (`du -L` equivalent), in 10^9 bytes.

| Archive | Source GB | Archive GB | links + regular in source | deref files src / archived | tar --diff | spot cmp | sha256 | Job |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | ---: |
| `Brava-prepared_swc_20261020-132057-transfer_splines_k8.tar.zst` | 2.93 | 2.86 | 50428 + 6 | 50434 / 50434 | OK | 8/8 OK | `66cb0ea253aaa9f7d6cd952632134bb08282b763732724f6549da18f3b2601f0` | 1998882 |
| `IXI-transfer_splines_k8_20261015.tar.zst` | 4.95 | 4.87 | 59308 + 6 | 59314 / 59314 | OK | 8/8 OK | `2a6ed23c1d69015cfa11f940a2ada192d5a2064f8cd44acc11133f7f15779869` | 1998880 |
| `IXI-transfer_splines_k8_loops_20261105.tar.zst` | 4.43 | 4.35 | 51844 + 6 | 51850 / 51850 | OK | 8/8 OK | `522d4373462e7926f2cfad88cd23d6d517ba3f48970abe356d9b9139b3808957` | 1998881 |
| `syntheticMRI-transfer_source_k8_25k_20261015.tar.zst` | 6.63 | 6.54 | 101920 + 5 | 101925 / 101925 | OK | 8/8 OK | `71d14372d283b6ad7877de56c50dd0d946bd11b5c857fe01b757651e94002896` | 1998877 |
| `syntheticMRI-transfer_source_k8_25k_loops_20261105.tar.zst` | 6.64 | 6.54 | 101920 + 5 | 101925 / 101925 | OK | 8/8 OK | `94d49c58070ae5960dea1484ff60323881be5c8ed04365b73d39105ba7f49812` | 1998878 |
| `TopBrain_tight_20261105-transfer_splines_k8_20261105.tar.zst` (mr + ct) | 2.32 | 2.27 | 38152 + 10 | 38162 / 38162 | OK | 8/8 OK | `b7c3dd17a9e06bd274059b2198145a547cfa8b7df8be1580ee5584c9a72dede5` | 1998625 (smoke test) |
| `TopBrain_tight_20261105-transfer_splines_k8_loops_20261105.tar.zst` (mr + ct) | 2.49 | 2.43 | 40384 + 10 | 40394 / 40394 | OK | 8/8 OK | `051012a1dc02e7fcc80d88ba6aee7e44d3dc171f2f3531d943fb8658ad43faa7` | 1998883 |
| `TopBrain_tight_20261105.tar.zst` | 6.81 | 6.42 | 150 + 86617 | 86867 / 86867 | 150 not stat-able, then byte-compared: 150/150 OK | 8/8 OK | `748a0b3bcfb853c0d3b6eb3736bf378fe9cf1aac545fa727525816e3f3500d59` | 1999211 (+ 2003615) |
| **Total** | **37.20** | **36.28** | | 530,871 files | | | | |

Splits per cache:

| Cache | Splits |
| :--- | :--- |
| IXI k8 | train, val, test |
| IXI loops | train, val |
| syntheticMRI (both caches) | train, val |
| Brava | train, val, test |
| TopBrain_tight (all four mr/ct caches) | train, val, test |

Excluded `candidates/` folders (the link layer between the split folders and the patch files; `du -sb`):

| Cache | Size |
| :--- | ---: |
| IXI k8 | 0.59 GB |
| IXI loops | 0.63 GB |
| syntheticMRI k8 | 0.82 GB |
| syntheticMRI loops | 0.86 GB |
| Brava | 0.26 GB |
| TopBrain k8 | 0.09 (mr) + 0.07 (ct) GB |
| TopBrain loops | 0.10 (mr) + 0.08 (ct) GB |

### TopBrain_tight_20261105 (our derived dataset, without its `transfer_splines_*` folders)

Members:
- `images_ct`, `images_mr`: skull-stripped images, 25 each, 1.08 GB;
- `labels_ct`, `labels_mr`: clipped labels;
- `masks_ct`, `masks_mr`;
- `manifest.csv`, `patient_split_{ct,mr}.csv`, `preparation.json`;
- `vascular_graphs` (500 files, 0.14 GB);
- `vascular_patches` (85,959 files, 4.44 GB);
- `vascular_graph_sources` (4 files + 150 symlinks, dereferenced).

Only `vascular_graph_sources` contains symlinks, so the global `-h` dereferences only that folder. Its 150 links are of two kinds:
- 100 file links to `images_*` / `labels_*`;
- 50 **directory** links to `vascular_graphs/.../graphs/adaptive`, holding 3 files each.

That is why the dereferenced count is 86,617 + 100 + 150 = 86,867, not links + regular. Dereferencing stores these files a second time, as full copies: about 1.1 GB of the 6.81 GB.

Two attempts failed before the final archive:
- **Job 1998879 (FAILED, by design of the checks):** without `--hard-dereference`, tar stored the 250 second paths as hard-link entries pointing at `images_*`, `labels_*` and `vascular_graphs` inside the archive. The file count did not match, so the job refused to finalize. I then added `--hard-dereference` to the script.
- **Job 1999211:** counts matched and the spot check passed. `tar --diff --dereference` could not stat the 150 files reached **through the directory symlinks** ("Cannot stat: Invalid cross-device link", a tar compare-mode limitation with absolute directory links; the files stat normally). The job therefore kept the archive as `.partial`.

Job 2003615 (`cluster/jean_zay/verify_archive_members.slurm`) then checked the partial archive:
- the sha256 matches;
- `zstd -t` passed;
- all 150 of those members, extracted and compared with `cmp`, are identical to the source (list: `logs/TopBrain_tight_20261105.diff-exceptions.txt`).

Only then was the archive renamed to `TopBrain_tight_20261105.tar.zst`.

## Restore

`zstd -dc <name>.tar.zst | tar -xf - -C <dest>`. Paths start at the dataset folder, for example `IXI/...`. The split folders come back as regular files: no `candidates/`, no links.

## Quota

- `idr_quota_user` (refreshed daily) still showed the pre-backup figures: STORE user 142.04 GiB and 58 inodes.
- Measured with `du -sb`: `datasets-originals-20261105` = 120.75 GB, which includes octa-synth-packed. This folder adds 36.27 GB (33.8 GiB) in 11 inodes (folder, 8 archives, `SHA256SUMS`, `INVENTORY.md`).
- Expected STORE user total after the refresh: about 142.0 + 112.5 + 33.8 ≈ 288 GiB of 50 TiB, and about 88 inodes.
