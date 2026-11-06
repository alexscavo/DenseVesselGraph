# Original-dataset backup in STORE, 2026-11-05

**Status: DONE.** All 14 archives below were packed and verified (job IDs at the end).

- Location: `$STORE/datasets-originals-20261105/` = `/lustre/fsstor/projects/rech/vnc/upz25mj/datasets-originals-20261105/`.
- Source: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets` (fsn1/SCRATCH, which purges untouched files). The source was only read: nothing there was moved, modified or deleted.
- Scope: the original images and the original annotations, labels or tracings of each release. Also included: the colleague's preprocessed TopCoW, TubeTK and SMILE-UHURA, which we cannot reproduce. Derived products (patches, extracted graphs, spline caches, staged symlink folders, re-extractions) are excluded.

## How it was made and verified

- Slurm script: `DenseVesselGraph/cluster/jean_zay/archive_dataset_originals.slurm`, run on partition `archive`, account `vnc@cpu`, with 4 CPUs.
- Packing: `tar -C datasets -cf - -T members/<name>.txt | zstd -T4 -3`. Paths inside each archive start at the dataset folder, for example `TopCoW/TopCoW_raw/...` or `AV Paper Online Files/...`. Symlinks are stored as links, not followed; none of the archived folders contained any.
- Member lists: `DenseVesselGraph/experiments/dataset_archive_20261105/members/<name>.txt`.
- Each archive had to pass all of the following checks:
  1. `zstd -t` integrity test;
  2. archive entries == `find` entries of the source, regular files == source regular files, and symlinks == source symlinks;
  3. `tar --diff` of the archive against the source reported **no** difference in content, size, mode or mtime (`logs/<name>.diff.txt` files are all empty);
  4. sha256 of the archive recorded here and in `SHA256SUMS`, then re-checked with `sha256sum -c` after the final rename (job 1989693).

  The archive was written as `.<name>.tar.zst.partial` and renamed only after checks 1–3 passed.
- Full per-archive file listings (`tar -tv`): `DenseVesselGraph/experiments/dataset_archive_20261105/logs/<name>.listing.txt`.
- The smallest archive (AV Paper Online Files) was also extracted back from STORE and compared byte for byte (`diff -r`) with the source: identical.

**Restore:** `zstd -dc <name>.tar.zst | tar -xf - -C <dest>`. The archive partition (or a login node) can read STORE; cpu_p1 compute nodes cannot write to it.

## Archives

Sizes in GB (10^9 bytes). "files" = regular files; "entries" = files + directories (+ links).

| Archive | Source GB | Archive GB | Ratio | files src / arc | entries src / arc | sha256 |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `AV_Paper_Online_Files.tar.zst` | 0.07 | 0.07 | 95.8% | 115 / 115 | 121 / 121 | `4414301475ffd23678214222fd3aa35968880a65697485b27a3fbcc9b73b16fe` |
| `Brava.tar.zst` | 1.08 | 1.05 | 98.1% | 235 / 235 | 238 / 238 | `c74dff1051e683b31bf23d2bc90087a4b9da9357e11688aa559574a7335f3d9c` |
| `HALVS_DATASET.tar.zst` | 30.35 | 30.34 | 100.0% | 5358 / 5358 | 5369 / 5369 | `d62074756ef5e75201143dc39c9754ed8387bb2248943b29fd0aa52bfaf1f745` |
| `IXI.tar.zst` | 4.50 | 4.50 | 100.0% | 319 / 319 | 321 / 321 | `ffcbd5caf1acbe64f3c5bf9171439eb11e791e527da14ad9b1791387cabd4b13` |
| `plants_3d2cut.tar.zst` | 5.28 | 5.18 | 98.0% | 3022 / 3022 | 3024 / 3024 | `3469ee38eb7cab90a4e65c17817d3834e562af5b9d1bc2eef9b16565279567c7` |
| `REAVER_Vascular_Networks_Image_Dataset.tar.zst` | 0.35 | 0.07 | 19.9% | 272 / 272 | 280 / 280 | `2a62da2df04074470b2c138b38631b8e282837147424acb5fa5e17a18bfcd735` |
| `RETA_16960855.tar.zst` | 0.88 | 0.88 | 99.9% | 8 / 8 | 9 / 9 | `a8c5ef0f14a589a8acdcff0084f56edd2bcf7d488eaa32654179beba390efed1` |
| `SMILE-UHURA_preprocessed.tar.zst` | 2.61 | 2.46 | 94.3% | 1485 / 1485 | 1626 / 1626 | `b2ff0268e61b9b9806b8a96ff481b81475d89c56cb097f91f3f2c1db3703f997` |
| `SMILE-UHURA_raw.tar.zst` | 0.74 | 0.74 | 99.9% | 34 / 34 | 41 / 41 | `5f5aa420af58b9d877dc29d2b8a56ca696c43316ceebd8d3768b16a76a261057` |
| `TopBrain.tar.zst` | 1.97 | 1.96 | 99.7% | 105 / 105 | 113 / 113 | `c8792a58d40af6eeccf89cc4a589181b827745718a7cd991d3d239cdfcf60622` |
| `TopCoW_preprocessed.tar.zst` | 39.78 | 39.46 | 99.2% | 13565 / 13565 | 15866 / 15866 | `131aec99320a7073f00cda5bc15836bd9f30b7deb5c2d7384be87be9e2853c05` |
| `TopCoW_raw.tar.zst` | 10.51 | 10.47 | 99.6% | 1012 / 1012 | 1018 / 1018 | `71f0f64a62e834d2c0b0aab8bb9656e0c67205e5018057eb9444293e5170a813` |
| `TubeTK_preprocessed.tar.zst` | 16.65 | 15.63 | 93.8% | 10337 / 10337 | 11368 / 11368 | `204fd2b1a230ede86b3738ea0a92f9fe992ab7472e00c8396997d3a0a6c957fb` |
| `TubeTK_raw.tar.zst` | 6.75 | 6.36 | 94.3% | 700 / 700 | 1341 / 1341 | `0d980e63c829720339d64ca090546afb458a8eca24f719aa5cadba4884c93d8e` |
| **Total** | **121.53** | **119.18** | 98.1% | 36567 | | |

Compression gains little because the sources are already compressed (`.nii.gz`, `.jpg`, `.rar`, `.mha` with zlib).

### What each archive contains (all paths relative to `datasets/`)

| Archive | Members | What it is |
| :--- | :--- | :--- |
| AV_Paper_Online_Files | `AV Paper Online Files/` (whole) | Estrada et al. 2015 AV-WIDE / AV-DRIVE / AV-INSPIRE `.mat` graphs with AV labels (AV-WIDE also has the colour image), README, MATLAB viewers |
| Brava | `Brava/brains`, `Brava/swc_files`, `Brava/mask_hull` | 56 BraVa MRA volumes (2013 release), 123 SWC tracings. `mask_hull` holds the 56 brain masks **supplied** to us (per `docs/research/BRAVA_preparation.md`); they are not made by our scripts. |
| HALVS_DATASET | `README.md`, `annotation/`, `original_image/` | HALVS leaf-vein release: 5,057 leaf images, plus 300 annotated images and masks |
| IXI | `IXI/imagesTr`, `IXI/labelsTr` | IXI MRA images (149) and vessel labels (170); see the issue noted below |
| plants_3d2cut | `plants_3d2cut/raw`, `plants_3d2cut/graphs` | 3D2Cut grapevine photos (1,511 `.jpg`) and their original annotation JSONs (`*_annotation.json`, "3D2Cut Grapevine Annotation") |
| REAVER_Vascular_Networks_Image_Dataset | whole folder | REAVER release: `_Original_Images`, `Manual` segmentations, outputs of 5 automatic methods (part of the release), CSV, summary docx |
| RETA_16960855 | `16960855/` (whole) | RETA benchmark (Zenodo record 16960855): 6 `.rar` parts (codes, images, mats, models, software, supports) and Readme |
| SMILE-UHURA_raw | `SMILE-UHURA/SMILE-UHURA_raw` | 7T TOF MRA as downloaded from Synapse (train/validate/test images and GT, Synapse manifests) |
| SMILE-UHURA_preprocessed | `SMILE-UHURA/SMILE-UHURA_preprocessed`, plus `README.md`, `participants.tsv`, `manifest.tsv`, `graph_metrics.tsv` of `SMILE-UHURA/` | Colleague's preprocessing (shared output contract) |
| TopBrain | `TopBrain/` (whole) | TopBrain 2025 batch-1+2 release (CT/MR images and labels, ITK-SNAP labelmaps, License, README). `note.txt` says that the Te CT split is artificial (moved from Tr). |
| TopCoW_raw | `TopCoW/TopCoW_raw` | TopCoW 2024 release as downloaded (imagesTr/Val, labelsTr, ROI and edge labels, License, README) |
| TopCoW_preprocessed | `TopCoW/TopCoW_preprocessed`, plus the 4 top-level `TopCoW/*.md,*.tsv` | Colleague's preprocessing with brain masks; not reproducible by us |
| TubeTK_raw | `TubeTK/TubeTK_raw` | TubeTK "Normal" MRA database as downloaded (109 subjects, `.mha`, Skwerer/Bullitt trees). The colleague renamed 3 misnamed folders, as recorded in their README. |
| TubeTK_preprocessed | `TubeTK/TubeTK_preprocessed`, plus the 4 top-level `TubeTK/*.md,*.tsv` | Colleague's preprocessing |

## Skipped

### Derived subfolders inside archived datasets

| Path | Size | Files (links) | Why skipped |
| :--- | ---: | ---: | :--- |
| `Brava/prepared_swc_20261020-132057` | 4.09 GB | 75,993 (76,065) | Our preparation: volumes, patches, splines, runtime |
| `HALVS_DATASET/patches_3d_transfer_20261015` | 0.80 GB | 304,577 (50,976) | Our 64³ transfer patches |
| `IXI/vascular_patches` | 5.15 GB | 34,294 | Patches |
| `IXI/vascular_graphs` | 0.18 GB | 659 | Extracted graphs |
| `IXI/vascular_graph_sources` | ~0 | 1 (510) | Staged symlinks |
| `IXI/transfer_splines_k8_20261015*` (3 folders) | 0.60 GB | 34,345 (93,664) | Spline caches |
| `plants_3d2cut/seg` | 0.07 GB | 1,511 | `*_seg.png` rasterised from the annotation JSONs; regenerable. Not part of the 3D2Cut release as far as we can tell (see "Needs a decision") |
| `plants_3d2cut/train`, `plants_3d2cut/val` | 5.35 GB each | 4,533 each | Full copies of `raw`/`seg`/`graphs` (all 1,511 names in each; `cmp` identical on a sample) |
| `plants_3d2cut/patches`, `patches_3d`, `prova` | 1.09 GB | 234,507 | Patches and a test sample |
| `plants_3d2cut/splits.csv` | 40 kB | 1 | Our split |

### Whole top-level folders

| Folder | Size | Why skipped |
| :--- | ---: | :--- |
| `20cities_2d`, `20cities_3d` | 0 files | **Already purged**: only empty `train/val/test/{raw,seg,vtp}` directories remain. The 42 MB / 31 MB that `du` reports is directory metadata. There is nothing to archive. |
| `biological_mixture_20261015` | 28 MB | 4 files and 101,952 symlinks into `experiments/gnbm/...`: a staged mixture, derived |
| `TopBrain_Data_Release_Batches1n2_081425` | 0 files | Empty skeleton (150 symlinks only); its files were purged (as instructed) |
| `TopBrain_stripped_20261104`, `TopBrain_full_20261104`, `TopBrain_tight_20261105` | 6.0 GB / 68 kB / – | Derived by us |
| `syntheticMRI.before-restore-627540` | ~0 (3 files) | The pre-restore shell of syntheticMRI; its data are gone |
| `.syntheticMRI-restore-627391` | empty | Leftover staging folder |
| `.transfer-helix-20261104` | 4.8 MB | Our checksum JSON and verify script |
| `syntheticMRI` | see below | Needs a decision |
| `octa-synth-packed` | 1.0 GB | Needs a decision |

## Needs a decision (not archived)

1. **syntheticMRI**: a generated dataset of 136 subjects. What is present:
   - `seg/`: 136 segmentation volumes, 0.09 GB;
   - `graphs/`: 136 × nodes/edges/`graph.vvg` files, 3.30 GB;
   - `raw/`: only **one** image (`10.nii.gz`), 0.07 GB; the other raw volumes are already missing;
   - derived: `new_patches` (30.4 GB, 391,680 files), `new_patches_boundary` (30.5 GB), `transfer_source_k8_25k_20261015` (0.85 GB, 158,888 links), `results`.

   These same `seg`, `graphs` and `raw` files (identical file counts) are **already in STORE**, inside `archives/2026-10-06/source-datasets-2026-10-05.tar`. That 46 GB **uncompressed** tar also holds the patches; it is the restore source of `cluster/jean_zay/restore_synthetic_mri.slurm`. Recommendation: no new archive is needed. If faster recovery is wanted, a compact `syntheticMRI_sources.tar.zst` (seg + graphs + raw, about 3.5 GB) costs one short archive job.
2. **octa-synth-packed**: synthetic OCTA dated 2025-10-28/11-07. It holds `raw` (500 PNG, 0.82 GB), `labels` (500, 0.09 GB), `graphs` (500 × nodes/edges, 0.05 GB), `splits.csv`, and derived `patches` (10,800 files, 0.09 GB). We found no generator or source for it in the workspace. If it cannot be regenerated, archive `raw` + `labels` + `graphs` + `splits.csv`: about 0.96 GB, 1 job, under 1 min.
3. **plants_3d2cut/seg** (0.07 GB): include it only if the `_seg.png` masks turn out to come with the 3D2Cut release, or to be costly to regenerate.

## Issues noticed in the source (not changed)

- **IXI**: `labelsTr` has 170 labels but `imagesTr` has only 149 images. 21 images with labels are missing from `imagesTr`, for example `IXI035-IOP-0873`, `IXI054-Guys-0707`, `IXI388-IOP-0973` (full list: `comm -13 <(ls imagesTr) <(ls labelsTr)`). The `imagesTr` directory mtime is 2026-11-03 15:01, so they were removed or purged recently. The originals are downloadable from the IXI site.
- **syntheticMRI/raw** holds only 1 of the 136 raw images (it was already so in the 2026-10-05 STORE tar).

## Quota

- Before (idr_quota, refreshed daily): STORE user 142.04 GiB, 58 inodes; group 770.17 GiB of 50 TiB, 1,307 of 100,000 inodes.
- Added: 119.18 GB (111.0 GiB) and 17 inodes: 14 archives, `SHA256SUMS`, this `INVENTORY.md`, and the folder.

## Jobs (sacct)

| Job | Name | State | Elapsed |
| :--- | :--- | :--- | ---: |
| 1989286 | ds-AV_Paper_Online_Files (smoke test) | COMPLETED | 0:06 |
| 1989329 | ds-TopCoW_preprocessed | COMPLETED | 8:03 |
| 1989330 | ds-HALVS_DATASET | COMPLETED | 6:22 |
| 1989331 | ds-TubeTK_preprocessed | COMPLETED | 4:19 |
| 1989332 | ds-TopCoW_raw | COMPLETED | 1:45 |
| 1989333 | ds-IXI | COMPLETED | 0:35 |
| 1989334 | ds-plants_3d2cut | COMPLETED | 1:50 |
| 1989335 | ds-TubeTK_raw | COMPLETED | 1:05 |
| 1989336 | ds-SMILE-UHURA_preprocessed | COMPLETED | 0:28 |
| 1989518 | ds-SMILE-UHURA_raw | COMPLETED | 0:09 |
| 1989530 | ds-Brava | COMPLETED | 0:12 |
| 1989531 | ds-TopBrain | COMPLETED | 0:16 |
| 1989532 | ds-REAVER_Vascular_Networks_Image_Dataset | COMPLETED | 0:06 |
| 1989562 | ds-RETA_16960855 | COMPLETED | 0:08 |
| 1989693 | ds-sha256-recheck | COMPLETED, 14/14 OK | 6:57 |

## Addendum 2026-11-05: IXI test images
IXI.tar.zst was packed while 21 IXI test images were purged from SCRATCH (149/170 in imagesTr). They were restored and verified voxel-for-voxel against 2,178 September test patches, and are archived separately in `IXI_test_images_restored_20261105.tar.zst` (21 files).

## Addendum 2026-11-05 (later): octa-synth-packed

**Status: DONE.** The user approved archiving the non-patch content of `octa-synth-packed`, which settles item 2 of "Needs a decision". The checks are the same as above: counts, `zstd -t`, empty `tar --diff`, and the sha256 appended to `SHA256SUMS` and re-checked.

| Archive | Source GB | Archive GB | Ratio | files src / arc | entries src / arc | sha256 |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `octa-synth-packed.tar.zst` | 0.97 | 0.93 | 96.2% | 2001 / 2001 | 2504 / 2504 | `18973b4d8740b4274c00c848cdc76194e9f8ab51c35b7ab324608f8e96227344` |

- Members: `octa-synth-packed/raw` (500 PNGs), `labels` (500), `graphs` (500 × nodes/edges CSV) and `splits.csv`.
- Skipped: `patches/` (10,800 files, 0.09 GB, derived).
- Job: 1998214, COMPLETED in 0:23.
