# AG-VPReID bidirectional evaluation and platform-aware E4

Run commands from `H:\WSY\ReID\VSLA-CLIP-master` in the `ReID` environment.
The feature is optional: existing G2A/MARS configurations keep their camera
filtering behavior. This change does not establish an experimental improvement.

## Sampling rule

`DYNAMIC_DFGS_STRATIFIED_PLATFORM: True` derives a second label from the real
camera ID using `DYNAMIC_DFGS_CAMERA_TO_PLATFORM: [0, 0, 0, 0, 1, 1]`:

| Cameras | Platform ID | Meaning |
| --- | --- | --- |
| C0, C1, C2, C3 | 0 | Ground: CCTV and wearable cameras |
| C4, C5 | 1 | Aerial: UAV cameras |

The camera mapping follows the [official AG-VPReID repository](https://github.com/agvpreid25/AG-VPReID).
Tracklets are compared within platform/length groups using the existing two-view
reliability and length bins. An identity is excluded from the hard-negative
candidate graph only when it has at least one low-reliability tracklet on each
of the two platforms. An identity seen on only one platform stays eligible.
Excluded identities remain in training and can still be anchors or appear in
random batches. Visual centroids, PK tracklet sampling, and losses are unchanged.

PBP still receives the six actual camera labels. Platform IDs are used only
inside the reliability filter; the data tuples and evaluation camera IDs remain
unchanged. This is a sampling variant, not a new platform prompt or a new
cross-platform positive sampler.

Two-camera, multicamera, and platform filtering are mutually exclusive. Missing
camera mappings, single-platform training sets, and non-finite features or
reliabilities in the new platform branch produce explicit errors. MARS should
continue using its camera configuration, not six identical platform labels.

## Dataset reader and audit

`DATASETS.NAMES: 'ag_vpreid'` supports the official directory names and the
`groud` spelling found in the local package. The prepared configurations point
explicitly to `case2_groud_to_aerial`. The reader checks the actual C labels:
G2A queries must be ground, and galleries must be aerial. It never fabricates
an A2G split by renaming directories or swapping query and gallery.

Person identity comes from the annotated parent folder, following the
[official dataset reader](https://github.com/agvpreid25/AG-VPReID/blob/main/baselines/tfclip/datasets/set/agreidvid.py).
Only training identities are relabelled to consecutive indices. Query and
gallery retain the same original folder IDs; they are never independently
relabelled. The filename P field is not assumed to be a globally unique identity.
All frames in a tracklet must share P/C metadata, and frame order uses numeric
F indices. The reader checks train/test folder-ID overlap and missing gallery
matches, and retains gallery distractors.

Read-only audit:

```powershell
python -B scripts/audit_ag_vpreid.py --root ../datas/AG-VPReID --direction g2a
```

The local package audit on 2026-10-20 found:

| Split | Annotated IDs | Tracklets | Frames |
| --- | ---: | ---: | ---: |
| Train | 1,604 | 13,300 | 3,853,034 |
| G2A query | 1,459 | 6,256 | 2,589,509 |
| G2A gallery | 5,078 | 12,775 | 3,206,170 |

All 1,604 training IDs cover both platforms. Annotated train/test folder IDs
are disjoint; all query IDs have a gallery match. There are 3,619 gallery-only
identities. Query and gallery each contain 113 parent folders with more than
one filename P code; that is reported, not silently converted into new labels.
These structural checks do not certify the semantic identity annotations or
equivalence to the CVPR/challenge protocol. Report results as the local G2A
split until its provenance and annotation differences are resolved.

The A2G download was checked on 2026-10-22. Its query/gallery ZIP sizes match
the official full release, and the extracted file-name/size manifests match
both ZIPs. A2G query has 1,459 IDs, 7,320 tracks and 1,355,138 frames; A2G
gallery has 1,459 IDs, 6,256 tracks and 2,589,509 frames. This supports local
bidirectional comparisons, while the ID/track-count differences from paper
Table 3 remain unresolved.

## Three matched configurations

All three use PBP, sequence length 8, batch size 16 (P4K4), seed 1234, 120
Stage-2 epochs, and 4 sampled tracklets per identity per epoch. The two E4
configurations share the random baseline's saved Stage-1 checkpoint. The
random arm uses the same PK sampler implementation with graph selection off.
With 1,604 identities, all three have 401 Stage-2 batches per epoch. This new
control differs from the historical MARS random baseline's natural epoch size.

First train the random arm, which saves Stage-1 as well as Stage-2 checkpoints:

```powershell
python train_reidadapter.py --config_file configs/adapter/reproduce/agvpreid_g2a_vifi_pbp_p4k4_seed1234_lr2e4.yml
```

Then train the camera and platform variants from that same Stage-1 checkpoint:

```powershell
python train_reidadapter.py --config_file configs/adapter/innovation/agvpreid_g2a_E4_camera_pbp_p4k4_seed1234_lr2e4.yml
```

```powershell
python train_reidadapter.py --config_file configs/adapter/innovation/agvpreid_g2a_E4_platform_pbp_p4k4_seed1234_lr2e4.yml
```

Each writes to a distinct new output directory. Reusing a previously used
output directory follows the existing training script's behavior, so choose
new output paths for additional seeds or runs.

## Dual-direction testing and output layout

The three YAML file names retain their historical `agvpreid_g2a_` names so
existing config commands remain valid. Their `OUTPUT_DIR`, test checkpoint
paths, and shared Stage-1 checkpoint path now use direction-neutral names:

```text
output/reproduce/agvpreid_vifi_pbp_p4k4_seed1234_lr2e4/
output/innovation/agvpreid_E4_camera_pbp_p4k4_seed1234_lr2e4/
output/innovation/agvpreid_E4_platform_pbp_p4k4_seed1234_lr2e4/
```

Each directory contains its original training log/checkpoints plus
`test_log_g2a.txt` and `test_log_a2g.txt`. Existing G2A/A2G test logs have
been relocated without changing their contents. Historical paths printed
inside those logs remain as originally recorded. Repeated AG tests append
to the direction-specific log instead of overwriting earlier results.

Each configuration now sets `TEST.AGVPREID_DIRECTIONS: ['g2a', 'a2g']`.
`test.py` loads one checkpoint and tests both directions in sequence:

```powershell
python test.py --config_file configs/adapter/reproduce/agvpreid_g2a_vifi_pbp_p4k4_seed1234_lr2e4.yml
python test.py --config_file configs/adapter/innovation/agvpreid_g2a_E4_camera_pbp_p4k4_seed1234_lr2e4.yml
python test.py --config_file configs/adapter/innovation/agvpreid_g2a_E4_platform_pbp_p4k4_seed1234_lr2e4.yml
```

To test only one direction, override the TEST list (PowerShell example):

```powershell
python test.py --config_file configs/adapter/innovation/agvpreid_g2a_E4_platform_pbp_p4k4_seed1234_lr2e4.yml TEST.AGVPREID_DIRECTIONS "['a2g']"
```

`DATASETS.AGVPREID_DIRECTION: 'g2a'` and its split directory still determine
training-time checkpoint selection. The TEST list affects standalone
evaluation only, so the saved best checkpoints retain their G2A mAP selection
rule. No retraining is needed. The opposite test directory is discovered by
the reader using its existing supported names, including `groud` spellings.

## Verification

```powershell
python -B scripts/test_platform_dfgs.py
```

```powershell
python -B scripts/test_multicamera_dfgs.py
```

The platform tests cover the difference between low scores on two ground
cameras and low scores on both platforms, partial platform coverage, invalid
labels, non-finite reliability, identity parsing, direction checks, matched
PK budgets, and graph refresh/serialization with real camera labels preserved.
`test_platform_dfgs.py --temp-dir PATH` can use an explicitly writable directory
for its small synthetic fixtures in restricted Windows environments.

`scripts/smoke_platform_dfgs.py` is an optional GPU probe using 80 real training
tracklets and one query/gallery batch. It creates only a new probe directory
and does not run optimization or save a trained checkpoint. Its initialized
prompts do not provide a performance result.
