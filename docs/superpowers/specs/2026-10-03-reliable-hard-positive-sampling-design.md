# Reliable Hard Positive Tracklet Sampling Design

## Objective

Add a platform-label-free positive-tracklet sampling method for the G2A-VReID video ReID pipeline. The method must select same-identity tracklets that are different in feature space but temporally reliable, rather than blindly choosing the farthest positive. It must support both random identity batching and the existing DFGS-TV hard-negative identity batching.

The experimental baseline remains the original paper's video PK sampler: random identities, random K tracklets per identity, and full with-replacement sampling when an identity has fewer than K tracklets.

## Scope

This change will:

1. Remove the cross-camera sampling mode and its tests.
2. Preserve `camid` in dataset records because the model and evaluation pipeline may still consume it.
3. Add an offline builder that extracts frame-level features from the baseline Stage-1 checkpoint and writes reliable-hard-positive relationships.
4. Add a reliable-hard-positive sampler that does not inspect camera, platform, or altitude labels.
5. Add one RHP-only configuration and one RHP plus DFGS-TV configuration.
6. Add focused unit tests for baseline compatibility, reliability-aware selection, fallback behavior, and DFGS-TV composition.

Historical output directories, logs, checkpoints, and result files are outside the deletion scope.

## Terminology

- **Tracklet center**: the normalized mean of the selected frame features for one video tracklet.
- **Temporal reliability**: the mean cosine similarity between each normalized frame feature and the tracklet center.
- **Hardness**: cosine distance between two tracklet centers belonging to the same identity.
- **Reliable Hard Positive (RHP)**: a same-identity candidate with a high product of hardness and pair reliability.
- **DFGS-TV**: the existing text-visual identity graph used to choose difficult negative identities in a mini-batch.

## Baseline Preservation

`DATALOADER.SAMPLER: 'softmax_triplet'` must continue to instantiate `RandomIdentitySampler_Video` directly. No RHP graph, DFGS graph, camera constraint, or additional sampling field may affect this path.

The baseline behavior for identity `y` with tracklet index list `I_y` and K instances is:

- if `|I_y| >= K`, shuffle the real indices and emit complete groups of K;
- if `|I_y| < K`, draw K indices from `I_y` with replacement;
- shuffle the resulting identity chunks globally.

## Removal of Cross-Camera Sampling

Remove the following implementation paths:

- sampler name `cross_camera`;
- sampler name `dfgs_cross_camera`;
- the `cross_camera` constructor argument;
- the camera-to-index dictionary used only by sampling;
- `_cross_camera_chunks`;
- cross-camera-specific unit tests.

Do not remove `camid` from `(img_paths, pid, camid)` dataset tuples or model calls.

## Offline Feature Extraction

### Inputs

The builder accepts:

- a training configuration;
- the baseline Stage-1 checkpoint;
- an output JSON path;
- the configured sequence length, normally 8;
- an inference batch size.

The initial G2A experiment uses:

```text
output/reproduce/vifi_no_pbp_baseline_seed1234_lr2e4_retrain/
ViT-B-16_stage1_120.pth
```

### Deterministic Frame Selection

Feature construction must be repeatable. For each tracklet, select `SEQ_LEN` positions deterministically across its temporal extent. If the tracklet is shorter than `SEQ_LEN`, repeat frames deterministically until the required length is reached. Do not use random temporal cropping, random frame shuffling, horizontal flipping, random erasing, or random cropping.

Apply the normal resize, tensor conversion, and normalization used by evaluation.

### Frame Features

Expose a model path used only by the offline builder that returns the projected CLIP feature for every selected frame before temporal averaging. For a batch of B tracklets and T frames, it returns shape `[B, T, D]`.

Normalize every frame feature:

```text
u_i,t = f_i,t / ||f_i,t||_2
```

Compute the tracklet center:

```text
c_i = normalize(mean_t(u_i,t))
```

Compute temporal reliability:

```text
r_i = clamp(mean_t(cos(u_i,t, c_i)), 0, 1)
```

This definition measures feature consistency, not pixel sharpness. A uniformly poor tracklet can still appear consistent; this limitation must be stated in documentation and considered when interpreting the experiment.

## Reliable Hard Positive Score

For two different tracklets `i` and `j` with the same identity:

```text
d_pos(i, j) = 1 - cos(c_i, c_j)
pair_reliability(i, j) = sqrt(r_i * r_j)
score(i, j) = d_pos(i, j) * pair_reliability(i, j)
```

Candidates must belong to the same relabeled training identity, and `i != j`.

For every anchor tracklet, store candidate tracklet indices sorted by descending score. The JSON must also contain metadata sufficient to reject incompatible files:

- schema version;
- dataset name and root;
- number of training tracklets and identities;
- sequence length;
- checkpoint path;
- feature dimensionality;
- score definition.

The training loader must fail with a clear error when the graph contains missing/out-of-range indices or does not match the current number of training tracklets.

## RHP Chunk Construction

For an identity with at least two distinct tracklets:

1. Randomly choose an anchor from that identity's available tracklets.
2. Select the highest-ranked valid RHP candidate for that anchor.
3. Fill the remaining `K - 2` positions by random sampling from all real tracklets of the identity, with replacement allowed.
4. Shuffle the K positions so anchor and positive do not occupy fixed locations.

For an identity with exactly one real tracklet, return K copies via with-replacement sampling, matching the baseline fallback.

For the initial G2A ablation, use one K-sized chunk per identity per epoch (`PID_SAMPLES_PER_EPOCH: 4`). This matches the existing G2A setup where the training set contains two real tracklets per identity and avoids repeatedly forcing the same hard pair within one epoch.

## Identity Batch Selection

Two sampler modes are required.

### RHP Only

```yaml
SAMPLER: 'reliable_hard_positive'
```

Identity chunks are shuffled randomly, matching the baseline identity-level behavior as closely as possible. Only within-identity tracklet composition changes.

### RHP Plus DFGS-TV

```yaml
SAMPLER: 'dfgs_tv_reliable_hard_positive'
```

DFGS-TV selects the P identities placed in the batch. RHP selects the K tracklets for each selected identity. These responsibilities must remain independent:

```text
DFGS-TV: inter-identity hard negatives
RHP: intra-identity reliable hard positives
```

The first experiment retains `DFGS_PROBABILITY: 0.25` and the existing text-visual graph so that the only new factor is RHP composition.

## Configuration Fields

Add:

```yaml
DATALOADER:
  RHP_GRAPH: ''
```

`RHP_GRAPH` is mandatory only for the two RHP sampler modes. No RHP-specific setting should affect `softmax_triplet`.

Create:

```text
configs/adapter/innovation/g2a_RHP_p2k4_seed1234_lr2e4.yml
configs/adapter/innovation/g2a_DFGS_TV_RHP_q025_p2k4_seed1234_lr2e4.yml
```

Both configurations use:

- seed 1234;
- P2K4 (`IMS_PER_BATCH: 8`, `NUM_INSTANCE: 4`);
- Stage-2 learning rate `2e-4`;
- PBP disabled;
- the baseline Stage-1 checkpoint;
- outputs below `output/innovation/`;
- the corrected dataset root `../datas/G2A-VReID`.

## Testing

Unit tests must cover:

1. `softmax_triplet` still returns `RandomIdentitySampler_Video`.
2. Cross-camera sampler names are rejected as unknown.
3. RHP selection does not inspect `camid`; changing camera values leaves selection unchanged under the same seed.
4. A difficult and reliable positive outranks an easy reliable positive.
5. A very difficult but unreliable positive can be suppressed by the reliability term.
6. A single-tracklet identity produces K entries by replacement.
7. Every RHP chunk has exactly K entries and one identity.
8. RHP-only identity selection remains random.
9. RHP plus DFGS-TV uses graph-based identity selection while retaining RHP within each identity.
10. Invalid or incompatible RHP graph files fail with actionable messages.
11. Fixed Python and NumPy seeds reproduce the same sampled index sequence.

The offline builder receives a lightweight test using synthetic frame features so that score computation can be verified without loading a GPU model.

## Experiment Interpretation

The primary comparison is:

```text
B0: original random P2K4
P1: random identities + RHP K4
P2: DFGS-TV(q=0.25) + RHP K4
```

P1 tests whether reliable hard positives help independently. P2 tests whether intra-identity hard positives and inter-identity hard negatives are complementary. DFGS-TV remains an attributed existing component, not a claimed original contribution.

RHP is platform-label-free. It may preferentially select aerial-ground pairs when they are the most different reliable same-identity tracklets, but it does not guarantee a cross-platform pair. Claims must therefore use terms such as "view-agnostic" or "platform-label-free," not "guaranteed cross-platform sampling."

## Non-Goals

- No online feature refresh during Stage-2 training.
- No explicit aerial/ground/platform mapping.
- No new loss function.
- No deletion of historical experiment outputs.
- No claim that DFGS-T, DFGS-V, or DFGS-TV is newly proposed by this work.
