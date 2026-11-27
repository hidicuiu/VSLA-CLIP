# Cross-platform two-head set contrast

The direct baseline is `g2a_E4_m3_frame_dual_mean_graph.yml`.
The new `g2a_E4_m3_frame_dual_mean_graph_set.yml` changes only the training
auxiliary loss and output/checkpoint destinations. Sampling, mean graph readout,
fusion, retrieval, original losses and training schedule remain the same.

Run from the repository root in the existing ReID environment:

```powershell
python train_reidadapter.py --config_file configs/adapter/innovation/g2a_E4_m3_frame_dual_mean_graph_set.yml
```

An AG-VPReID counterpart is supplied as
`agvpreid_E4_m3_frame_dual_mean_graph_set.yml`; it retains the source config's
dataset paths and bidirectional evaluation. Check dataset/checkpoint paths on
the training machine before starting. Both configs load the same original
Stage-1 weights as their direct baselines; do not use a trained Stage-2 baseline
checkpoint for a from-scratch matched comparison.

For each video, concatenate the pre-fusion final CLS and projection descriptors
within each head, then L2-normalize the resulting [B,2,1280] tensor. Set similarity
is the maximum of the average cosine for same-order and crossed one-to-one
assignments. All opposite-platform videos form the denominator; all same-ID
opposite-platform videos are positives. Average negative log probability over
positives, then valid anchors (SupCon outside-log form). Both platform directions
are included. Anchors without a positive or a negative identity are skipped;
an entirely invalid batch returns a differentiable zero.

Defaults: loss weight 0.05, temperature 0.1. These are starting values, not proven
optimal. The global default weight is zero, preserving existing configs.
No new trainable parameters or retrieval dimensions are introduced.

Controls via command-line overrides (use a fresh OUTPUT_DIR for each run):

- `MODEL.CROSS_PLATFORM_SET_WEIGHT 0.0`: disable the auxiliary loss.
- `MODEL.CROSS_PLATFORM_SET_MATCHING fixed`: fixed head correspondence control.
- `MODEL.CROSS_PLATFORM_SET_TEMPERATURE 0.07`: temperature sensitivity example.

## Four-way ablation

For both `g2a` and `agvpreid`, configurations share the prefix
`<dataset>_E4_m3_frame_dual_mean_graph`:

| Group | Config suffix | Auxiliary input / similarity |
| --- | --- | --- |
| A | `.yml` | Original baseline, no auxiliary contrast |
| B | `_supcon_fused.yml` | Concatenated post-fusion final CLS + projection, cosine |
| C | `_set_fixed.yml` | Pre-fusion heads, average same-order cosine |
| D | `_set.yml` | Pre-fusion heads, maximum of two assignment scores |

B/C/D share weight 0.05, temperature 0.1, opposite-platform candidate masks,
multi-positive outside-log reduction and valid-anchor rules. C averages the two
head similarities before applying ONE SupCon loss, rather than summing two losses.
B uses the same pre-BN fused branches as retrieval, not classifier logits or BN
features. It backpropagates through the residual fusion. Matching/head diagnostics
are not applicable to B and are omitted from its contrast log. All configurations
have independent output and test checkpoint paths; the existing A/D files are
unchanged. No additional model parameters or RNG initialization are introduced.

Example commands (run separately; C first is recommended):

```powershell
python train_reidadapter.py --config_file configs/adapter/innovation/g2a_E4_m3_frame_dual_mean_graph_set_fixed.yml
python train_reidadapter.py --config_file configs/adapter/innovation/g2a_E4_m3_frame_dual_mean_graph_supcon_fused.yml
```

Camera-to-platform maps are explicit and independent of graph filtering:
G2A uses its two camera domains [0,1] (binary naming does not affect the loss);
AG-VPReID uses [0,0,0,0,1,1] from dataset metadata. Camera prompt labels remain
camera IDs. The original sampler does not enforce platform coverage per ID.

Epoch diagnostics: unweighted set loss, valid-anchor fraction, positive count
(per-batch valid-anchor mean), positive crossed-assignment fraction, absolute
matching-score gap, and within-video head cosine. Batch summaries are averaged
by batch size; pair diagnostics are not globally pair-weighted. Crossed-assignment
frequency alone is not evidence of complementary heads, particularly near ties.
An epoch with no valid anchors emits a warning.

Verification:

```powershell
python -m unittest discover -s tests -p test_cross_platform_set.py
```

This checks a loop-reference multi-positive loss, permutation invariance,
invalid/mixed batches, scorer gradients, config loading and CUDA mixed precision.
Training performance and cross-platform coverage on real data still require a run.
