"""Read-only configuration, budget, and fixed-epoch result audit for G2A Phase 1.

This script never launches training, loads model weights, or writes outputs.
"""

import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import cfg
from scripts.summarize_fixed120_ablation import parse_test_log, WEIGHT_RE


METHODS = ('baseline', 'E1', 'E4')
SEEDS = (1234, 2025, 2026)
EXISTING_OUTPUTS = {
    'baseline': 'output/reproduce/g2a_vifi_pbp_p4k4_seed1234_lr2e4',
    'E1': 'output/innovation/g2a_E1_pbp_dynamic_DFGS_V_q025_p4k4_seed1234_lr2e4',
    'E4': 'output/innovation/g2a_E4_pbp_multicamera_dynamic_DFGS_V_q025_p4k4_seed1234_lr2e4',
}


def load_configurations(root):
    configurations = {}
    outputs = set()
    reference = None
    reference_protocol = None
    for seed in SEEDS:
        for method in METHODS:
            path = root / 'configs/adapter/phase1' / (
                'g2a_{}_pbp_p4k4_seed{}.yml'.format(method, seed))
            current = cfg.clone()
            current.merge_from_file(str(path))
            current.freeze()
            assert current.SOLVER.SEED == seed, path
            assert current.DATASETS.NAMES == 'g2a', path
            assert current.MODEL.PBP_CAMERA and current.MODEL.USE_IFA and current.MODEL.USE_CFAA, path
            assert not current.MODEL.STAGE1_STRICT_LOAD, path
            assert not current.SOLVER.STAGE2.RESET_SEED, path
            assert not current.DATALOADER.DYNAMIC_DFGS_PRESERVE_RNG_STATE, path
            assert current.SOLVER.STAGE2.PROTECT_OUTPUT, path
            assert current.SOLVER.STAGE2.EVALUATION_ONLY == (seed == 1234), path
            assert current.SOLVER.STAGE2.IMS_PER_BATCH == 16, path
            assert current.DATALOADER.NUM_INSTANCE == 4, path
            assert current.DATALOADER.SEQ_LEN == 8, path
            assert current.SOLVER.STAGE2.MAX_EPOCHS == 120, path
            assert current.SOLVER.STAGE2.CHECKPOINT_PERIOD == 120, path
            assert current.SOLVER.STAGE2.EVAL_PERIOD == 10, path
            assert current.SOLVER.STAGE2.BASE_LR == 0.0002, path
            assert current.DATALOADER.DYNAMIC_DFGS_ENABLED == (method != 'baseline'), path
            assert current.DATALOADER.DYNAMIC_DFGS_RELIABLE_NEGATIVES == (method == 'E4'), path
            assert current.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA == (method == 'E4'), path
            assert not current.DATALOADER.DYNAMIC_DFGS_QUALITY_WEIGHTED, path
            assert not current.DATALOADER.DYNAMIC_DFGS_STRATIFIED_TWO_CAMERA, path
            assert not current.DATALOADER.DYNAMIC_DFGS_STRATIFIED_PLATFORM, path
            assert not current.DATALOADER.DYNAMIC_DFGS_SOFT_NEGATIVES, path
            # Baseline keeps the original default; it is unused with DFGS off.
            expected_probability = 1.0 if method == 'baseline' else 0.25
            assert current.DATALOADER.DFGS_PROBABILITY == expected_probability, path
            assert current.DATALOADER.SAMPLER == (
                'softmax_triplet' if method == 'baseline' else 'mixed_dfgs_baseline_k4'), path
            assert current.DATALOADER.PID_SAMPLES_PER_EPOCH == (0 if method == 'baseline' else 4), path
            weight_output = EXISTING_OUTPUTS[method] if seed == 1234 else current.OUTPUT_DIR
            assert current.TEST.WEIGHT == weight_output + '/ViT-B-16_120.pth', path
            if seed == 1234:
                assert (root / current.TEST.WEIGHT).is_file(), current.TEST.WEIGHT
                assert current.OUTPUT_DIR == weight_output + '/phase1_epoch120_eval', path
            output = (root / current.OUTPUT_DIR).resolve()
            assert root.resolve() in output.parents and output not in outputs, path
            outputs.add(output)
            normalized = current.clone()
            normalized.defrost()
            normalized.MODEL.STAGE1_WEIGHT = ''
            normalized.SOLVER.STAGE2.EVALUATION_ONLY = False
            # Everything affecting the backbone and losses must match all arms.
            backbone_signature = normalized.MODEL.dump()
            if reference is None:
                reference = backbone_signature
            assert backbone_signature == reference, path
            protocol = (current.INPUT.dump(), current.SOLVER.STAGE1.dump(),
                        normalized.SOLVER.STAGE2.dump(), current.DATASETS.dump())
            if reference_protocol is None:
                reference_protocol = protocol
            assert protocol == reference_protocol, path
            configurations[(seed, method)] = current

    for seed in SEEDS:
        baseline = configurations[(seed, 'baseline')]
        stage1 = (EXISTING_OUTPUTS['baseline'] + '/ViT-B-16_stage1_120.pth' if seed == 1234 else
                  baseline.MODEL.STAGE1_WEIGHT or
                  baseline.OUTPUT_DIR + '/ViT-B-16_stage1_120.pth')
        for method in ('E1', 'E4'):
            assert configurations[(seed, method)].MODEL.STAGE1_WEIGHT == stage1
        if seed == 1234:
            assert (root / stage1).is_file(), 'Missing seed1234 Stage-1 checkpoint'
        else:
            assert not baseline.MODEL.STAGE1_WEIGHT
    return configurations


def print_results(root, configurations):
    results = {}
    print('\nOnly standalone test_log.txt results from the configured epoch-120 weight:')
    for key, current in configurations.items():
        seed, method = key
        log = root / current.OUTPUT_DIR / 'test_log.txt'
        if not log.is_file():
            print('{} {:8} pending'.format(seed, method))
            continue
        try:
            metric = parse_test_log(log)
            recorded = WEIGHT_RE.findall(log.read_text(encoding='utf-8', errors='replace'))
            if (root / recorded[-1]).resolve() != (root / current.TEST.WEIGHT).resolve():
                raise ValueError('Log checkpoint path differs from this configuration')
        except (ValueError, IndexError) as error:
            print('{} {:8} INVALID: {}'.format(seed, method, error))
            continue
        results[key] = metric
        print('{} {:8} mAP={:.2f}% R-1={:.2f}%'.format(seed, method, *metric))

    for metric_index, metric_name in ((0, 'mAP'), (1, 'Rank-1')):
        print('\nPaired {} changes (percentage points):'.format(metric_name))
        for left, right in (('E1', 'baseline'), ('E4', 'E1'), ('E4', 'baseline')):
            values = []
            for seed in SEEDS:
                if (seed, left) in results and (seed, right) in results:
                    value = results[(seed, left)][metric_index] - results[(seed, right)][metric_index]
                    values.append(value)
                    print('{} {}-{} {:+.2f}'.format(seed, left, right, value))
            if len(values) >= 2:
                print('{}-{} mean={:+.2f}, sample SD={:.2f}, n={}'.format(
                    left, right, statistics.mean(values), statistics.stdev(values), len(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check-data', action='store_true', help='also read dataset metadata and check sampler budgets')
    args = parser.parse_args()
    root = args.root.resolve()
    configurations = load_configurations(root)
    print('PASS: seed1234 reuses THREE COMPLETED RUNS (evaluation only); SIX new training configurations for seeds2025/2026.')
    print('Original RNG handling retained: Stage-2 reset OFF, graph RNG preservation OFF.')
    if args.check_data:
        from datasets import data_manager
        from datasets.make_video_dataloader import _make_stage2_sampler
        first = configurations[(1234, 'baseline')]
        dataset = data_manager.init_dataset(
            name=first.DATASETS.NAMES, root=str(root / first.DATASETS.ROOT_DIR))
        counts = Counter(pid for _, pid, _ in dataset.train)
        # Natural-epoch baseline and fixed-K graph arms have matched budgets
        # for this G2A release because every identity has exactly two tracks.
        if set(counts.values()) != {2}:
            raise ValueError('G2A tracklet distribution changed; recheck natural vs fixed-K budgets')
        budgets = {len(_make_stage2_sampler(current, dataset.train)) // 16
                   for current in configurations.values()}
        if len(budgets) != 1:
            raise ValueError('Unmatched Stage-2 batch counts: {}'.format(budgets))
        print('PASS: {} IDs, {} tracks; {} batches/epoch in all arms.'.format(
            len(counts), len(dataset.train), next(iter(budgets))))
    for (seed, method), current in configurations.items():
        output = root / current.OUTPUT_DIR
        status = ('REUSE CHECKPOINT, TEST ONLY' if seed == 1234 else
                  'existing (training refuses overwrite)' if output.is_dir() and any(output.iterdir()) else 'new/empty')
        print('{} {:8} {}: {}'.format(seed, method, status, current.OUTPUT_DIR))
    print_results(root, configurations)


if __name__ == '__main__':
    main()
