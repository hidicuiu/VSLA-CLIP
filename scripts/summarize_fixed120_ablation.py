"""Read-only summary for the fixed-epoch G2A baseline/D1/D4 comparison.

Only standalone test_log.txt results using ViT-B-16_120.pth are accepted.
Missing runs remain pending; this script never starts training or writes files.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


METRIC_RE = re.compile(r"mAP:\s*([0-9.]+)%\s*\|\s*R-1\s*:\s*([0-9.]+)%")
WEIGHT_RE = re.compile(r"^\s*WEIGHT:\s*['\"]?([^\s'\"]+\.pth)", re.MULTILINE)
METHODS = ("baseline", "D1", "D4")
SEEDS = (1234, 2025, 2026)


def result_dir(root, method, seed):
    if seed == 1234:
        return root / "output" / "evaluation" / "fixed120" / (
            "seed1234_" + method)
    if method == "baseline":
        name = "vifi_no_pbp_stage2_seed{}_fixed120_lr2e4".format(seed)
        return root / "output" / "reproduce" / name
    variant = ("dynamic" if method == "D1" else "stratified_dynamic")
    name = "g2a_{}_{}_DFGS_V_q025_p2k4_seed{}_fixed120_lr2e4".format(
        method, variant, seed)
    return root / "output" / "innovation" / name


def parse_test_content(content):
    weights = WEIGHT_RE.findall(content)
    if not weights or not weights[-1].replace("\\", "/").endswith(
            "/ViT-B-16_120.pth"):
        raise ValueError("Test did not use ViT-B-16_120.pth")
    metrics = METRIC_RE.findall(content)
    if not metrics:
        raise ValueError("No mAP / R-1 line found")
    return tuple(float(value) for value in metrics[-1])


def parse_test_log(path):
    if not path.is_file():
        return None
    try:
        return parse_test_content(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError as error:
        raise ValueError("{}: {}".format(error, path)) from error


def d4_graph_diagnostics(root, seed):
    graph_dir = result_dir(root, "D4", seed) / "dynamic_dfgs_graphs"
    for graph_path in sorted(graph_dir.glob("epoch_*.json")):
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        diagnostic = payload.get("filter_diagnostics", {})
        yield (
            payload.get("metadata", {}).get("epoch"),
            diagnostic.get("excluded_identity_count"),
            diagnostic.get("single_camera_low_count"),
            diagnostic.get("both_camera_low_count"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="VSLA-CLIP-master project root")
    parser.add_argument("--show-d4-exclusions", action="store_true",
                        help="also list every D4 graph update's exclusion counts")
    args = parser.parse_args()
    root = args.root.resolve()
    results = {}
    print("Only fixed epoch-120 standalone test results are included.\n")
    print("seed   method     mAP     R-1    test log")
    for seed in SEEDS:
        for method in METHODS:
            log_path = result_dir(root, method, seed) / "test_log.txt"
            try:
                metric = parse_test_log(log_path)
            except ValueError as error:
                print("{}  {:8}  INVALID: {}".format(seed, method, error))
                continue
            if metric is None:
                print("{}  {:8}  pending  {}".format(seed, method, log_path))
                continue
            results[(seed, method)] = metric
            print("{}  {:8}  {:5.2f}%  {:5.2f}%  {}".format(
                seed, method, metric[0], metric[1], log_path))

    print("\nPaired mAP differences (percentage points):")
    print("seed   D1-baseline   D4-D1   D4-baseline")
    paired = {"D1-baseline": [], "D4-D1": [], "D4-baseline": []}
    for seed in SEEDS:
        if all((seed, method) in results for method in METHODS):
            baseline = results[(seed, "baseline")][0]
            d1 = results[(seed, "D1")][0]
            d4 = results[(seed, "D4")][0]
            differences = (d1 - baseline, d4 - d1, d4 - baseline)
            for key, value in zip(paired, differences):
                paired[key].append(value)
            print("{}  {:+11.2f}  {:+7.2f}  {:+11.2f}".format(
                seed, *differences))
        else:
            print("{}  pending".format(seed))
    if len(paired["D1-baseline"]) >= 2:
        print("mean  {:+11.2f}  {:+7.2f}  {:+11.2f}".format(
            *(statistics.mean(values) for values in paired.values())))
        print("sd    {:11.2f}  {:7.2f}  {:11.2f}".format(
            *(statistics.stdev(values) for values in paired.values())))

    if args.show_d4_exclusions:
        print("\nD4 graph updates: seed epoch excluded one-camera-low both-camera-low")
        for seed in SEEDS:
            for epoch, excluded, single_low, both_low in d4_graph_diagnostics(root, seed):
                print("{} {} {} {} {}".format(
                    seed, epoch, excluded, single_low, both_low))


if __name__ == "__main__":
    main()
