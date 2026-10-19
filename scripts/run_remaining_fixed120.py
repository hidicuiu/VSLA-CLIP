"""Sequentially finish the six fixed-120 runs without overwriting outputs."""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = r"D:\anaconda\envs\ReID\python.exe"
RUNS = [
    ("baseline2025", "reproduce/g2a_vifi_no_pbp_stage2_seed2025_fixed120_lr2e4.yml", "reproduce/vifi_no_pbp_stage2_seed2025_fixed120_lr2e4"),
    ("D1_2025", "innovation/g2a_D1_dynamic_DFGS_V_q025_p2k4_seed2025_fixed120_lr2e4.yml", "innovation/g2a_D1_dynamic_DFGS_V_q025_p2k4_seed2025_fixed120_lr2e4"),
    ("D4_2025", "innovation/g2a_D4_stratified_dynamic_DFGS_V_q025_p2k4_seed2025_fixed120_lr2e4.yml", "innovation/g2a_D4_stratified_dynamic_DFGS_V_q025_p2k4_seed2025_fixed120_lr2e4"),
    ("baseline2026", "reproduce/g2a_vifi_no_pbp_stage2_seed2026_fixed120_lr2e4.yml", "reproduce/vifi_no_pbp_stage2_seed2026_fixed120_lr2e4"),
    ("D1_2026", "innovation/g2a_D1_dynamic_DFGS_V_q025_p2k4_seed2026_fixed120_lr2e4.yml", "innovation/g2a_D1_dynamic_DFGS_V_q025_p2k4_seed2026_fixed120_lr2e4"),
    ("D4_2026", "innovation/g2a_D4_stratified_dynamic_DFGS_V_q025_p2k4_seed2026_fixed120_lr2e4.yml", "innovation/g2a_D4_stratified_dynamic_DFGS_V_q025_p2k4_seed2026_fixed120_lr2e4"),
]


def complete(directory):
    log = directory / "train_log.txt"
    return (directory / "ViT-B-16_120.pth").is_file() and log.is_file() and "Total running time:" in log.read_text(encoding="utf-8", errors="replace")


def run(args, log):
    print("Starting", " ".join(args), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Exit {}: {}".format(result.returncode, log))


def main():
    first = ROOT / "output" / RUNS[0][2]
    deadline = time.monotonic() + 8 * 3600
    print("Waiting for baseline2025 to finish", flush=True)
    while not complete(first):
        if time.monotonic() > deadline:
            raise TimeoutError("baseline2025 has not finished within eight hours")
        time.sleep(60)
    logs = ROOT / "output" / "evaluation" / "fixed120"
    for index, (name, config_suffix, output_suffix) in enumerate(RUNS):
        config = "configs/adapter/" + config_suffix
        output = ROOT / "output" / output_suffix
        if index:
            if output.exists():
                raise FileExistsError("Refusing to overwrite " + str(output))
            run([PYTHON, "-B", "train_reidadapter.py", "--config_file", config], logs / (name + "_train_runner.log"))
            if not complete(output):
                raise RuntimeError("Missing completed checkpoint: " + str(output))
        if (output / "test_log.txt").exists():
            raise FileExistsError("Refusing to overwrite test log in " + str(output))
        run([PYTHON, "-B", "test.py", "--config_file", config], logs / (name + "_test_runner.log"))
        if "mAP:" not in (output / "test_log.txt").read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError("Missing test result: " + str(output))
        print("Completed", name, flush=True)
    print("All six runs completed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("QUEUE STOPPED:", type(exc).__name__, exc, flush=True)
        sys.exit(1)
