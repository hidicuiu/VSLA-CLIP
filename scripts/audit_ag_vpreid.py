"""Read-only AG-VPReID metadata audit; does not train or open frame images."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets.data_manager import AGVPReID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--direction", choices=("g2a", "a2g"), default="g2a")
    parser.add_argument("--split-dir", default="")
    parser.add_argument("--output", help="Optional new JSON file; existing files are not overwritten")
    args = parser.parse_args()
    dataset = AGVPReID(root=args.root, direction=args.direction, split_dir=args.split_dir)
    report = json.dumps(dataset.audit, indent=2)
    print(report)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(report + "\n")


if __name__ == "__main__":
    main()
