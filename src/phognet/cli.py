from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="phognet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Run training")
    p_train.add_argument("--config", default="configs/train.yaml")

    p_ab = sub.add_parser("ablation", help="Run ablation/sweep runner")
    p_ab.add_argument("--args", nargs="*", default=[])

    args = parser.parse_args()

    if args.cmd == "train":
        raise SystemExit(
            subprocess.call([sys.executable, "scripts/train.py", "--config", args.config])
        )
    if args.cmd == "ablation":
        raise SystemExit(subprocess.call([sys.executable, "scripts/run_ablation.py", *args.args]))
