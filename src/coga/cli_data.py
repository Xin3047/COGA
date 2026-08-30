"""Command-line interface for the RST-to-COGA data preparation pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .cohorts import build as build_cohorts
from .common import read_json
from .rst import download, extract, verify


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="coga-data")
    parser.add_argument("command", choices=("download", "verify", "extract", "cohorts", "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/qwen3_8b_4090.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = read_json(args.config.resolve())
    commands = ("download", "verify", "extract", "cohorts") if args.command == "all" else (args.command,)
    handlers = {
        "download": download,
        "verify": verify,
        "extract": extract,
        "cohorts": build_cohorts,
    }
    for command in commands:
        print(json.dumps(handlers[command](config), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
