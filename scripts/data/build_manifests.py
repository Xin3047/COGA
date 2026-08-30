#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from coga.common import atomic_json, read_json, resolve_project_path, write_jsonl
from coga.data import build_pair_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real COGA pair/control manifest.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    output_dir = resolve_project_path(config["paths"]["run_dir"])
    rows = build_pair_specs(config)
    stats = write_jsonl(output_dir / "manifests" / "gradient_pairs.jsonl", (row.to_json() for row in rows))
    counts = Counter(row.control for row in rows)
    report = {
        "status": "REAL_PAIR_AND_CONTROL_MANIFESTS_BUILT",
        "manifest": stats,
        "rows_by_control": dict(sorted(counts.items())),
        "scientific_gradient_results": None,
    }
    atomic_json(output_dir / "manifests" / "manifest_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
