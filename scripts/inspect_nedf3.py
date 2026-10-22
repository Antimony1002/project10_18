#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from towel_act.config import load_project_config  # noqa: E402
from towel_act.nedf3 import discover_episodes, inspect_episode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate NEDF3 episode exports")
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "towel_folding.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config = load_project_config(args.config)
    episodes = discover_episodes(args.raw_root)
    reports = [inspect_episode(path, config) for path in episodes]
    payload = {
        "raw_root": str(args.raw_root.resolve()),
        "episodes": [report.to_dict() for report in reports],
        "valid": sum(report.valid for report in reports),
        "invalid": sum(not report.valid for report in reports),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    return 0 if reports and all(report.valid for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
