#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from towel_act.config import load_project_config  # noqa: E402
from towel_act.convert import convert_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert exported NEDF3 episodes to LeRobot Dataset v3")
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "towel_folding.json")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--episode-list", type=Path, help="Text file with one episode directory name per line")
    args = parser.parse_args()

    names = None
    if args.episode_list:
        names = {
            line.strip()
            for line in args.episode_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    report = convert_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        config=load_project_config(args.config),
        skip_invalid=args.skip_invalid,
        episode_names=names,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
