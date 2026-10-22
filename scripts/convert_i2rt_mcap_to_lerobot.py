#!/usr/bin/env python3
"""Batch-convert a reviewed A/B/C/D source manifest to a local ACT dataset."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from towel_act.i2rt_mcap.convert import (
    Options,
    action_directions,
    align_episode,
    convert,
    discover_sources,
    read_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--teleop-config",
        type=Path,
        required=True,
        help="reviewed teleop configuration used during recording",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id", default="local/i2rt_act")
    parser.add_argument(
        "--arms", nargs="+", choices=("left", "right"), default=["left", "right"]
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=("left", "right", "head"),
        help="default: wrist cameras for the selected arms; synthetic head is rejected",
    )
    parser.add_argument("--collectors", nargs="+", choices=("A", "B", "C", "D"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gap-policy", choices=("reject", "split"), default="reject")
    parser.add_argument("--max-gap-ms", type=float, default=200)
    parser.add_argument("--min-seconds", type=float, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect alignment without creating a dataset",
    )
    args = parser.parse_args()
    try:
        options = Options(
            fps=args.fps,
            arms=tuple(args.arms),
            cameras=tuple(args.cameras or args.arms),
            gap_policy=args.gap_policy,
            max_gap_ms=args.max_gap_ms,
            min_seconds=args.min_seconds,
        )
        sources, excluded = discover_sources(
            args.manifest, set(args.collectors) if args.collectors else None
        )
        if args.dry_run:
            signs, _ = action_directions(args.teleop_config, options.arms)
            print(f"Excluded {len(excluded)} episodes; checking {len(sources)}")
            for source in sources:
                with tempfile.TemporaryDirectory(prefix="i2rt-preflight-") as tmp:
                    episode = read_episode(source, Path(tmp), options)
                    segments, gaps = align_episode(episode, options, signs)
                    print(
                        json.dumps(
                            {
                                "source": str(source.path),
                                "collector": source.collector,
                                "segment_frames": [len(s.times) for s in segments],
                                "stream_gaps": len(gaps),
                            },
                            ensure_ascii=False,
                        )
                    )
        else:
            if args.output is None:
                parser.error("--output is required unless --dry-run is used")
            report = convert(
                sources,
                args.output,
                args.repo_id,
                options,
                args.teleop_config,
                excluded,
            )
            print(
                f"Saved {len(report['episodes'])} episodes / {report['total_frames']} frames to {args.output}"
            )
        return 0
    except (ValueError, OSError) as exc:
        print(f"Conversion stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
