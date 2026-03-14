#!/usr/bin/env python3
"""Convert rerun.io recordings to a LeRobot dataset.

Each .rrd file in the rerun directory becomes one episode. Empty recordings are skipped.
LeRobot episode IDs are sequential (0, 1, 2, ...) for converted episodes only.
"""

import argparse
from pathlib import Path

from slobot.lerobot.rerun_to_lerobot_converter import RerunToLeRobotConverter

import logging
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Convert rerun.io recordings to a LeRobot dataset."
    )
    parser.add_argument(
        "--rerun-dir",
        type=str,
        required=True,
        help="Directory containing .rrd recording files (e.g. /tmp/slobot/recordings/)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Target LeRobot dataset ID (e.g. alexis779/so100_ball_cup_sim)",
    )
    parser.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        help="Comma-separated list of episode IDs to process (e.g. 0,1,2). Episode ID = index in sorted .rrd list. Default: all.",
    )
    args = parser.parse_args()

    episode_ids = None
    if args.episode_ids is not None:
        episode_ids = [int(x.strip()) for x in args.episode_ids.split(",")]

    converter = RerunToLeRobotConverter(
        rerun_dir=Path(args.rerun_dir),
        dataset_id=args.dataset_id,
        episode_ids=episode_ids,
    )
    converter.convert()


if __name__ == "__main__":
    main()
