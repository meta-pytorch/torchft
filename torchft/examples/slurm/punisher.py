# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import random
import time

from torchx import specs
from torchx.runner import get_runner, Runner

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

_SCHEDULER = "slurm"


def kill_all(runner: Runner) -> None:
    jobs = runner.list(_SCHEDULER)
    jobs = [job for job in jobs if job.state == specs.AppState.RUNNING]
    for job in jobs:
        if "ft_" not in job.name:
            continue
        print(f"killing {job.app_handle}")
        runner.cancel(job.app_handle)


def kill_one(runner: Runner) -> None:
    jobs = runner.list(_SCHEDULER)
    jobs = [job for job in jobs if job.state == specs.AppState.RUNNING]
    candidates = []
    for job in jobs:
        if "ft_" not in job.name:
            continue
        if "ft_0" in job.name:
            continue
        candidates.append(job.app_handle)
    choice = random.choice(candidates)
    print(f"killing {choice=} {candidates=}")
    runner.cancel(choice)


def kill_loop(runner: Runner, args: argparse.Namespace) -> None:
    for _ in range(args.num_failures):
        kill_one(runner)
        dur = random.random() * (2 * args.mtbf_secs)
        print(f"sleeping for {dur=} {args.mtbf_secs=}")
        time.sleep(args.mtbf_secs)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to inject failures on slurm")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # kill_loop subcommand
    kill_loop_parser = subparsers.add_parser("kill_loop", help="Kill jobs in a loop")
    kill_loop_parser.add_argument(
        "--mtbf-secs",
        type=float,
        default=5,
        help="Mean time between failures",
    )
    kill_loop_parser.add_argument(
        "--num-failures",
        type=int,
        default=1,
        help="Number of failures to inject",
    )

    # kill_one subcommand
    subparsers.add_parser("kill_one", help="Kill a single job")

    # kill_all subcommand
    subparsers.add_parser("kill_all", help="Kill all jobs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    with get_runner() as runner:
        if args.command == "kill_loop":
            kill_loop(runner, args)
        elif args.command == "kill_one":
            kill_one(runner)
        elif args.command == "kill_all":
            kill_all(runner)


if __name__ == "__main__":
    main()
