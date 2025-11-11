# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time

from torchx import specs
from torchx.components.dist import ddp
from torchx.runner import get_runner, Runner

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

_SCHEDULER = "slurm"


def _make_app(replica_id: int, cli_args: argparse.Namespace) -> specs.AppDef:
    args = [
        "--comm.trace_buf_size=0",
        "--comm.train_timeout_seconds=60",
        "--metrics.log_freq=1",
        "--profiling.enable_profiling",
        "--experimental.custom_args_module=torchtitan.components.ft.config",
        "--job.config_file=./torchtitan/models/llama3/train_configs/llama3_8b.toml",
        "--model.name=llama3_ft",
        "--training.dataset=c4",
        "--training.steps=10000",
        "--training.local_batch_size=2",
        f"--parallelism.data_parallel_shard_degree={cli_args.nodes * cli_args.nproc_per_node}",
        "--fault_tolerance.enable",
        f"--fault_tolerance.replica_id={replica_id}",
        f"--fault_tolerance.group_size={cli_args.replica_count}",
        f"--fault_tolerance.process_group={cli_args.process_group}",
        f"--fault_tolerance.process_group_timeout_ms={600 * 1000}",
    ]

    if cli_args.enable_semi_sync:
        args += [
            f"--fault_tolerance.semi_sync_method={cli_args.semi_sync_method}",
        ]

    if cli_args.semi_sync_method == "diloco":
        args += [
            "--fault_tolerance.sync_steps=20",
            "--fault_tolerance.fragment_sync_delay=1",
            f"--fault_tolerance.num_fragments={cli_args.num_fragments}",
        ]

    if replica_id == 0:
        args += [
            "--metrics.enable-wandb",
            "--checkpoint.interval=100",
        ]

    env = {}

    # use agent store in torchelastic to avoid TCPStore init race condition
    env["TORCH_SHARE_RDZV_TCP_STORE"] = "1"
    env["TORCH_CPP_LOG_LEVEL"] = "INFO"

    env["TORCH_CUDA_SANITIZER=1"] = "1"

    # NCCL envs for debugging
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = "ALL"
    env["NCCL_PROTO"] = "Simple"

    # gloo
    if os.environ.get("GLOO_SOCKET_IFNAME") is not None:
        env["GLOO_SOCKET_IFNAME"] = os.environ.get("GLOO_SOCKET_IFNAME")

    # application log levels
    env["LOGLEVEL"] = "INFO"
    env["RUST_LOGS"] = "INFO"
    env["TORCH_CPP_LOG_LEVEL"] = "INFO"

    # application timeouts
    env["TORCHFT_QUORUM_TIMEOUT_SEC"] = "900"
    env["TORCHFT_TIMEOUT_SEC"] = "600"
    env["TORCHFT_QUORUM_RETRIES"] = "0"

    env["TORCHFT_LIGHTHOUSE"] = os.environ.get(
        "TORCHFT_LIGHTHOUSE", "http://slurm-head-node-0:29510"
    )

    env["WANDB_PROJECT"] = "torchft"

    app = ddp(
        *args,
        name=f"ft_{replica_id}",
        env=env,
        script="./torchtitan/train.py",
        gpu=cli_args.nproc_per_node,
        j=f"{cli_args.nodes}x{cli_args.nproc_per_node}",
    )
    app.roles[0].name = app.name
    return app


def start_replica(
    runner: Runner, replica_id: int, args: argparse.Namespace
) -> specs.AppHandle:
    app = _make_app(replica_id, args)

    app_handle = runner.run(
        app,
        scheduler=_SCHEDULER,
    )

    return app_handle


def monitor(runner: Runner, args: argparse.Namespace) -> None:
    jobs = runner.list(_SCHEDULER)
    jobs = [job for job in jobs if job.state == specs.AppState.RUNNING]

    active_replicas = {}

    for job in jobs:
        if "ft_" not in job.name:
            continue
        name, _, _ = job.name.partition("-")
        _, _, replica_id_str = name.partition("_")
        replica_id = int(replica_id_str)
        active_replicas[replica_id] = job

    to_launch = set()
    for replica_id in range(args.replica_count):
        alive = replica_id in active_replicas

        if alive:
            job = active_replicas[replica_id]
            print(f" - {replica_id=:2d}: ALIVE {job.app_handle}")
        else:
            print(f" - {replica_id=:2d}: DEAD")
            to_launch.add(replica_id)

    for replica_id in to_launch:
        app_handle = start_replica(
            runner,
            replica_id,
            args,
        )
        print(f"launched {replica_id=}: {app_handle=}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI tool lauch data parallel replicas on slurm"
    )

    parser.add_argument(
        "--workspace-dir", type=str, help="Location of torchtitan folder"
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=10,
        help="Number of nodes per replica",
    )

    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=10,
        help="Number of ranks per node",
    )

    parser.add_argument(
        "--replica-count",
        type=int,
        default=10,
        help="Number of data parallel replicas",
    )

    parser.add_argument(
        "--process-group",
        type=str,
        default="gloo",
        help="The process group to use for data parallel",
    )

    parser.add_argument(
        "--enable-semi-sync",
        type=bool,
        default=True,
        help="Whether to enable semi-sync method for data parallel",
    )

    parser.add_argument(
        "--semi-sync-method",
        type=str,
        default="diloco",
        help="The semi-sync method to use for data parallel. Options: diloco, local_sgd",
    )

    parser.add_argument(
        "--num-fragments",
        type=int,
        default=2,
        help="The number of fragments to use for data parallel. Only used for diloco semi-sync method",
    )

    args = parser.parse_args()

    os.chdir(args.workspace_dir)

    with get_runner() as runner:
        while True:
            monitor(runner, args)
            time.sleep(10)


if __name__ == "__main__":
    main()
