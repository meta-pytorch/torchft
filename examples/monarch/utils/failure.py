# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import ctypes
import logging
import os
import random
from enum import Enum
from typing import Dict, TYPE_CHECKING

import torch
from monarch.actor import Actor, current_rank, endpoint

logger = logging.getLogger()

if TYPE_CHECKING:
    from ..train_distributed_k8s import MonarchKubernetes, Replica


class Failure(Enum):
    """
    A set of supported failures injected by FailureController
    """

    SEGFAULT = 0
    KILL_PROC = 1
    COMMS = 2
    KILL_JOB = 3
    DEADLOCK = 4


class FailureActor(Actor):
    @endpoint
    def fail(self, failure: Failure) -> None:
        rank = current_rank().rank
        logger.info(f"[FailureController] {rank} chosen to fail with {failure}")
        match failure:
            case Failure.SEGFAULT:
                self.segfault()
            case Failure.KILL_PROC:
                self.kill_proc()
            case Failure.COMMS:
                self.kill_comms()
            case Failure.DEADLOCK:
                self.deadlock()

    def segfault(self):
        """
        Triggers a SIGSEGV on the process
        """
        crash_func = ctypes.CFUNCTYPE(None)()
        crash_func()

    def kill_proc(self):
        """
        Immediately kills the process with non-zero exit code
        """
        os._exit(1)

    def kill_comms(self):
        """
        Forcefully aborts the ProcessGroup and NCCL communicator
        """
        torch.distributed.distributed_c10d._abort_process_group()

    def deadlock(self):
        """
        Deadlocks the GIL, resulting in ProcessGroupNCCL timeout and terminal failure
        """
        libc = ctypes.PyDLL(None)
        libc.sleep.argtypes = tuple([ctypes.c_uint])
        libc.sleep.restype = ctypes.c_uint
        libc.sleep(70)


class FailureController:
    @staticmethod
    def kill_job(scheduler: "MonarchKubernetes") -> None:
        """
        Kills a random replica K8s job
        """
        candidates = [
            mesh_name
            for mesh_name in scheduler.job_handles.keys()
            if "replica" in mesh_name
        ]
        selected = random.choice(candidates)
        logger.info(f"[FailureController] Killing K8s job for {selected}")
        scheduler.kill_job(selected)

    @staticmethod
    async def execute_failures(
        replicas: Dict[int, "Replica"],
        scheduler: "MonarchKubernetes",
        startup_wait: int = 120,
        rest_time: int = 120,
    ):
        logger.info(
            f"[FailureController] Starting failure injection in {startup_wait} seconds"
        )
        await asyncio.sleep(startup_wait)  # allow startups.

        last_replica = last_failure = None
        while replicas:
            try:
                running_replicas = list(replicas.values())
                # allow deadlocked replicas more time to recover
                if last_failure == Failure.DEADLOCK and last_replica:
                    running_replicas = [
                        r for r in running_replicas if r.rid != last_replica.rid
                    ]

                last_replica = random.choice(running_replicas)
                # Exclude KILL_JOB on K8s: deleting the CRD doesn't kill
                # already-connected actors, so it's a no-op.
                process_failures = [f for f in Failure if f != Failure.KILL_JOB]
                last_failure = random.choice(process_failures)
                try:
                    if last_failure == Failure.KILL_JOB:
                        FailureController.kill_job(scheduler)
                    elif hasattr(last_replica, 'actor') and last_replica.actor:
                        await last_replica.actor.inject_failure.call_one(last_failure)
                    elif hasattr(last_replica, 'failure_actors') and last_replica.failure_actors:
                        await last_replica.failure_actors.fail.choose(last_failure)
                    logger.info(
                        f"[FailureController] Failure injection ({last_failure}) sent to replica {last_replica.rid}"
                    )
                except Exception as e:
                    logger.exception(
                        f"[FailureController] Failed to inject failure in replica {last_replica.rid}: {e}"
                    )
                await asyncio.sleep(rest_time)
            except Exception as e:
                logger.exception(
                    f"[FailureController] Something went wrong while injecting failure: {e}"
                )
