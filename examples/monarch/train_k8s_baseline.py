# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
K8s FT training — BASELINE (no Monarch recovery benefits).

No __supervise__. When a process dies, the root actor crashes,
both replicas die, and the outer loop restarts everything from scratch
with fresh K8s pods. TorchFT loads from the last checkpoint.

This is the "what it would look like without Monarch" baseline.
Compare recovery time against train_k8s_minimal.py (HostMesh reuse).
"""

import argparse
import asyncio
import atexit
import os
import textwrap
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import torch
from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1PodSpec,
    V1PodTemplateSpec,
    V1ResourceRequirements,
    V1Volume,
    V1VolumeMount,
)
from monarch.actor import Actor, current_rank, endpoint, HostMesh, ProcMesh, this_host
from monarch.job.kubernetes import KubernetesJob
from monarch.spmd import setup_torch_elastic_env_async
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    TrainingConfig,
)
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ft.llama3 import model_registry
from torchtitan.experiments.ft.optimizer import FTOptimizersContainer
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import ProfilingConfig

try:
    from utils.failure import Failure, FailureActor, FailureController
except ModuleNotFoundError:
    FailureActor = None
    FailureController = None
    Failure = None


# ==== K8s allocation ====

_WORKER_BOOTSTRAP_SCRIPT: str = textwrap.dedent("""\
    import os
    import socket
    from monarch.actor import run_worker_loop_forever
    port = os.environ.get("MONARCH_PORT", "26600")
    hostname = socket.getfqdn()
    address = f"tcp://{hostname}:{port}"
    run_worker_loop_forever(address=address, ca="trust_all_connections")
""")


def build_gpu_pod_spec(image: str, gpus_per_host: int) -> V1PodSpec:
    gpu_resources = {"nvidia.com/gpu": str(gpus_per_host)}
    return V1PodSpec(
        containers=[
            V1Container(
                name="worker",
                image=image,
                command=["python", "-u", "-c", _WORKER_BOOTSTRAP_SCRIPT],
                env=[V1EnvVar(name="MONARCH_PORT", value="26600")],
                resources=V1ResourceRequirements(
                    limits=gpu_resources, requests=gpu_resources,
                ),
                volume_mounts=[V1VolumeMount(name="dshm", mount_path="/dev/shm")],
            )
        ],
        volumes=[
            V1Volume(
                name="dshm",
                empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi"),
            )
        ],
    )


# ==== Actors — NO __supervise__, failure kills everything ====


class TrainingActor(Actor):
    def __init__(self, trainer_config: FaultTolerantTrainer.Config, replica_id: int) -> None:
        self.trainer_config = trainer_config
        rank = current_rank().rank
        self.uid = f"[replica_{replica_id}_trainer_{rank}]"

    @endpoint(instrument=False)
    async def start_training(self, lighthouse_address: str) -> None:
        init_logger()

        os.environ["TORCHFT_LIGHTHOUSE"] = lighthouse_address
        trainer = self.trainer_config.build()
        logger.info(f"{self.uid} initialized successfully on {os.getpid()}")

        try:
            logger.info(f"{self.uid} starting training")
            trainer.train()
        except Exception:
            if trainer:
                trainer.close()
            raise
        else:
            trainer.close()
        finally:
            torch.distributed.destroy_process_group()
            logger.info(f"{self.uid} trainer cleaned up")


class ReplicaActor(Actor):
    """NO __supervise__. Failure propagates to root actor → script crashes."""

    def __init__(self, spec: "JobSpec", replica_id: int, scheduler) -> None:
        self.spec = deepcopy(spec)
        self.replica_id = replica_id
        self.spec.trainer_config.fault_tolerance.replica_id = replica_id
        self.scheduler = scheduler
        self.failure_actors = None
        self.uid = f"[replica_{replica_id}]"

    @endpoint(instrument=False)
    async def start_replica(self) -> None:
        init_logger()
        logger.info(f"{self.uid} Spawning trainers")

        trainers_proc_mesh = self.scheduler.proc_mesh(
            f"replica{self.replica_id}",
            num_procs=self.spec.gpus_per_host,
        )

        async with trainers_proc_mesh:
            await setup_torch_elastic_env_async(trainers_proc_mesh)

            training_actors = trainers_proc_mesh.spawn(
                "training_actors",
                TrainingActor,
                self.spec.trainer_config,
                self.replica_id,
            )

            if FailureActor is not None and self.spec.with_failures:
                self.failure_actors = trainers_proc_mesh.spawn(
                    "failure_actors", FailureActor
                )

            logger.info(f"{self.uid} Starting trainers")
            await training_actors.start_training.call(self.spec.lighthouse_address)

    @endpoint(instrument=False)
    async def inject_failure(self, failure_type: "Failure"):
        if self.failure_actors:
            try:
                logger.info(f"{self.uid} Injecting failure ({failure_type}) into random trainer")
                await self.failure_actors.fail.choose(failure_type)
            except Exception as e:
                logger.exception(f"{self.uid} Injected failure: {e}")
        else:
            logger.error(f"{self.uid} No failure actors available")


# ==== Orchestration ====


@dataclass
class JobSpec:
    trainer_config: FaultTolerantTrainer.Config
    replica_count: int
    gpus_per_host: int
    with_failures: bool
    namespace: str = ""
    image: str | None = None
    timeout: int | None = None
    lighthouse_address: str = ""


@dataclass
class Replica:
    rid: int
    proc_mesh: ProcMesh
    actor: "ReplicaActor"


class MonarchKubernetes:
    def __init__(self, namespace: str, image: str | None = None, gpus_per_host: int = 8, timeout: int | None = None):
        self.namespace = namespace
        self.image = image
        self.gpus_per_host = gpus_per_host
        self.timeout = timeout
        self.job_handles: Dict[str, KubernetesJob] = {}
        self._is_owner = True
        atexit.register(self.kill_jobs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_is_owner"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def get_or_create_job(self, mesh_name: str) -> None:
        job = KubernetesJob(namespace=self.namespace, timeout=self.timeout)
        if self.image is not None:
            pod_spec = build_gpu_pod_spec(self.image, self.gpus_per_host)
            job.add_mesh(mesh_name, num_replicas=1, pod_template=V1PodTemplateSpec(spec=pod_spec))
        else:
            job.add_mesh(mesh_name, num_replicas=1)
        self.job_handles[mesh_name] = job

    def kill_jobs(self):
        if not self._is_owner:
            return
        for mesh_name in list(self.job_handles.keys()):
            self.kill_job(mesh_name)

    def kill_job(self, mesh_name: str):
        try:
            job = self.job_handles.pop(mesh_name, None)
            if job is not None:
                logger.info(f"Destroying job for mesh {mesh_name}")
                job.kill()
        except Exception as e:
            logger.exception(f"Failed to destroy job for {mesh_name}: {e}")

    def proc_mesh(self, mesh_name: str, num_procs: int) -> ProcMesh:
        job = self.job_handles[mesh_name]
        mesh: HostMesh = getattr(job.state(cached_path=None), mesh_name)
        return mesh.spawn_procs({"gpus": num_procs})


class OrchestrationManager:
    def __init__(self, spec: JobSpec) -> None:
        self.spec = spec
        self.replicas: Dict[int, Replica] = {}
        self.lighthouse = None
        self.scheduler = MonarchKubernetes(
            namespace=spec.namespace,
            image=spec.image,
            gpus_per_host=spec.gpus_per_host,
            timeout=spec.timeout,
        )

    async def start_training(self) -> None:
        logger.info(f"[Controller] Creating training system with {self.spec.replica_count} replicas")

        for replica_id in range(self.spec.replica_count):
            await self.scheduler.get_or_create_job(f"replica{replica_id}")

        mesh_futures = {}
        for i in range(self.spec.replica_count):
            mesh_futures[i] = asyncio.create_task(self._run_replica(i))

        failure_future = None
        if self.spec.with_failures:
            failure_future = asyncio.create_task(
                FailureController.execute_failures(self.replicas, self.scheduler, startup_wait=120, rest_time=600)
            )

        await asyncio.gather(*mesh_futures.values(), return_exceptions=True)

        if failure_future:
            failure_future.cancel()

    def start_lighthouse(self) -> None:
        import socket as _socket
        from torchft.coordination import LighthouseServer

        self.lighthouse = LighthouseServer(bind="[::]:0", min_replicas=1, join_timeout_ms=60000)
        addr = self.lighthouse.address()
        short_hostname = _socket.gethostname()
        fqdn = _socket.getfqdn()
        self.spec.lighthouse_address = addr.replace(short_hostname, fqdn)
        logger.info(f"[Controller] Lighthouse started at {self.spec.lighthouse_address}")

    def stop_lighthouse(self) -> None:
        try:
            if self.lighthouse:
                self.lighthouse.shutdown()
            logger.info("[Controller] Lighthouse stopped")
        except Exception as e:
            logger.exception(f"[Controller] Failed to stop lighthouse: {e}")

    async def _run_replica(self, replica_id: int) -> None:
        spawn_start = time.time()
        replica_proc_mesh = this_host().spawn_procs({"gpus": 1})
        await replica_proc_mesh.logging_option(aggregate_window_sec=None)

        replica_actor = replica_proc_mesh.spawn(
            "replica_actor", ReplicaActor, self.spec, replica_id, self.scheduler
        )

        self.replicas[replica_id] = Replica(replica_id, replica_proc_mesh, replica_actor)

        logger.info(f"[Controller] Replica {replica_id} starting training (spawn took {time.time()-spawn_start:.1f}s)")
        await replica_actor.start_replica.call_one()


# === CLI / CONFIG === #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K8s FT Training — BASELINE (no Monarch recovery)")
    parser.add_argument("--replica-count", type=int, default=2)
    parser.add_argument("--gpus-per-host", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=10000)
    parser.add_argument("--tokenizer-path", type=str, default="/opt/torchtitan/tests/assets/tokenizer")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--with-failures", action="store_true")
    parser.add_argument("--namespace", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-restarts", type=int, default=3)
    return parser.parse_args()


def make_job_spec(args: argparse.Namespace) -> JobSpec:
    trainer_config = FaultTolerantTrainer.Config(
        hf_assets_path=args.tokenizer_path,
        profiling=ProfilingConfig(),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        model_spec=model_registry("debugmodel"),
        optimizer=FTOptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2, decay_ratio=0.8, decay_type="linear", min_lr_factor=0.0),
        training=TrainingConfig(local_batch_size=8, seq_len=2048, steps=args.training_steps),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4" if args.dataset_path is None else "c4_test",
            dataset_path=args.dataset_path,
        ),
        checkpoint=CheckpointManager.Config(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        comm=CommConfig(train_timeout_seconds=300),
        fault_tolerance=FaultTolerance(
            enable=True,
            group_size=args.gpus_per_host,
            process_group="nccl",
            process_group_timeout_ms=60000,
        ),
    )

    return JobSpec(
        trainer_config=trainer_config,
        replica_count=args.replica_count,
        gpus_per_host=args.gpus_per_host,
        with_failures=args.with_failures,
        namespace=args.namespace,
        image=args.image,
        timeout=args.timeout,
    )


async def run_training(job_spec: JobSpec) -> None:
    orchestrator = OrchestrationManager(job_spec)
    try:
        orchestrator.start_lighthouse()
        await orchestrator.start_training()
    finally:
        orchestrator.stop_lighthouse()
        orchestrator.scheduler.kill_jobs()


def main() -> None:
    init_logger()
    args = parse_args()
    job_spec = make_job_spec(args)

    for attempt in range(args.max_restarts + 1):
        restart_start = time.time()
        if attempt > 0:
            logger.info(f"===== FULL RESTART {attempt}/{args.max_restarts} — both replicas from scratch =====")

        try:
            asyncio.run(run_training(job_spec))
            logger.info("[Controller] Training completed successfully")
            break
        except (KeyboardInterrupt, Exception) as e:
            crash_time = time.time()
            elapsed = crash_time - restart_start
            logger.error(f"[Controller] CRASHED after {elapsed:.1f}s (attempt {attempt}): {type(e).__name__}")

            if attempt >= args.max_restarts:
                logger.error(f"[Controller] Max restarts reached. Giving up.")
                break

            logger.info(f"[Controller] Both replicas dead. Full restart with fresh pods...")


if __name__ == "__main__":
    main()
