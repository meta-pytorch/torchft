# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
K8s FT training — NO MONARCH baseline.

Uses torchrun-style launch directly on K8s pods. No Monarch actors,
no broadcast tree, no proc_mesh. Just TorchFT + TorchTitan + K8s.

Each replica is a K8s Job that runs training directly.
When a pod dies, K8s restarts it. TorchFT handles quorum and checkpoint.

Compare recovery time against train_k8s_minimal.py (Monarch HostMesh reuse).

Usage:
  1. Start lighthouse on controller:
     python train_k8s_baseline.py --mode lighthouse --namespace monarch-tests

  2. In separate terminals (or as K8s Jobs), start each replica:
     # On pod with 8 GPUs — replica 0:
     torchrun --nproc_per_node=8 train_k8s_baseline.py --mode train --replica-id 0 \
       --lighthouse-address <address> --gpus-per-host 8 --training-steps 10000

     # On pod with 8 GPUs — replica 1:
     torchrun --nproc_per_node=8 train_k8s_baseline.py --mode train --replica-id 1 \
       --lighthouse-address <address> --gpus-per-host 8 --training-steps 10000

  For the K8s automated version, use --mode controller which creates
  K8s Jobs for each replica and monitors them.
"""

import argparse
import os
import subprocess
import sys
import textwrap
import time

import torch
import torch.distributed as dist
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
    from utils.failure import Failure
    import ctypes
    import random
except ImportError:
    pass


def make_trainer_config(args) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
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
            replica_id=args.replica_id,
            group_size=args.gpus_per_host,
            process_group="nccl",
            process_group_timeout_ms=60000,
        ),
    )


def run_lighthouse(args):
    """Run the lighthouse server on the controller pod."""
    import socket
    from torchft.coordination import LighthouseServer

    init_logger()
    lighthouse = LighthouseServer(bind="[::]:0", min_replicas=1, join_timeout_ms=60000)
    addr = lighthouse.address()
    short_hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    full_addr = addr.replace(short_hostname, fqdn)
    logger.info(f"[Lighthouse] Listening at {full_addr}")
    logger.info(f"[Lighthouse] Use --lighthouse-address {full_addr}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        lighthouse.shutdown()
        logger.info("[Lighthouse] Stopped")


def run_train(args):
    """Run training directly — no Monarch, just torchrun + TorchFT."""
    init_logger()

    os.environ["TORCHFT_LIGHTHOUSE"] = args.lighthouse_address

    trainer_config = make_trainer_config(args)
    trainer = trainer_config.build()

    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
    logger.info(f"[replica_{args.replica_id}_rank_{rank}] initialized on pid={os.getpid()}")

    try:
        logger.info(f"[replica_{args.replica_id}_rank_{rank}] starting training")
        trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info(f"[replica_{args.replica_id}_rank_{rank}] done")


def run_controller(args):
    """
    Controller mode: creates K8s pods for each replica, monitors them,
    restarts on failure. Simulates what a job scheduler would do.
    """
    import socket
    from kubernetes import client as k8s_client, config as k8s_config
    from torchft.coordination import LighthouseServer

    init_logger()

    # Start lighthouse
    lighthouse = LighthouseServer(bind="[::]:0", min_replicas=1, join_timeout_ms=60000)
    addr = lighthouse.address()
    short_hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    lighthouse_address = addr.replace(short_hostname, fqdn)
    logger.info(f"[Controller] Lighthouse at {lighthouse_address}")

    # K8s API
    k8s_config.load_incluster_config()
    batch_v1 = k8s_client.BatchV1Api()
    core_v1 = k8s_client.CoreV1Api()

    train_script = textwrap.dedent(f"""\
        import os, sys
        os.environ["TORCHFT_LIGHTHOUSE"] = "{lighthouse_address}"
        sys.path.insert(0, "/workspace/torchft/examples/monarch")
        from train_k8s_baseline import run_train, parse_args
        import argparse
        args = argparse.Namespace(
            mode="train",
            replica_id=int(os.environ["REPLICA_ID"]),
            lighthouse_address="{lighthouse_address}",
            gpus_per_host={args.gpus_per_host},
            training_steps={args.training_steps},
            tokenizer_path="{args.tokenizer_path}",
            dataset_path={repr(args.dataset_path)},
        )
        run_train(args)
    """)

    gpu_resources = {"nvidia.com/gpu": str(args.gpus_per_host)}

    def create_replica_job(replica_id: int) -> str:
        job_name = f"baseline-replica{replica_id}"

        # Delete old job if exists
        try:
            batch_v1.delete_namespaced_job(
                name=job_name, namespace=args.namespace,
                body=k8s_client.V1DeleteOptions(propagation_policy="Foreground"))
            logger.info(f"[Controller] Deleted old job {job_name}")
            time.sleep(5)
        except k8s_client.ApiException as e:
            if e.status != 404:
                raise

        job = k8s_client.V1Job(
            metadata=k8s_client.V1ObjectMeta(name=job_name, namespace=args.namespace),
            spec=k8s_client.V1JobSpec(
                backoff_limit=10,
                template=k8s_client.V1PodTemplateSpec(
                    metadata=k8s_client.V1ObjectMeta(labels={"app": job_name}),
                    spec=k8s_client.V1PodSpec(
                        restart_policy="OnFailure",
                        containers=[
                            k8s_client.V1Container(
                                name="trainer",
                                image=args.image,
                                command=["torchrun", "--nproc_per_node", str(args.gpus_per_host),
                                         "-c", train_script],
                                env=[
                                    k8s_client.V1EnvVar(name="REPLICA_ID", value=str(replica_id)),
                                ],
                                resources=k8s_client.V1ResourceRequirements(
                                    limits=gpu_resources, requests=gpu_resources),
                                volume_mounts=[k8s_client.V1VolumeMount(name="dshm", mount_path="/dev/shm")],
                            )
                        ],
                        volumes=[
                            k8s_client.V1Volume(
                                name="dshm",
                                empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi")),
                        ],
                    ),
                ),
            ),
        )

        batch_v1.create_namespaced_job(namespace=args.namespace, body=job)
        logger.info(f"[Controller] Created job {job_name}")
        return job_name

    # Create replica jobs
    job_names = []
    for replica_id in range(args.replica_count):
        job_names.append(create_replica_job(replica_id))

    # Inject failure after startup_wait
    logger.info(f"[Controller] Waiting {args.startup_wait}s before failure injection")
    time.sleep(args.startup_wait)

    # Find a running pod for replica 0 and kill a process
    logger.info("[Controller] Injecting failure into replica 0")
    failure_time = time.time()
    try:
        pods = core_v1.list_namespaced_pod(
            namespace=args.namespace, label_selector=f"app=baseline-replica0")
        if pods.items:
            pod_name = pods.items[0].metadata.name
            logger.info(f"[Controller] Killing process on {pod_name}")
            # Kill one of the training processes
            from kubernetes.stream import stream
            stream(core_v1.connect_get_namespaced_pod_exec,
                   pod_name, args.namespace,
                   command=["bash", "-c", "kill -9 $(pgrep -f 'torchrun' | head -1)"],
                   container="trainer", stderr=True, stdout=True, stdin=False, tty=False)
    except Exception as e:
        logger.error(f"[Controller] Failure injection failed: {e}")
        logger.info("[Controller] Continuing — K8s will handle pod restart")

    # Monitor until training completes or timeout
    logger.info("[Controller] Monitoring jobs...")
    timeout = time.time() + 1800  # 30 min timeout
    while time.time() < timeout:
        all_done = True
        for job_name in job_names:
            try:
                job = batch_v1.read_namespaced_job(name=job_name, namespace=args.namespace)
                if job.status.succeeded and job.status.succeeded >= 1:
                    continue
                all_done = False
                if job.status.failed and job.status.failed > 0:
                    recovery_time = time.time() - failure_time
                    logger.info(f"[Controller] {job_name} has {job.status.failed} failures, "
                                f"recovery ongoing ({recovery_time:.1f}s since injection)")
            except Exception:
                all_done = False

        if all_done:
            total_time = time.time() - failure_time
            logger.info(f"[Controller] All jobs completed. Total time since failure: {total_time:.1f}s")
            break

        time.sleep(5)

    # Cleanup
    for job_name in job_names:
        try:
            batch_v1.delete_namespaced_job(
                name=job_name, namespace=args.namespace,
                body=k8s_client.V1DeleteOptions(propagation_policy="Foreground"))
        except Exception:
            pass

    lighthouse.shutdown()
    logger.info("[Controller] Done")


def parse_args():
    parser = argparse.ArgumentParser(description="K8s FT Training — NO MONARCH baseline")
    parser.add_argument("--mode", required=True, choices=["lighthouse", "train", "controller"],
                        help="lighthouse: run lighthouse server; train: run training; controller: full automated run")
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--replica-count", type=int, default=2)
    parser.add_argument("--gpus-per-host", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=10000)
    parser.add_argument("--tokenizer-path", type=str, default="/opt/torchtitan/tests/assets/tokenizer")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--lighthouse-address", type=str, default="")
    parser.add_argument("--namespace", type=str, default="monarch-tests")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--startup-wait", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "lighthouse":
        run_lighthouse(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "controller":
        run_controller(args)
