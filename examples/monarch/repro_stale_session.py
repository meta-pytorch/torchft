"""
Repro: stale TCP session blocks HostMesh reuse after process death.

Matches train_k8s_minimal.py's exact code path:
- Scheduler object with job_handles
- ReplicaActor on this_host().spawn_procs() calls scheduler.proc_mesh()
- Kill a process, teardown ReplicaActor, spawn new one
- New ReplicaActor calls scheduler.proc_mesh() → expect "out-of-sequence"
"""

import argparse
import asyncio
import os
import textwrap
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

from monarch.actor import Actor, current_rank, endpoint, HostMesh, ProcMesh, this_host
from monarch.job.kubernetes import KubernetesJob
from kubernetes.client import (
    V1Container, V1EmptyDirVolumeSource, V1EnvVar, V1PodSpec,
    V1PodTemplateSpec, V1ResourceRequirements, V1Volume, V1VolumeMount,
)

_WORKER_SCRIPT = textwrap.dedent("""\
    import os, socket
    from monarch.actor import run_worker_loop_forever
    port = os.environ.get("MONARCH_PORT", "26600")
    address = f"tcp://{socket.getfqdn()}:{port}"
    run_worker_loop_forever(address=address, ca="trust_all_connections")
""")


class PingActor(Actor):
    @endpoint(instrument=False)
    async def ping(self) -> str:
        import socket
        return f"rank={current_rank().rank} pid={os.getpid()} host={socket.gethostname()}"

    @endpoint(instrument=False)
    async def die(self) -> None:
        os._exit(1)


class MonarchKubernetes:
    """Same scheduler as train_k8s_minimal.py"""
    def __init__(self, namespace, image, gpus):
        self.namespace = namespace
        self.image = image
        self.gpus = gpus
        self.job_handles: Dict[str, KubernetesJob] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def get_or_create_job(self, mesh_name):
        gpu_res = {"nvidia.com/gpu": str(self.gpus)}
        pod_spec = V1PodSpec(
            containers=[V1Container(
                name="worker", image=self.image,
                command=["python", "-u", "-c", _WORKER_SCRIPT],
                env=[V1EnvVar(name="MONARCH_PORT", value="26600")],
                resources=V1ResourceRequirements(limits=gpu_res, requests=gpu_res),
                volume_mounts=[V1VolumeMount(name="dshm", mount_path="/dev/shm")],
            )],
            volumes=[V1Volume(name="dshm",
                              empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi"))],
        )
        job = KubernetesJob(namespace=self.namespace)
        job.add_mesh(mesh_name, num_replicas=1, pod_template=pod_spec)
        self.job_handles[mesh_name] = job

    def proc_mesh(self, mesh_name, num_procs):
        job = self.job_handles[mesh_name]
        mesh: HostMesh = getattr(job.state(cached_path=None), mesh_name)
        return mesh.spawn_procs({"gpus": num_procs})


class ReplicaActor(Actor):
    """Same pattern as train_k8s_minimal.py — owns the proc_mesh."""
    def __init__(self, scheduler, mesh_name, gpus):
        self.scheduler = scheduler
        self.mesh_name = mesh_name
        self.gpus = gpus
        self.failed = False

    async def __supervise__(self, failure) -> bool:
        print(f"  [ReplicaActor] __supervise__: {type(failure).__name__}")
        self.failed = True
        return True

    @endpoint(instrument=False)
    async def run(self) -> str:
        pm = self.scheduler.proc_mesh(self.mesh_name, self.gpus)
        actors = pm.spawn("ping", PingActor)
        r = await actors.ping.call()
        print(f"  [ReplicaActor] Ping OK on {self.mesh_name}")

        print(f"  [ReplicaActor] Killing one process")
        try:
            await actors.die.choose()
        except Exception:
            pass
        if self.failed:
            return "KILLED"
        return "NO_FAILURE"


@dataclass
class Replica:
    proc_mesh: ProcMesh
    actor: ReplicaActor


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--wait", type=int, default=90)
    args = parser.parse_args()

    # Clean up old CRDs
    print("Cleaning up old CRDs...")
    for name in ["replica0"]:
        try:
            j = KubernetesJob(namespace=args.namespace)
            gpu_res = {"nvidia.com/gpu": str(args.gpus)}
            j.add_mesh(name, num_replicas=1, pod_template=V1PodSpec(
                containers=[V1Container(name="worker", image=args.image,
                    command=["python", "-u", "-c", _WORKER_SCRIPT],
                    env=[V1EnvVar(name="MONARCH_PORT", value="26600")],
                    resources=V1ResourceRequirements(
                        limits=gpu_res, requests=gpu_res),
                    volume_mounts=[V1VolumeMount(name="dshm", mount_path="/dev/shm")])],
                volumes=[V1Volume(name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi"))]))
            j.kill()
        except Exception:
            pass
    from kubernetes import client as k8s_client, config as k8s_config
    k8s_config.load_incluster_config()
    api = k8s_client.CustomObjectsApi()
    for name in ["replica0"]:
        for _ in range(60):
            try:
                api.get_namespaced_custom_object(
                    group="monarch.pytorch.org", version="v1",
                    namespace=args.namespace, plural="monarchmeshes", name=name)
                time.sleep(2)
            except k8s_client.ApiException as e:
                if e.status == 404:
                    break
    print("Clean.\n")

    # Create scheduler — same as train_k8s_minimal.py
    scheduler = MonarchKubernetes(args.namespace, args.image, args.gpus)
    await scheduler.get_or_create_job("replica0")

    # === Attempt 1: spawn ReplicaActor, ping, kill ===
    print("1. Spawning ReplicaActor (attempt 1)")
    owner_pm = this_host().spawn_procs({"gpus": 1})
    owner = owner_pm.spawn("replica", ReplicaActor, scheduler, "replica0", args.gpus)
    try:
        result = await owner.run.call_one()
        print(f"   {result}")
    except Exception as e:
        print(f"   Error: {e}")

    # Teardown — same as train_k8s_minimal.py _teardown
    print("\n2. Tearing down ReplicaActor")
    try:
        await owner_pm.stop()
        print("   Stopped.")
    except Exception as e:
        print(f"   Stop failed: {e}")

    # Wait for orphans
    print(f"\n3. Waiting {args.wait}s for orphan cleanup")
    for r in range(args.wait, 0, -10):
        print(f"   {r}s...")
        await asyncio.sleep(min(10, r))

    # === Attempt 2: spawn NEW ReplicaActor on same scheduler ===
    print("\n4. Spawning NEW ReplicaActor (attempt 2) — same scheduler, same job")
    t = time.time()
    owner_pm2 = this_host().spawn_procs({"gpus": 1})
    owner2 = owner_pm2.spawn("replica2", ReplicaActor, scheduler, "replica0", args.gpus)
    try:
        result2 = await owner2.run.call_one()
        print(f"   {result2}")
        print(f"\n=== RESULT: Reconnection WORKS ({time.time()-t:.1f}s) ===")
    except Exception as e:
        print(f"   FAILED: {e}")
        print(f"\n=== RESULT: Reconnection BLOCKED ({time.time()-t:.1f}s) ===")

    # Cleanup
    print("\nCleaning up...")
    scheduler.job_handles["replica0"].kill()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
