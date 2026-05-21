"""
Repro: After a process dies on a K8s pod, Monarch's worker rejects
new connections with "out-of-sequence" errors, blocking HostMesh reuse.

Single command. Uses the same __supervise__ + ownership pattern that
works in train_k8s_minimal.py — the actor that owns the proc_mesh
catches the failure, script survives, then tries to reconnect.
"""

import argparse
import asyncio
import os
import textwrap
import time

from copy import deepcopy
from monarch.actor import Actor, current_rank, endpoint, this_host
from monarch.job.kubernetes import KubernetesJob
from kubernetes.client import (
    V1Container, V1EmptyDirVolumeSource, V1EnvVar, V1PodSpec,
    V1ResourceRequirements, V1Volume, V1VolumeMount,
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


class ReplicaOwner(Actor):
    """Owns the proc_mesh. __supervise__ catches all child failures including logger."""

    def __init__(self, namespace, image, gpus, mesh_name):
        self.namespace = namespace
        self.image = image
        self.gpus = gpus
        self.mesh_name = mesh_name
        self.failure_occurred = False

    async def __supervise__(self, failure) -> bool:
        print(f"  [ReplicaOwner] __supervise__ caught: {type(failure).__name__}")
        self.failure_occurred = True
        return True

    @endpoint(instrument=False)
    async def ping_and_kill(self) -> str:
        gpu_res = {"nvidia.com/gpu": str(self.gpus)}
        spec = V1PodSpec(
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
        job.add_mesh(self.mesh_name, num_replicas=1, pod_spec=spec)
        hm = getattr(job.state(cached_path=None), self.mesh_name)
        pm = hm.spawn_procs({"gpus": self.gpus})

        actors = pm.spawn("ping", PingActor)
        r = await actors.ping.call()
        print(f"  [ReplicaOwner] Ping OK: {self.mesh_name}")

        print(f"  [ReplicaOwner] Killing one process on {self.mesh_name}")
        try:
            await actors.die.choose()
        except Exception as e:
            if self.failure_occurred:
                return "KILLED"
            raise
        return "KILLED"


def make_job(namespace, image, gpus, name):
    gpu_res = {"nvidia.com/gpu": str(gpus)}
    spec = V1PodSpec(
        containers=[V1Container(
            name="worker", image=image,
            command=["python", "-u", "-c", _WORKER_SCRIPT],
            env=[V1EnvVar(name="MONARCH_PORT", value="26600")],
            resources=V1ResourceRequirements(limits=gpu_res, requests=gpu_res),
            volume_mounts=[V1VolumeMount(name="dshm", mount_path="/dev/shm")],
        )],
        volumes=[V1Volume(name="dshm",
                          empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi"))],
    )
    job = KubernetesJob(namespace=namespace)
    job.add_mesh(name, num_replicas=1, pod_spec=spec)
    return job


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--wait", type=int, default=90)
    args = parser.parse_args()

    # Clean slate
    print("Cleaning up old CRDs...")
    for name in ["replica0", "replica1"]:
        try:
            make_job(args.namespace, args.image, args.gpus, name).kill()
        except Exception:
            pass
    from kubernetes import client as k8s_client, config as k8s_config
    k8s_config.load_incluster_config()
    api = k8s_client.CustomObjectsApi()
    for name in ["replica0", "replica1"]:
        for _ in range(60):
            try:
                api.get_namespaced_custom_object(
                    group="monarch.pytorch.org", version="v1",
                    namespace=args.namespace, plural="monarchmeshes", name=name)
                time.sleep(2)
            except k8s_client.ApiException as e:
                if e.status == 404:
                    break
    print("Clean.")

    # Step 1: Verify replica0 works (direct, no ownership tricks)
    print(f"\n1. Creating replica0, pinging {args.gpus} GPUs")
    job0 = make_job(args.namespace, args.image, args.gpus, "replica0")
    hm0 = getattr(job0.state(cached_path=None), "replica0")
    pm0 = hm0.spawn_procs({"gpus": args.gpus})
    a0 = pm0.spawn("p0", PingActor)
    await a0.ping.call()
    print("   replica0 OK")

    # Step 2: Use ReplicaOwner to ping replica1 then kill a process
    print(f"\n2. Spawning ReplicaOwner to manage replica1")
    owner_pm = this_host().spawn_procs({"gpus": 1})
    owner = owner_pm.spawn("owner", ReplicaOwner,
                           args.namespace, args.image, args.gpus, "replica1")
    try:
        result = await owner.ping_and_kill.call_one()
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Exception (expected): {type(e).__name__}: {e}")

    # Stop owner — orphans replica1's procs
    try:
        await owner_pm.stop()
    except Exception:
        pass

    # Step 3: Wait for orphans
    print(f"\n3. Waiting {args.wait}s for orphan cleanup")
    for r in range(args.wait, 0, -10):
        print(f"   {r}s...")
        await asyncio.sleep(min(10, r))

    # Step 4: Try to reuse replica1's HostMesh
    print("\n4. Reconnecting to replica1's HostMesh")
    t = time.time()
    try:
        job1_new = make_job(args.namespace, args.image, args.gpus, "replica1")
        hm1 = getattr(job1_new.state(cached_path=None), "replica1")
        pm1_new = hm1.spawn_procs({"gpus": args.gpus})
        a1_new = pm1_new.spawn("p1_new", PingActor)
        await a1_new.ping.call()
        print(f"   RESULT: HostMesh reuse WORKS ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"   RESULT: HostMesh reuse BLOCKED ({time.time()-t:.1f}s)")
        print(f"   Error: {e}")

    # Cleanup
    print("\nCleaning up...")
    try:
        job0.kill()
    except Exception:
        pass
    try:
        job1_new.kill()
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
