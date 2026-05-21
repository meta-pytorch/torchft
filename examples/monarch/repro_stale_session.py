"""
Repro: After a process dies on a K8s pod, can we spawn a new ProcMesh
on the SAME HostMesh (same job, same pod)?

Single command. Creates 2 jobs, pings, kills a process via a supervised
owner actor, waits for orphan cleanup, then tries spawn_procs on the
same HostMesh again.
"""

import argparse
import asyncio
import os
import textwrap
import time

from monarch.actor import Actor, current_rank, endpoint, HostMesh, this_host
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


class KillOwner(Actor):
    """Spawns actors on a given HostMesh, pings, kills one.
    Owns the proc_mesh so __supervise__ catches the failure."""

    def __init__(self, gpus: int) -> None:
        self.gpus = gpus
        self.failed = False

    async def __supervise__(self, failure) -> bool:
        print(f"  [KillOwner] __supervise__ caught: {type(failure).__name__}")
        self.failed = True
        return True

    @endpoint(instrument=False)
    async def ping_then_kill(self, host_mesh: HostMesh) -> str:
        pm = host_mesh.spawn_procs({"gpus": self.gpus})
        actors = pm.spawn("victims", PingActor)
        await actors.ping.call()
        print("  [KillOwner] Ping OK, now killing one process")
        try:
            await actors.die.choose()
        except Exception:
            pass
        if self.failed:
            return "KILLED"
        return "NO_FAILURE_DETECTED"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--wait", type=int, default=90)
    args = parser.parse_args()

    gpu_res = {"nvidia.com/gpu": str(args.gpus)}
    pod_spec = V1PodSpec(
        containers=[V1Container(
            name="worker", image=args.image,
            command=["python", "-u", "-c", _WORKER_SCRIPT],
            env=[V1EnvVar(name="MONARCH_PORT", value="26600")],
            resources=V1ResourceRequirements(limits=gpu_res, requests=gpu_res),
            volume_mounts=[V1VolumeMount(name="dshm", mount_path="/dev/shm")],
        )],
        volumes=[V1Volume(name="dshm",
                          empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi"))],
    )

    # Clean slate
    print("Cleaning up old CRDs...")
    for name in ["replica0", "replica1"]:
        try:
            j = KubernetesJob(namespace=args.namespace)
            j.add_mesh(name, num_replicas=1, pod_spec=pod_spec)
            j.kill()
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
    print("Clean.\n")

    # Step 1: Create jobs, get HostMeshes
    print(f"1. Creating 2 jobs ({args.gpus} GPUs each)")
    job0 = KubernetesJob(namespace=args.namespace)
    job0.add_mesh("replica0", num_replicas=1, pod_spec=pod_spec)
    job1 = KubernetesJob(namespace=args.namespace)
    job1.add_mesh("replica1", num_replicas=1, pod_spec=pod_spec)

    hm0 = getattr(job0.state(cached_path=None), "replica0")
    hm1 = getattr(job1.state(cached_path=None), "replica1")

    # Step 2: Verify replica0 works directly
    print("\n2. Pinging replica0")
    pm0 = hm0.spawn_procs({"gpus": args.gpus})
    a0 = pm0.spawn("p0", PingActor)
    await a0.ping.call()
    print("   OK")

    # Step 3: Use KillOwner to ping replica1 then kill a process
    print("\n3. Pinging replica1, then killing one process (supervised)")
    owner_pm = this_host().spawn_procs({"gpus": 1})
    owner = owner_pm.spawn("kill_owner", KillOwner, args.gpus)
    result = await owner.ping_then_kill.call_one(hm1)
    print(f"   {result}")

    try:
        await owner_pm.stop()
    except Exception:
        pass

    # Step 4: Wait for orphans
    print(f"\n4. Waiting {args.wait}s for orphan cleanup")
    for r in range(args.wait, 0, -10):
        print(f"   {r}s...")
        await asyncio.sleep(min(10, r))

    # Step 5a: Reuse via held hm1 reference (direct)
    print("\n5a. Spawning via HELD hm1 reference (direct)")
    t = time.time()
    try:
        pm1_a = hm1.spawn_procs({"gpus": args.gpus})
        a1_a = pm1_a.spawn("p1_direct", PingActor)
        await a1_a.ping.call()
        print(f"   DIRECT reuse WORKS ({time.time()-t:.1f}s)")
        await pm1_a.stop()
    except Exception as e:
        print(f"   DIRECT reuse BLOCKED ({time.time()-t:.1f}s): {e}")

    # Wait again for cleanup from 5a
    print("   Waiting 30s for cleanup...")
    await asyncio.sleep(30)

    # Step 5b: Reuse via job.state() (matches training script path)
    print("\n5b. Spawning via job.state() path (matches training script)")
    t = time.time()
    try:
        hm1_fresh = getattr(job1.state(cached_path=None), "replica1")
        pm1_b = hm1_fresh.spawn_procs({"gpus": args.gpus})
        a1_b = pm1_b.spawn("p1_jobstate", PingActor)
        await a1_b.ping.call()
        print(f"   JOB.STATE reuse WORKS ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"   JOB.STATE reuse BLOCKED ({time.time()-t:.1f}s): {e}")

    print("\nCleaning up...")
    job0.kill()
    job1.kill()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
