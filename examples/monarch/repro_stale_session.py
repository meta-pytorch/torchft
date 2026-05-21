"""
Repro: Monarch worker rejects new ProcMesh after process death on same HostMesh.

Setup: 2 K8s pods × 8 GPUs (matches training setup).

Steps:
  1. Create 2 KubernetesJobs (replica0, replica1), spawn 8-GPU ProcMesh on each
  2. Spawn actors, ping all 16 — verify both replicas work
  3. Kill one process on replica1 via segfault
  4. Wait for orphan cleanup
  5. Spawn a NEW ProcMesh on replica1's same HostMesh
  6. Try to spawn actors and ping — expect "out-of-sequence" failure

Expected: Step 6 fails with "out-of-sequence message, expected seq 0, got N"
  because the worker loop keeps stale TCP session state from the killed connection.

This blocks the HostMesh reuse recovery path for fault-tolerant training on K8s.
"""

import argparse
import asyncio
import ctypes
import os
import textwrap
import time

from monarch.actor import Actor, current_rank, endpoint, this_host
from monarch.job.kubernetes import KubernetesJob

from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1PodSpec,
    V1ResourceRequirements,
    V1Volume,
    V1VolumeMount,
)

_WORKER_BOOTSTRAP_SCRIPT = textwrap.dedent("""\
    import os, socket
    from monarch.actor import run_worker_loop_forever
    port = os.environ.get("MONARCH_PORT", "26600")
    hostname = socket.getfqdn()
    address = f"tcp://{hostname}:{port}"
    run_worker_loop_forever(address=address, ca="trust_all_connections")
""")


class PingActor(Actor):
    @endpoint(instrument=False)
    async def ping(self) -> str:
        rank = current_rank().rank
        pid = os.getpid()
        import socket
        hostname = socket.gethostname()
        return f"pong from rank={rank} pid={pid} host={hostname}"


class CrashActor(Actor):
    @endpoint(instrument=False)
    async def crash(self) -> None:
        rank = current_rank().rank
        print(f"[CrashActor] rank={rank} pid={os.getpid()} — triggering segfault")
        crash_func = ctypes.CFUNCTYPE(None)()
        crash_func()


def build_pod_spec(image: str, gpus: int) -> V1PodSpec:
    gpu_resources = {"nvidia.com/gpu": str(gpus)}
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


async def main():
    parser = argparse.ArgumentParser(description="Repro: stale TCP session on HostMesh reuse")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--wait", type=int, default=90,
                        help="Seconds to wait for orphan cleanup before respawn (default: 90)")
    args = parser.parse_args()

    pod_spec = build_pod_spec(args.image, args.gpus)

    # === Step 1: Create 2 jobs ===
    print(f"\n=== STEP 1: Creating 2 K8s jobs ({args.gpus} GPUs each) ===")
    job0 = KubernetesJob(namespace=args.namespace)
    job0.add_mesh("replica0", num_replicas=1, pod_spec=pod_spec)
    job1 = KubernetesJob(namespace=args.namespace)
    job1.add_mesh("replica1", num_replicas=1, pod_spec=pod_spec)

    hm0 = getattr(job0.state(cached_path=None), "replica0")
    hm1 = getattr(job1.state(cached_path=None), "replica1")

    print(f"Spawning {args.gpus}-GPU ProcMesh on each replica...")
    pm0 = hm0.spawn_procs({"gpus": args.gpus})
    pm1 = hm1.spawn_procs({"gpus": args.gpus})

    # === Step 2: Ping all actors ===
    print("\n=== STEP 2: Spawning actors and pinging all ranks ===")
    actors0 = pm0.spawn("ping0", PingActor)
    actors1 = pm1.spawn("ping1", PingActor)

    r0 = await actors0.ping.call()
    print(f"  replica0: {r0}")
    r1 = await actors1.ping.call()
    print(f"  replica1: {r1}")
    print("Both replicas responding OK.")

    # === Step 3: Kill one process on replica1 ===
    print("\n=== STEP 3: Killing one process on replica1 via segfault ===")
    crash_actors = pm1.spawn("crash", CrashActor)
    try:
        await crash_actors.crash.choose()
    except Exception as e:
        print(f"  Crash triggered (expected error): {type(e).__name__}")

    # === Step 4: Wait for orphan cleanup ===
    print(f"\n=== STEP 4: Waiting {args.wait}s for orphan cleanup ===")
    for i in range(args.wait, 0, -10):
        print(f"  {i}s remaining...")
        await asyncio.sleep(min(10, i))
    print("  Done waiting.")

    # === Step 5: Spawn NEW ProcMesh on same HostMesh ===
    print(f"\n=== STEP 5: Spawning NEW ProcMesh on replica1's HostMesh ===")
    t_start = time.time()
    try:
        pm1_new = hm1.spawn_procs({"gpus": args.gpus})
        print(f"  spawn_procs succeeded in {time.time() - t_start:.1f}s")
    except Exception as e:
        print(f"  FAILED to spawn_procs: {type(e).__name__}: {e}")
        print("\n=== RESULT: HostMesh reuse BLOCKED at spawn_procs ===")
        job0.kill()
        job1.kill()
        return

    # === Step 6: Try to use new ProcMesh ===
    print(f"\n=== STEP 6: Spawning actors on new ProcMesh and pinging ===")
    try:
        actors1_new = pm1_new.spawn("ping1_new", PingActor)
        r1_new = await actors1_new.ping.call()
        print(f"  replica1 (new): {r1_new}")
        print(f"\n=== RESULT: HostMesh reuse WORKS! Recovery took {time.time() - t_start:.1f}s ===")
    except Exception as e:
        print(f"  FAILED to ping new actors: {type(e).__name__}: {e}")
        print(f"\n=== RESULT: HostMesh reuse BLOCKED — stale TCP session ===")

    # Cleanup
    print("\nCleaning up...")
    job0.kill()
    job1.kill()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
