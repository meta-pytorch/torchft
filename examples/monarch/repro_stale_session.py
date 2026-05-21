"""
Repro: Monarch worker rejects new ProcMesh after process death on same HostMesh.

Setup: 2 K8s pods × 8 GPUs (matches training setup).

Steps:
  1. Create 2 KubernetesJobs (replica0, replica1), spawn 8-GPU ProcMesh on each
  2. Spawn actors, ping all 16 — verify both replicas work
  3. Kill one process on replica1 by kubectl exec kill
  4. Wait for orphan cleanup
  5. Spawn a NEW ProcMesh on replica1's same HostMesh
  6. Try to spawn actors and ping — expect "out-of-sequence" failure

Expected: Step 6 fails with "out-of-sequence message, expected seq 0, got N"
  because the worker loop keeps stale TCP session state from the killed connection.

This blocks the HostMesh reuse recovery path for fault-tolerant training on K8s.
"""

import argparse
import asyncio
import os
import subprocess
import textwrap
import time

from monarch.actor import Actor, current_rank, endpoint
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

    @endpoint(instrument=False)
    async def get_pid(self) -> int:
        return os.getpid()


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


def kill_process_on_pod(namespace: str, pod_name: str, pid: int) -> None:
    """Kill a specific process on a K8s pod via kubectl exec."""
    cmd = ["kubectl", "exec", "-n", namespace, pod_name, "--", "kill", "-9", str(pid)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Killed pid {pid} on {pod_name}")
    else:
        print(f"  kill returned {result.returncode}: {result.stderr.strip()}")


async def main():
    parser = argparse.ArgumentParser(description="Repro: stale TCP session on HostMesh reuse")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--wait", type=int, default=90,
                        help="Seconds to wait for orphan cleanup before respawn (default: 90)")
    args = parser.parse_args()

    pod_spec = build_pod_spec(args.image, args.gpus)

    # === Step 0: Clean up any leftover CRDs from previous runs ===
    print("\n=== STEP 0: Cleaning up leftover MonarchMesh CRDs ===")
    cleanup0 = KubernetesJob(namespace=args.namespace)
    cleanup0.add_mesh("replica0", num_replicas=1, pod_spec=pod_spec)
    cleanup1 = KubernetesJob(namespace=args.namespace)
    cleanup1.add_mesh("replica1", num_replicas=1, pod_spec=pod_spec)
    try:
        cleanup0.kill()
    except Exception:
        pass
    try:
        cleanup1.kill()
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
                    namespace=args.namespace, plural="monarchmeshes", name=name,
                )
                print(f"  Waiting for '{name}' to be deleted...")
                await asyncio.sleep(2)
            except k8s_client.ApiException as e:
                if e.status == 404:
                    print(f"  '{name}' deleted.")
                    break
    print("  Clean slate.")

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
    print(f"  replica0: OK ({args.gpus} ranks responded)")
    r1 = await actors1.ping.call()
    print(f"  replica1: OK ({args.gpus} ranks responded)")

    # Get a PID from replica1 rank 0 so we can kill it externally
    pids = await actors1.get_pid.call()
    print(f"  replica1 pids: {pids}")
    # Extract first PID from the result
    target_pid = None
    for item in pids:
        if isinstance(item, tuple):
            target_pid = item[1]
        else:
            target_pid = item
        break
    if target_pid is None:
        print("  ERROR: Could not extract PID from result")
        job0.kill()
        job1.kill()
        return
    print(f"  replica1 target pid: {target_pid}")

    # === Step 3: Kill one process on replica1 via kubectl ===
    print("\n=== STEP 3: Killing replica1 rank 0 process via kubectl exec ===")
    kill_process_on_pod(args.namespace, "replica1-0", target_pid)

    # === Step 4: Wait for orphan cleanup ===
    print(f"\n=== STEP 4: Waiting {args.wait}s for orphan cleanup ===")
    for remaining in range(args.wait, 0, -10):
        print(f"  {remaining}s remaining...")
        await asyncio.sleep(min(10, remaining))
    print("  Done waiting.")

    # === Step 5: Spawn NEW ProcMesh on same HostMesh ===
    print(f"\n=== STEP 5: Spawning NEW ProcMesh on replica1's HostMesh ===")
    t_start = time.time()
    try:
        pm1_new = hm1.spawn_procs({"gpus": args.gpus})
        elapsed = time.time() - t_start
        print(f"  spawn_procs succeeded in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  FAILED to spawn_procs after {elapsed:.1f}s: {type(e).__name__}: {e}")
        print("\n=== RESULT: HostMesh reuse BLOCKED at spawn_procs ===")
        job0.kill()
        job1.kill()
        return

    # === Step 6: Try to use new ProcMesh ===
    print(f"\n=== STEP 6: Spawning actors on new ProcMesh and pinging ===")
    try:
        actors1_new = pm1_new.spawn("ping1_new", PingActor)
        r1_new = await actors1_new.ping.call()
        total = time.time() - t_start
        print(f"  replica1 (new): OK")
        print(f"\n=== RESULT: HostMesh reuse WORKS! Recovery took {total:.1f}s ===")
    except Exception as e:
        total = time.time() - t_start
        print(f"  FAILED after {total:.1f}s: {type(e).__name__}: {e}")
        print(f"\n=== RESULT: HostMesh reuse BLOCKED — stale TCP session ===")

    # Cleanup
    print("\nCleaning up...")
    job0.kill()
    job1.kill()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
