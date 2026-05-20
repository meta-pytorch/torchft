"""Simple test: spawn actors on 2 pods, verify both respond."""
import asyncio
import os
import textwrap

from monarch.actor import Actor, current_rank, endpoint, this_host
from monarch.config import configure
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

configure(enable_log_forwarding=True, message_delivery_timeout="2m")

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


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    gpu_resources = {"nvidia.com/gpu": str(args.gpus)}
    pod_spec = V1PodSpec(
        containers=[V1Container(
            name="worker",
            image=args.image,
            command=["python", "-u", "-c", _WORKER_BOOTSTRAP_SCRIPT],
            env=[V1EnvVar(name="MONARCH_PORT", value="26600")],
            resources=V1ResourceRequirements(limits=gpu_resources, requests=gpu_resources),
            volume_mounts=[V1VolumeMount(name="dshm", mount_path="/dev/shm")],
        )],
        volumes=[V1Volume(name="dshm", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi"))],
    )

    print("Creating 2 jobs...")
    job0 = KubernetesJob(namespace=args.namespace)
    job0.add_mesh("replica0", num_replicas=1, pod_spec=pod_spec)

    job1 = KubernetesJob(namespace=args.namespace)
    job1.add_mesh("replica1", num_replicas=1, pod_spec=pod_spec)

    print("Getting host meshes...")
    hm0 = getattr(job0.state(cached_path=None), "replica0")
    hm1 = getattr(job1.state(cached_path=None), "replica1")

    print("Spawning procs on replica0...")
    pm0 = hm0.spawn_procs({"gpus": args.gpus})
    print("Spawning procs on replica1...")
    pm1 = hm1.spawn_procs({"gpus": args.gpus})

    print("Spawning ping actors...")
    actors0 = pm0.spawn("ping0", PingActor)
    actors1 = pm1.spawn("ping1", PingActor)

    print("Pinging replica0...")
    r0 = await actors0.ping.call_one()
    print(f"  replica0: {r0}")

    print("Pinging replica1...")
    r1 = await actors1.ping.call_one()
    print(f"  replica1: {r1}")

    print("\nBOTH REPLICAS WORKING!")

    print("Cleaning up...")
    await pm0.stop()
    await pm1.stop()
    job0.kill()
    job1.kill()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
