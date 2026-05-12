### Monarch-TorchFT-TorchTitan Distributed Training Orchestrator

#### Overview
This directory contains scripts for orchestrating fault-tolerant distributed training using
TorchTitan and Monarch. Two scheduler backends are supported:

- **`train_distributed.py`** — SLURM-based (bare-metal / HPC clusters)
- **`train_distributed_k8s.py`** — Kubernetes-based (containerized environments)

Both scripts manage multiple training replicas with automatic failure recovery and
TorchFT lighthouse coordination.

##### PREREQUISITES

**Common:**
- Python 3.12+
- PyTorch with CUDA support
- `torchmonarch` >= 0.4.1
- `torchft` (TorchFT fault tolerance library)
- `torchtitan` (training framework)

**SLURM (`train_distributed.py`):**
- Access to a SLURM cluster with GPU nodes
- Munge authentication configured across nodes
- Training dataset (`c4_test`) and tokenizer in script directory

**Kubernetes (`train_distributed_k8s.py`):**
- Access to a Kubernetes cluster with GPU nodes
- Monarch K8s operator installed (creates MonarchMesh CRDs)
- A controller pod with RBAC permissions and a headless Service for DNS (see `launcher.yaml`)
- A container image with Monarch, TorchFT, and TorchTitan installed
- Tokenizer baked into the image (default: `/opt/torchtitan/tests/assets/tokenizer`)
- Dataset: downloads C4 from HuggingFace by default, or pass `--dataset-path` for a local copy

##### SETUP (Kubernetes)

**1. Container Image**

The worker pods need a container image with all dependencies. Build from an existing
Monarch image:

    podman run -d --name ft-build <base-monarch-image> sleep infinity
    podman exec ft-build pip install --break-system-packages torchmonarch==0.4.1
    podman commit ft-build <registry>/monarch:ft-v1
    podman push <registry>/monarch:ft-v1

**2. Controller Pod**

Deploy the controller pod and RBAC resources:

    kubectl apply -f launcher.yaml

The `launcher.yaml` creates:
- Namespace
- ServiceAccount with RBAC for creating MonarchMesh CRDs and listing pods
- Headless Service for DNS resolution (so worker pods can reach the lighthouse)
- Controller pod (runs the orchestration script)

**3. Clone Training Code**

On the controller pod:

    kubectl exec -it <controller-pod> -n <namespace> -- bash
    git clone https://github.com/HosseinKaviani-H/torchft.git /workspace/torchft
    cd /workspace/torchft/examples/monarch

##### USAGE

**SLURM:**

    python train_distributed.py --help

    # Basic usage with 2 replicas, each with 1 node and 8 GPUs:
    python train_distributed.py

    # Custom configuration:
    python train_distributed.py --replica-count 3 --gpu-per-node 8 \
        --host-per-replica 2 --training-steps 100

    # With remote TorchFT lighthouse:
    python train_distributed.py --remote-lighthouse

    # With failure injection testing:
    python train_distributed.py --training-steps 8000 --with-failures

**Kubernetes:**

    python train_distributed_k8s.py --help

    # Provisioning mode (Monarch operator creates GPU pods):
    python train_distributed_k8s.py --namespace monarch-tests \
        --image <registry>/monarch:ft-v1 \
        --replica-count 2 --gpus-per-host 8 --training-steps 10000

    # With failure injection testing:
    python train_distributed_k8s.py --namespace monarch-tests \
        --image <registry>/monarch:ft-v1 \
        --replica-count 2 --gpus-per-host 8 --training-steps 10000 \
        --with-failures

    # With custom dataset and tokenizer:
    python train_distributed_k8s.py --namespace monarch-tests \
        --image <registry>/monarch:ft-v1 \
        --dataset-path /data/c4_test \
        --tokenizer-path /data/tokenizer

    # With timeout for pod readiness:
    python train_distributed_k8s.py --namespace monarch-tests \
        --image <registry>/monarch:ft-v1 --timeout 300

##### ARCHITECTURE

```
Controller Pod
├── Root actor (OrchestrationManager, LighthouseServer)
├── ReplicaActor 0 (local proc_mesh, __supervise__ boundary)
└── ReplicaActor 1 (local proc_mesh, __supervise__ boundary)

Worker Pod replica0 [HostMesh 0]
└── trainers_proc_mesh (8 GPU processes)
    ├── TrainingActor × 8
    └── FailureActor × 8 (test only, --with-failures)

Worker Pod replica1 [HostMesh 1]
└── trainers_proc_mesh (8 GPU processes)
    ├── TrainingActor × 8
    └── FailureActor × 8
```

##### KEY COMPONENTS
- **LighthouseServer**: TorchFT coordination server for quorum-based fault tolerance (runs in-process on controller for K8s, as a remote actor for SLURM)
- **TrainingActor**: Individual trainer processes, one per GPU. Runs `FaultTolerantTrainer.train()`.
- **ReplicaActor**: Supervision boundary. Owns the trainers ProcMesh, handles child failures via `__supervise__`, implements inner retry loop (K8s only).
- **OrchestrationManager**: Top-level orchestration — creates K8s jobs, manages outer retry loop, runs failure injector.
- **MonarchSlurm / MonarchKubernetes**: Scheduler abstraction for job lifecycle (create/kill pods or SLURM allocations).
- **FailureController**: Optional (`--with-failures`), periodically injects SEGFAULT, KILL_PROC, COMMS, or DEADLOCK failures into random ranks.

##### FAILURE RECOVERY

**Two-level retry:**
- **Inner retry (K8s only):** `ReplicaActor` catches child failures via `__supervise__`, calls `spawn_procs()` on the same HostMesh (same pod, no restart). Up to `PROC_ATTEMPTS` (4) retries.
- **Outer retry:** Controller tears down the ReplicaActor entirely and creates a fresh one. Every `PROC_ATTEMPTS` failures, kills and recreates the K8s pod. Gives up after `MAX_ATTEMPT` (16).

**Checkpoint recovery:** New trainers load the latest checkpoint from a healthy replica via TorchFT's HTTP-based in-memory checkpoint transfer (~1.3s). No disk I/O required.

**SLURM-specific:** No inner retry or `__supervise__`. Every failure goes through the outer loop. Process crashes may kill the entire SLURM job, requiring all replicas to be recreated.

**Kubernetes-specific:** Pod crashes are isolated — only the failed replica's job is recreated while healthy replicas continue training uninterrupted.

##### KNOWN WORKAROUNDS (K8s)

These workarounds are applied in `train_distributed_k8s.py` due to open Monarch issues:

1. **`@endpoint(instrument=False)`** — Disables PySpan telemetry on all endpoints to avoid a Rust thread-safety panic.
2. **Manual proc_mesh stop in `__supervise__`** — `call()` does not raise when a child dies, so we stop the ProcMesh manually to unblock recovery.
3. **`call_soon_threadsafe` in `__supervise__`** — `__supervise__` runs on a tokio thread with no asyncio event loop, requiring cross-thread scheduling.

See `MONARCH_GAPS.md` for details and tracked GitHub issues.

##### OUTPUT
- Training logs streamed from all GPU processes to the controller via `stream_to_client=True`
- TensorBoard metrics enabled by default
- Monarch internal logs written to `/tmp/root/monarch_log.log` on the controller pod

##### CLEANUP
All jobs (SLURM or Kubernetes) are automatically terminated at script exit via atexit handlers.
The `__getstate__`/`__setstate__` pickle protection ensures that remote copies of the
scheduler do not accidentally kill jobs when garbage collected.
