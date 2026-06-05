### Monarch-TorchFT-TorchTitan Distributed Training Orchestrator

#### Overview
This directory contains scripts for orchestrating fault-tolerant distributed training using
TorchTitan and Monarch. Two scheduler backends are supported:

- **`train_distributed.py`** — SLURM-based (bare-metal / HPC clusters)
- **`train_distributed_k8s.py`** — Kubernetes with HostMesh reuse (fault-tolerant recovery)

Both scripts manage multiple training replicas with automatic failure recovery and
TorchFT lighthouse coordination.

##### PREREQUISITES

**Common:**
- Python 3.12+
- PyTorch with CUDA support
- `torchmonarch` >= 0.6.0.dev (with session cache eviction fix, PR #4067)
- `torchft` (TorchFT fault tolerance library)
- `torchtitan` (training framework)

**SLURM (`train_distributed.py`):**
- Access to a SLURM cluster with GPU nodes
- Munge authentication configured across nodes
- Training dataset (`c4_test`) and tokenizer in script directory

**Kubernetes (`train_distributed_k8s.py`):**
- Access to a Kubernetes cluster with GPU nodes
- Monarch K8s operator >= 0.2.0 installed (`helm install monarch-operator`)
- A controller pod with RBAC permissions and a headless Service for DNS
- A container image with Monarch, TorchFT, and TorchTitan installed
- Tokenizer baked into the image (default: `/opt/torchtitan/tests/assets/tokenizer`)
- Dataset: downloads C4 from HuggingFace by default, or pass `--dataset-path` for a local copy

##### USAGE

**SLURM:**

    python train_distributed.py --replica-count 2 --gpu-per-node 8 --training-steps 10000

    # With failure injection:
    python train_distributed.py --training-steps 10000 --with-failures

**Kubernetes (HostMesh reuse — recommended):**

    python train_distributed_k8s.py --namespace monarch-tests \
        --image <registry>/monarch:<tag> \
        --replica-count 2 --gpus-per-host 8 --training-steps 10000

    # With failure injection:
    python train_distributed_k8s.py --namespace monarch-tests \
        --image <registry>/monarch:<tag> \
        --replica-count 2 --gpus-per-host 8 --training-steps 10000 \
        --with-failures

    # Multi-host replicas (e.g., 2 pods × 8 GPUs per replica):
    python train_distributed_k8s.py --namespace monarch-tests \
        --image <registry>/monarch:<tag> \
        --replica-count 2 --hosts-per-replica 2 --gpus-per-host 8 \
        --training-steps 10000

##### ARCHITECTURE

```
Controller Pod (no GPUs)
├── Lighthouse (TorchFT quorum coordination)
├── ReplicaActor 0 ──→ GPU Pod(s) (8 GPUs each, HostMesh reuse on failure)
└── ReplicaActor 1 ──→ GPU Pod(s) (8 GPUs each, HostMesh reuse on failure)

Communication layers:
  Intra-replica:   NCCL (GPU-to-GPU, FSDP gradient sharding)
  Cross-replica:   TorchFT ManagedProcessGroup (gradient sync between replicas)
  Orchestration:   Monarch actors over TCP (controller ↔ GPU pods)
  Coordination:    TorchFT Lighthouse (quorum, step agreement, checkpoint transfer)
```

##### FAILURE RECOVERY (K8s HostMesh Reuse)

When a GPU process dies:

1. **`__supervise__`** catches the failure, returns `True` (prevents root actor crash)
2. **`call()` raises** → endpoint converts to `TrainerFailure`
3. **Controller catches it** via `except Exception` in `_run_replica`
4. **`_teardown`** stops the ReplicaActor's local proc_mesh → remote processes become orphans
5. **`mesh_orphan_timeout` (10s)** → Monarch's proc_agent kills orphans, frees GPUs
6. **`_spin_up_replica`** → `scheduler.proc_mesh()` → `hm.spawn_procs()` on same pod → training resumes

The healthy replica **never stops training** — it continues solo via TorchFT's quorum.

##### CONFIGURATION

Key settings in `train_distributed_k8s.py`:

```python
configure(mesh_orphan_timeout="10s")   # How fast orphans are killed
PROC_ATTEMPT_DELAY = 15               # Wait before respawn (> orphan timeout)
train_timeout_seconds = 300            # NCCL PG timeout (> process_group_timeout_ms)
process_group_timeout_ms = 60000       # TorchFT abort timer (< train_timeout_seconds)
```

##### KEY COMPONENTS
- **LighthouseServer**: TorchFT coordination for quorum-based fault tolerance
- **TrainingActor**: One per GPU. Runs `FaultTolerantTrainer.train()`
- **ReplicaActor**: Owns the trainers ProcMesh. Catches failures via `__supervise__`, raises `TrainerFailure`
- **OrchestrationManager**: Creates K8s jobs, manages retry loop, runs failure injector
- **MonarchKubernetes**: Scheduler abstraction for K8s job lifecycle
- **FailureController**: Optional (`--with-failures`), injects SEGFAULT/KILL_PROC into random ranks

##### OUTPUT
- Quorum step progress visible via lighthouse logs on the controller
- TensorBoard metrics enabled by default
- Monarch internal logs at `/tmp/root/monarch_log.log` on the controller pod

##### CLEANUP
All K8s jobs are automatically terminated at script exit via atexit handlers.
