# Advanced Fault-Tolerant Training on Kubernetes with Monarch + TorchFT

## Overview

We enabled fault-tolerant distributed training on Kubernetes using the Monarch actor framework, TorchFT, and TorchTitan. Our approach recovers from GPU process failures in **~17 seconds** by reusing the existing K8s pod, while the healthy replica **never stops training**. This compares to **53–115 seconds** for a full K8s pod restart, and even longer in clusters with GPU scheduling contention where pods may wait minutes for available resources.

## The Problem: Hanging Processes on the Broadcast Tree

Monarch uses a tree-based message broadcasting system for efficient O(1) communication within a ProcMesh. When you send a message to 8 GPU processes, it routes through a binary tree — the root forwards to its children, which forward to theirs. This is efficient during normal operation, but creates a critical problem during failures.

When a GPU process dies (hardware fault, SEGFAULT, OOM), it breaks the broadcast tree. Any surviving processes that are children of the dead node become **unreachable** — messages to them must route through the dead node, which can't forward anything.

```
Normal:                          After Process 6 dies:

       P0 (root)                        P0 (root)
      /       \                        /       \
    P1         P2                    P1         P2
   /  \       /  \                  /  \       /  \
  P3  P4    P5   P6               P3  P4    P5   ██ DEAD
                  |                                |
                 P7                               P7 (unreachable)
```

The standard cleanup pattern uses `async with proc_mesh:`, which calls `proc_mesh.stop()` on exit. This sends a stop message through the broadcast tree — but with a dead node, the message never reaches the unreachable processes. `stop()` hangs forever waiting for acknowledgment, freezing the entire training script.

```python
# This hangs when a process in the mesh has died:
async with trainers_proc_mesh:
    await training_actors.start_training.call(...)
# __aexit__ calls proc_mesh.stop() → hangs on broken tree
```

## Our Solution: The Orphan Pattern

Rather than attempting cleanup through the broken broadcast tree, we skip it entirely and let Monarch's garbage collection handle it.

**Key idea:** Stop the *owner* of the processes (the ReplicaActor on the controller pod), not the processes themselves. When the owner dies, the remote GPU processes become orphans with no parent. Monarch's proc_agent detects ownerless processes and kills them after a configurable timeout.

### How It Works

1. **`__supervise__`** catches the child process failure, returns `True` to prevent propagation to the root actor
2. **`call()` raises `SupervisionError`** → the endpoint converts it to `TrainerFailure`
3. **`_run_replica`** catches the exception via `except Exception`
4. **`_teardown`** calls `proc_mesh.stop()` on the ReplicaActor's **local** proc_mesh (on the controller pod — always succeeds since it's not on the broken tree). Remote GPU processes become orphans.
5. **`mesh_orphan_timeout` (10s)** → Monarch's proc_agent kills the orphans, freeing GPUs
6. **`_spin_up_replica`** → `scheduler.proc_mesh()` → `hm.spawn_procs()` creates fresh processes on the **same pod**, same HostMesh. Training resumes.

```python
class ReplicaActor(Actor):
    async def __supervise__(self, failure) -> bool:
        self.failure_occurred = True
        return True  # Don't propagate to root actor

    @endpoint(instrument=False)
    async def start_replica(self) -> None:
        # NO async with — don't attempt cleanup through broken tree
        trainers_proc_mesh = self.scheduler.proc_mesh(...)
        training_actors = trainers_proc_mesh.spawn("training_actors", TrainingActor, ...)
        try:
            await training_actors.start_training.call(lighthouse_address)
        except Exception as e:
            if self.failure_occurred:
                raise TrainerFailure("trainer process died") from e
            raise
```

### Why Not Just Restart the Pod?

Two reasons:

1. **Allocation retention:** In a competitive cluster, GPU nodes are scarce. When you delete a pod, you release the GPU allocation back to K8s. Another job may claim those GPUs before your replacement pod is scheduled. With HostMesh reuse, you never release the allocation — the pod stays alive, and you spawn new processes on it immediately. The GPUs are yours throughout the recovery.

2. **Faster recovery:** Spawning new processes on an existing pod takes **~0.2 seconds**. Getting a new pod from K8s takes **53–115 seconds** (scheduling + container pull + worker startup). In a congested cluster, pod scheduling alone can take minutes.

## Results

We ran 10,000-step fault-tolerant training with 2 replicas × 8 GPUs, injecting process failures (SEGFAULT, KILL_PROC) every 10 minutes.

### Recovery Time Comparison

| Method | Recovery Time | Healthy Replica | GPU Allocation |
|---|---|---|---|
| **Monarch HostMesh reuse** | **17–53s** | Keeps training | Retained |
| K8s pod restart | 53–115s | Keeps training | Released and reacquired |
| Full job restart (no FT) | 90–300s+ | Both replicas idle | Released and reacquired |

### Detailed Timeline (HostMesh Reuse, 15s Delay)

```
t=0s      Process killed (SEGFAULT/KILL_PROC)
t=0.1s    __supervise__ catches failure
t=0.1s    TrainerFailure raised → controller catches it
t=0.2s    ReplicaActor torn down → GPU processes orphaned
t=0-10s   Orphaned processes killed by proc_agent
t=15s     Controller spawns new ReplicaActor
t=15.2s   New processes created on same pod (spawn: 0.2s)
t=17-36s  Recovered replica joins quorum, training resumes
```

During the entire recovery, the healthy replica **never stops training** — it continues solo via TorchFT's quorum mechanism with `min_replicas=1`.

### Training Completion

- 10,000 training steps completed successfully
- Multiple failure injections survived (quorum_id progressed through 6 changes)
- Both replicas finished together
- Zero steps lost — healthy replica continued training during every recovery

## Permanent Fix: Resilient Casting (In Progress)

The Monarch team is working on making the broadcast tree resilient to partial failures. Instead of hanging when a dead node blocks message delivery, the tree will route around unavailable ranks:

> *"The fix for this is to make casting messages resilient to some ranks being unavailable."*
> — Monarch team

Once resilient casting lands:
- `proc_mesh.stop()` will succeed even with dead processes
- `async with proc_mesh:` will work correctly during failures
- The orphan pattern becomes unnecessary — direct cleanup will work
- Recovery time drops further since there's no orphan timeout wait

The orphan pattern is the correct workaround until then and is production-ready today.

## Session Cache Fix (Landed)

During development, we discovered that Monarch's transport layer cached TCP sessions by destination address. After a process death and orphan cleanup, the controller's cached session retained stale sequence numbers. When reconnecting, the worker rejected the stale session with "out-of-sequence" errors, blocking HostMesh reuse entirely.

We filed a [repro script](https://github.com/HosseinKaviani-H/torchft/blob/Monarch_K8s/examples/monarch/repro_stale_session.py) demonstrating the bug. The Monarch team shipped a fix in [PR #4067](https://github.com/meta-pytorch/monarch/pull/4067) — the `DialMailboxRouter` sender cache now evicts entries when the underlying connection is closed, allowing clean reconnection after failures.

## Architecture

```
Controller Pod (no GPUs)
  ├── Lighthouse (TorchFT quorum coordination)
  ├── ReplicaActor 0 ──→ GPU Pod 0 (8 GPUs, HostMesh reuse on failure)
  └── ReplicaActor 1 ──→ GPU Pod 1 (8 GPUs, HostMesh reuse on failure)

Communication layers:
  Intra-replica:   NCCL (GPU-to-GPU on same pod, FSDP gradient sharding)
  Cross-replica:   TorchFT ManagedProcessGroup (gradient sync between replicas)
  Orchestration:   Monarch actors over TCP (controller ↔ GPU pods)
  Coordination:    TorchFT Lighthouse (quorum, step agreement, checkpoint transfer)
```

## Configuration

Key settings for the orphan pattern:

```python
from monarch.config import configure
configure(mesh_orphan_timeout="10s")  # How fast orphans are killed

PROC_ATTEMPT_DELAY = 15  # Seconds to wait before respawn (> orphan timeout)

# TorchFT settings
comm=CommConfig(train_timeout_seconds=300)  # NCCL PG timeout (must be > process_group_timeout_ms)
fault_tolerance=FaultTolerance(
    enable=True,
    group_size=8,                    # GPUs per replica
    process_group="nccl",
    process_group_timeout_ms=60000,  # TorchFT abort timer (must be < train_timeout_seconds)
)
```

## Scripts

| Script | Description |
|---|---|
| `train_k8s_minimal.py` | HostMesh reuse with orphan pattern (**recommended**) |
| `train_k8s_clean.py` | Pod restart comparison (delete + recreate K8s job) |
| `train_k8s_baseline.py` | No-Monarch baseline (torchrun + K8s Jobs) |
| `repro_stale_session.py` | Repro for the session cache bug (fixed in PR #4067) |

## Key Takeaways

1. **HostMesh reuse is 3–6x faster than pod restart** (17s vs 53–115s) and retains GPU allocations in competitive clusters.
2. **The healthy replica never stops** — TorchFT's quorum handles solo training while the failed replica recovers.
3. **The orphan pattern is production-ready** — no Monarch code changes needed, just skip `async with` and let garbage collection clean up.
4. **Resilient casting will eliminate the workaround** — once Monarch ships it, direct cleanup via `proc_mesh.stop()` will work through partial failures.
