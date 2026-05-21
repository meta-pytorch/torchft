# Monarch Broadcast Tree: Failure Recovery Gap

## Step 1: Normal State (O(1) Casting)

Controller sends ONE message to root of tree. Each node forwards to its children. `stop()` cascades the same way — O(1) from sender. All 8 processes reachable through the tree.

```
Controller Pod
  └── ReplicaActor (owns ProcMesh)
        └── GPU Pod — Broadcast Tree:

                     P0 (root)
                    /       \
                  P1         P2
                 /  \       /  \
                P3  P4    P5   P6
                                |
                               P7

            All nodes reachable ✓
```

---

## Step 2: Process 6 Dies (SEGFAULT / KILL_PROC)

Process 6 is killed (HW failure, SEGFAULT, etc). P6 was the parent of P7 in the broadcast tree. P7 is still alive but UNREACHABLE — messages to P7 must route through dead P6.

```
                     P0 (root)
                    /       \
                  P1         P2
                 /  \       /  \
                P3  P4    P5   ██ P6 DEAD ██
                                    |
                                   P7 (alive but
                                       UNREACHABLE)

  stop() message flow:
    Controller → P0 → P1, P2
    P1 → P3, P4    ✓
    P2 → P5         ✓
    P2 → P6         ✗ DEAD — can't deliver
    P6 → P7         ✗ NEVER REACHED

  stop() HANGS waiting for P7 ack
```

---

## Step 3: The Hang — `async with proc_mesh`

The SLURM script does:

```python
async with trainers_proc_mesh:
    await training_actors.start_training.call(...)
```

When `call()` raises (process died), `async with` exits and calls `proc_mesh.stop()` for cleanup.

`stop()` sends through the broadcast tree → hangs because P6 is dead and P7 can't be reached.

The ReplicaActor is STUCK in `__aexit__`.
The controller's `call_one()` hangs.
The retry loop never fires.
The entire script is frozen.

---

## Solution: Skip Cleanup — Orphan Pattern

Rather than calling `proc_mesh.stop()` through the broken broadcast tree, we skip cleanup entirely. The ReplicaActor catches the failure via `__supervise__`, and the controller stops only the ReplicaActor's local proc_mesh on the controller pod — severing the ownership link. The remote training processes become orphans with no owner, and Monarch's orphan detection kills them after ~60s, freeing the GPUs. No Monarch changes needed, no hang. The tradeoff is GPUs are held for ~60s while orphans time out.

```
1. __supervise__ catches failure → returns True (don't propagate)
2. call() raises → endpoint converts to TrainerFailure
3. Controller catches TrainerFailure
4. Controller stops ReplicaActor's LOCAL proc_mesh
   (on controller pod — NOT the GPU pod — always succeeds)
5. Remote training processes lose their owner → ORPHANS
6. Monarch's orphan detection kills them after ~60s
   (mesh_orphan_timeout)
7. GPUs freed
8. Controller spawns new ReplicaActor
9. New ReplicaActor calls scheduler.proc_mesh()
   → hm.spawn_procs() on the same pod

✓ No hang — we never send stop() through broken tree
✓ No Monarch changes needed
✗ Orphans hold GPUs for ~60s (can't spawn until freed)
```

We confirmed this works — the healthy replica trained solo for 490 steps during cleanup.

---

## Blocker: Stale TCP Session on Reconnect

The orphan pattern works for the FIRST failure:
- ✓ `__supervise__` catches it
- ✓ `TrainerFailure` propagated to controller
- ✓ ReplicaActor torn down, trainers orphaned
- ✓ Healthy replica trains solo (490 steps proved)

But after 90s, the RECONNECT fails:

```
New ReplicaActor → scheduler.proc_mesh() → job.state()
  → hm.spawn_procs({gpus: 8})

ERROR: "out-of-sequence message, expected seq 0, got 727"
   or: "Timeout=0..8"
```

**Root cause:** Monarch's transport layer caches TCP sessions by destination address. The first ReplicaActor's session (seq 727 after training) persists in the controller's transport cache. The worker pod reset after orphans died (expects seq 0). Controller sends seq 727 → rejected.

**Repro:** `repro_stale_session.py` (confirmed blocked)

---

## Timeline: What We Proved Works vs What's Blocked

```
t=0s     SEGFAULT injected into replica 1, rank 6
t=0.1s   __supervise__ catches logger, SPMD, training actor failures  ✓
t=0.1s   TrainerFailure raised → controller catches it               ✓
t=0.2s   Controller tears down ReplicaActor (orphans trainers)       ✓
t=0.2s   Controller starts 90s wait for orphan cleanup               ✓
t=0-90s  Replica 0 continues training SOLO (490 steps!)              ✓
t=60s    Orphaned processes die (mesh_orphan_timeout)                 ✓
t=90s    Controller spawns new ReplicaActor                           ✓
t=90s    New ReplicaActor calls scheduler.proc_mesh()                  ✓
t=90s    hm.spawn_procs() → "out-of-sequence" or timeout              ✗ BLOCKED

Everything works EXCEPT the final reconnection.
The pod is alive, GPUs are free, worker is ready.
But Monarch's cached session blocks the new connection.
```
