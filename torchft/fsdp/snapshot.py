# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-memory snapshots of sharded training state.

Each rank keeps ``num_local`` pinned CPU slots holding its parameter and
optimizer shards plus pickled CPU state. After every snapshot step the copy
runs on a side stream from a worker thread, so the training loop only pays
for recording an event. The next optimizer step waits on the copy's event
before mutating the tensors.

Committed slots are replicated with ``torch.distributed._transport`` to the
same local rank on the next host (rank ``r`` -> rank ``(r + G) % W``), which
holds two replica slots. A write invalidates the replica header, writes the
body, then writes a valid header, so a reader never sees a torn snapshot.
"""

import logging
import pickle
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torchft.fsdp.membership import RestartRequiredError

logger = logging.getLogger(__name__)

_MAGIC = 0x46544653
_HEADER_WORDS = 4  # magic, step, meta_len, nbytes
_ALIGN = 64


@dataclass
class _Slot:
    body: torch.Tensor
    step: int = -1
    meta_len: int = 0
    reserved: bool = False


@dataclass
class _Job:
    step: int
    slot: int
    meta: bytes
    ready: torch.cuda.Event
    epoch: int
    replicate: bool = True
    launched: threading.Event = field(default_factory=threading.Event)
    copy_done: torch.cuda.Event | None = None


@dataclass
class _LinkOut:
    peer: str
    seq: int
    """Recovery that built the link."""
    transport: Any
    body_mems: list
    header_src: torch.Tensor
    header_mem: Any
    remote_headers: list
    remote_bodies: list
    broken: bool = False
    """A transfer failed, so the link is rebuilt on the next recovery."""


@dataclass
class _LinkIn:
    peer: str
    seq: int
    transport: Any


def _local(t: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return t._local_tensor if isinstance(t, DTensor) else t


def collect_state_tensors(
    model_parts: list[torch.nn.Module], optimizers: list[torch.optim.Optimizer]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (device tensors, CPU tensors) of parameters, persistent buffers
    and optimizer state in a canonical order shared by all ranks of the same
    slot. Snapshots hold these tensors, so all of them must be updated in
    place (e.g. no ``optimizer.load_state_dict`` afterwards)."""
    device_tensors: list[torch.Tensor] = []
    cpu_tensors: list[torch.Tensor] = []

    def add(t: torch.Tensor) -> None:
        t = _local(t)
        (cpu_tensors if t.device.type == "cpu" else device_tensors).append(t)

    for part in model_parts:
        for p in part.parameters():
            add(p.detach())
        for mod in part.modules():
            for name, b in mod.named_buffers(recurse=False):
                if name not in mod._non_persistent_buffers_set:
                    add(b)
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state[p]
                for key in sorted(state):
                    value = state[key]
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(
                            f"unsupported non-tensor optimizer state {key!r}"
                        )
                    add(value)
    return device_tensors, cpu_tensors


def wait_for_keys(
    store: dist.Store,
    keys: list[str],
    timeout: float,
    abort: Callable[[], bool],
    poll: float = 1.0,
) -> None:
    """Wait for ``keys``, giving up once ``abort()`` returns true.

    Waits on the store server in slices of ``poll`` seconds rather than
    polling ``check``, which costs O(world) ops per rank.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        try:
            store.wait(keys, timedelta(seconds=max(min(poll, remaining), 0.001)))
            return
        except dist.DistStoreError:
            pass
        if abort():
            raise RuntimeError(f"recovery aborted while waiting for {keys}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {keys}")


class _StaleLinkError(Exception):
    """The link being written was replaced during recovery."""


class _TransferAbortedError(RuntimeError):
    """A transfer was abandoned for recovery."""


class SnapshotStalledError(dist.DistError):
    """The snapshot copy did not launch, e.g. because replication is stuck on
    a dead successor. A ``DistError`` so the training loop recovers."""


class TransportPool:
    """NIXL transports created ahead of use.

    Creating a NIXL agent takes about 15 s on a host with many NICs, so a
    process starts creating them before it is assigned a slot and each link
    takes a ready one.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"transport pool size must be >= 1, got {size}")
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="ftfsdp-nixl")
        self._ready: queue.SimpleQueue[Future] = queue.SimpleQueue()
        for _ in range(size):
            self._refill()

    def _refill(self) -> None:
        self._ready.put(self._executor.submit(_new_nixl_transport))

    def get(self):
        fut = self._ready.get()
        self._refill()
        return fut.result()

    def close(self) -> None:
        """Close transports that were created but never taken, without
        waiting for one still being created."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        while True:
            try:
                fut = self._ready.get_nowait()
            except queue.Empty:
                return
            fut.add_done_callback(_close_unused)


def _close_unused(fut: Future) -> None:
    if not fut.cancelled() and fut.exception() is None:
        Snapshotter._close_transport(fut.result())


def _new_nixl_transport():
    # pyrefly: ignore [missing-import]
    from torch.distributed._transport import new_transport

    return new_transport("nixl", "cpu")


def _bootstrap(
    tr,
    store: dist.Store,
    *,
    rank: int,
    peer_rank: int,
    timeout: float,
    abort: Callable[[], bool],
):
    """Exchange agent metadata like ``new_transport_rank``, on an existing
    transport. ``store`` must be unique to the link."""
    try:
        store.set(str(rank), tr.bind(timeout=timeout))
        wait_for_keys(store, [str(peer_rank)], timeout, abort)
        tr.connect(store.get(str(peer_rank)), timeout=timeout)
    except BaseException:
        Snapshotter._close_transport(tr)
        raise
    return tr


class Snapshotter:
    def __init__(
        self,
        tensors: list[torch.Tensor],
        *,
        transports: TransportPool,
        device: torch.device,
        ident: str,
        store: dist.Store,
        num_local: int,
        meta_capacity: int,
        interval: int,
        procs_per_host: int,
        timeout: float,
        comm_failed: Callable[[], bool],
        recovery_pending: Callable[[], bool],
    ) -> None:
        if num_local < 3:
            raise ValueError(f"num_local_snapshots must be >= 3, got {num_local}")
        self.tensors = tensors
        self.transports = transports
        self.device = device
        self.ident = ident
        self.store = store
        self.meta_capacity = meta_capacity
        self.interval = interval
        self.procs_per_host = procs_per_host
        self.timeout = timeout
        self.comm_failed = comm_failed
        self.recovery_pending = recovery_pending

        offsets = []
        offset = meta_capacity
        for t in tensors:
            offset = (offset + _ALIGN - 1) // _ALIGN * _ALIGN
            offsets.append(offset)
            offset += t.numel() * t.element_size()
        self.nbytes = offset
        self._offsets = offsets

        start = time.perf_counter()
        self.slots = [
            _Slot(torch.empty(self.nbytes, dtype=torch.uint8, pin_memory=True))
            for _ in range(num_local)
        ]
        self._views = [self._make_views(s.body) for s in self.slots]
        self.replica_headers = [
            torch.zeros(_HEADER_WORDS, dtype=torch.int64, pin_memory=True)
            for _ in range(2)
        ]
        # Sized for the predecessor, whose FSDP shards may be smaller.
        self.replica_nbytes = self.nbytes
        self.replica_bodies = self._alloc_replicas(self.nbytes)
        logger.info(
            f"snapshot: {len(tensors)} tensors, {self.nbytes / 2**20:.1f} MiB per "
            f"slot, {num_local} local + 2 replica slots allocated in "
            f"{time.perf_counter() - start:.2f}s"
        )

        self.link_out: _LinkOut | None = None
        self.link_in: _LinkIn | None = None

        self.aborted = threading.Event()
        self._epoch = 0
        self._cv = threading.Condition()
        self._queue: list[_Job] = []
        self._busy = False
        self._replicating = False
        self._pending: _Job | None = None
        self._closed = False
        self._stream = torch.cuda.Stream(device=device)
        self._worker = threading.Thread(
            target=self._run, name="ftfsdp-snapshot", daemon=True
        )
        self._worker.start()

    @staticmethod
    def _alloc_replicas(nbytes: int) -> list[torch.Tensor]:
        return [
            torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)
        ]

    def _make_views(self, body: torch.Tensor) -> list[torch.Tensor]:
        views = []
        for t, off in zip(self.tensors, self._offsets):
            n = t.numel() * t.element_size()
            views.append(body[off : off + n].view(t.dtype).view(t.shape))
        return views

    # Training loop side.

    def register_optimizer_hooks(self, optimizers: list[torch.optim.Optimizer]):
        for opt in optimizers:
            opt.register_step_pre_hook(lambda *_: self.wait_copy_launched())

    def wait_copy_launched(self) -> None:
        """Order the current stream after the in-flight snapshot copy so the
        optimizer cannot overwrite tensors that are still being read."""
        job = self._pending
        if job is None:
            return
        # The worker may be stuck writing to a dead successor. This wait is
        # outside any collective, so poll for failures elsewhere.
        deadline = time.monotonic() + self.timeout
        while not job.launched.wait(0.5):
            if self.aborted.is_set() or self.comm_failed() or self.recovery_pending():
                raise SnapshotStalledError(
                    f"snapshot copy for step {job.step} abandoned for recovery"
                )
            if time.monotonic() > deadline:
                raise SnapshotStalledError(
                    f"snapshot copy for step {job.step} not launched"
                )
        self._pending = None
        if job.copy_done is not None:
            torch.cuda.current_stream().wait_event(job.copy_done)

    def capture(self, step: int, meta: bytes, replicate: bool = True) -> None:
        """Snapshot the current state as ``step``. Must be called on the
        training thread after the optimizer step. With ``replicate`` unset
        the worker only commits it locally."""
        if len(meta) > self.meta_capacity:
            raise ValueError(
                f"snapshot metadata is {len(meta)} bytes, larger than "
                f"meta_capacity_bytes={self.meta_capacity}"
            )
        self.wait_copy_launched()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        with self._cv:
            free = [i for i, s in enumerate(self.slots) if not s.reserved]
            slot = min(free, key=lambda i: self.slots[i].step)
            self.slots[slot].reserved = True
            self.slots[slot].step = -1
            job = _Job(
                step=step,
                slot=slot,
                meta=meta,
                ready=ready,
                epoch=self._epoch,
                replicate=replicate,
            )
            self._queue.append(job)
            self._cv.notify_all()
        self._pending = job

    def abort(self) -> None:
        self.aborted.set()

    def pause(self) -> None:
        """Drop queued snapshots and wait for the worker to go idle."""
        self.aborted.set()
        self._pending = None
        with self._cv:
            self._epoch += 1
            for job in self._queue:
                self.slots[job.slot].reserved = False
                job.launched.set()
            self._queue.clear()
            # A write to a dead successor can block until the transport
            # timeout. Do not wait for it: update_links rebuilds its link.
            if not self._cv.wait_for(
                lambda: not self._busy or self._replicating, self.timeout
            ):
                raise TimeoutError("snapshot worker did not go idle")

    def resume(self) -> None:
        self.aborted.clear()

    def flush(self) -> None:
        """Wait for queued snapshots to commit and replicate."""
        with self._cv:
            if not self._cv.wait_for(
                lambda: not self._queue and not self._busy, self.timeout
            ):
                raise TimeoutError("snapshot worker did not drain")
        self._pending = None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            link_out, self.link_out = self.link_out, None
            # An in-flight write may still own the transport. The worker
            # closes it once the write sees the link replaced.
            if self._replicating:
                link_out = None
        link_in, self.link_in = self.link_in, None
        for link in (link_out, link_in):
            if link is not None:
                self._close_transport(link.transport)

    # Worker side.

    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._closed)
                if self._closed:
                    return
                job = self._queue.pop(0)
                self._busy = True
            try:
                self._process(job)
            except _TransferAbortedError as e:
                logger.info(f"snapshot for step {job.step} abandoned: {e}")
                job.launched.set()
            except Exception:
                logger.exception(f"snapshot for step {job.step} failed")
                job.launched.set()
            finally:
                with self._cv:
                    self.slots[job.slot].reserved = False
                    self._busy = False
                    self._cv.notify_all()

    def _process(self, job: _Job) -> None:
        slot = self.slots[job.slot]
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(job.ready)
            torch._foreach_copy_(self._views[job.slot], self.tensors, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self._stream)
        job.copy_done = done
        job.launched.set()
        n = len(job.meta)
        slot.body[:n].copy_(torch.frombuffer(bytearray(job.meta), dtype=torch.uint8))
        done.synchronize()
        if self.aborted.is_set() or job.epoch != self._epoch or self.comm_failed():
            return
        with self._cv:
            slot.step = job.step
            slot.meta_len = n
        if not job.replicate:
            return
        with self._cv:
            link = self.link_out
            if link is None:
                return
            self._replicating = True
            # pause() waits for this.
            self._cv.notify_all()
        try:
            self._replicate(link, job.slot, self._worker_abort)
        except _StaleLinkError:
            pass
        finally:
            with self._cv:
                self._replicating = False
                stale = link is not self.link_out
        if stale:
            # Replaced by update_links during recovery; nobody else owns it.
            # An abandoned write may hold it until the transport times out.
            threading.Thread(
                target=self._close_transport, args=(link.transport,), daemon=True
            ).start()

    def _replica_index(self, step: int) -> int:
        return (step // self.interval) % 2

    def _worker_abort(self) -> bool:
        return self.aborted.is_set() or self.comm_failed()

    def _recovery_abort(self) -> bool:
        # Snapshots stay aborted until recovery finishes, so only another
        # recovery or a comm failure abandons a recovery transfer.
        return self.recovery_pending() or self.comm_failed()

    def _transfer(
        self, link: _LinkOut, op: Callable, local, remote, abort: Callable[[], bool]
    ) -> None:
        """Run ``op`` (the transport's read or write) and wait, giving up once
        recovery replaces ``link`` or ``abort()`` returns true. A transfer
        with a dead successor may not fail before the transport timeout, and
        the next snapshot copy or recovery must not wait for it."""
        deadline = time.monotonic() + self.timeout
        try:
            work = op(local, remote, async_op=True, timeout=self.timeout)
            while not work.is_completed():
                if link is not self.link_out:
                    raise _StaleLinkError()
                if abort():
                    # The abandoned transfer may still own the transport.
                    raise _TransferAbortedError("snapshot transfer aborted")
                if time.monotonic() > deadline:
                    raise TimeoutError("snapshot transfer timed out")
                time.sleep(0.001)
            work.wait()
        except _StaleLinkError:
            raise
        except BaseException:
            link.broken = True
            raise

    def _write_header(
        self,
        link: _LinkOut,
        index: int,
        words: list[int],
        abort: Callable[[], bool],
    ) -> None:
        link.header_src.copy_(torch.tensor(words, dtype=torch.int64))
        self._transfer(
            link,
            link.transport.write,
            link.header_mem.to_view(),
            link.remote_headers[index],
            abort,
        )

    def _replicate(
        self, link: _LinkOut, slot_index: int, abort: Callable[[], bool]
    ) -> None:
        slot = self.slots[slot_index]
        index = self._replica_index(slot.step)
        self._write_header(link, index, [0, 0, 0, 0], abort)
        self._transfer(
            link,
            link.transport.write,
            link.body_mems[slot_index].to_view(),
            link.remote_bodies[index],
            abort,
        )
        self._write_header(
            link, index, [_MAGIC, slot.step, slot.meta_len, self.nbytes], abort
        )

    # Recovery side. Callers must pause() first.

    def committed_steps(self) -> list[int]:
        return sorted(s.step for s in self.slots if s.step >= 0)

    def replica_steps(self) -> dict[int, tuple[int, int]]:
        """Valid replicas held for the predecessor: step -> (index, meta_len)."""
        out = {}
        for i, h in enumerate(self.replica_headers):
            magic, step, meta_len, nbytes = h.tolist()
            if magic == _MAGIC and nbytes == self.replica_nbytes:
                out[step] = (i, meta_len)
        return out

    def fetch_replica(self, step: int, index: int, meta_len: int) -> None:
        """Read the successor's replica of this rank into a local slot."""
        link = self.link_out
        assert link is not None
        slot_index = min(range(len(self.slots)), key=lambda i: self.slots[i].step)
        slot = self.slots[slot_index]
        slot.step = -1
        try:
            self._transfer(
                link,
                link.transport.read,
                link.body_mems[slot_index].to_mutable_view(),
                link.remote_bodies[index],
                self._recovery_abort,
            )
        except BaseException as e:
            # An abandoned read may still fill the slot until its transport
            # is closed, and a later capture or fetch may reuse the slot.
            with self._cv:
                self.link_out = None
            if not self._close_transport(link.transport):
                raise RestartRequiredError(
                    "abandoned replica read may still overwrite a local slot"
                ) from e
            raise
        slot.step = step
        slot.meta_len = meta_len

    def restore(self, step: int) -> bytes:
        """Copy the slot holding ``step`` into the training tensors and return
        its metadata. Newer local slots are invalidated."""
        matches = [i for i, s in enumerate(self.slots) if s.step == step]
        if not matches:
            raise RuntimeError(
                f"no local snapshot for step {step}: {self.committed_steps()}"
            )
        slot = self.slots[matches[0]]
        torch._foreach_copy_(self.tensors, self._views[matches[0]], non_blocking=True)
        torch.cuda.synchronize()
        for s in self.slots:
            if s.step > step:
                s.step = -1
        return bytes(slot.body[: slot.meta_len].numpy())

    def replicate_now(self, step: int, *, fetched: bool = False) -> None:
        """Synchronously replicate ``step`` and invalidate the other replica.

        With ``fetched`` the successor's replica of ``step`` was just read,
        so it is kept rather than rewritten: rewriting invalidates the only
        copy of this rank's state until the write completes."""
        link = self.link_out
        if link is None:
            return
        matches = [i for i, s in enumerate(self.slots) if s.step == step]
        if not matches:
            # E.g. the worker skipped the commit after a comm failure.
            raise RuntimeError(f"step {step} not committed")
        if not fetched:
            self._replicate(link, matches[0], self._recovery_abort)
        # Only _replicate writes a valid replica, at _replica_index(step), so
        # a fetched replica of step is held there too.
        other = 1 - self._replica_index(step)
        self._write_header(link, other, [0, 0, 0, 0], self._recovery_abort)

    # Links.

    def update_links(
        self, *, rank: int, ident_of_rank: list[str], seq: int, timeout: float
    ) -> None:
        """Connect to the successor (writes) and predecessor (receives).

        A link whose peer is unchanged is kept if both ends still hold it
        from the same recovery and no transfer on it failed or was in flight,
        which the ends agree on through the store. Memory registration (the
        slow part) is then skipped. Other links are rebuilt on a fresh
        transport because an abandoned write may still own the old one.
        Callers must pause() first. Replica contents are kept."""
        world = len(ident_of_rank)
        g = self.procs_per_host
        succ = pred = None
        if world > g:
            succ = ident_of_rank[(rank + g) % world]
            pred = ident_of_rank[(rank - g) % world]
        with self._cv:
            old_out = self.link_out
            # An in-flight write was abandoned by pause() and may still own
            # the transport, so its link is rebuilt. The worker closes it.
            busy_out = self._replicating
            usable_out = (
                old_out is not None
                and old_out.peer == succ
                and not busy_out
                and not old_out.broken
            )
            self.link_out = None
        old_in, self.link_in = self.link_in, None
        usable_in = old_in is not None and old_in.peer == pred
        if old_out is not None and not usable_out and not busy_out:
            self._close_transport(old_out.transport)
        if old_in is not None and not usable_in:
            self._close_transport(old_in.transport)
        if succ is None or pred is None:
            return
        start = time.perf_counter()
        with ThreadPoolExecutor(2) as pool:
            fut_out = pool.submit(
                self._link,
                f"{self.ident}->{succ}",
                0,
                old_out if usable_out else None,
                lambda store: self._connect_out(store, succ, seq, timeout),
                seq,
                timeout,
            )
            fut_in = pool.submit(
                self._link,
                f"{pred}->{self.ident}",
                1,
                old_in if usable_in else None,
                lambda store: self._connect_in(store, pred, seq, timeout),
                seq,
                timeout,
            )
        # Store every link that connected; the next recovery's agreement
        # rebuilds it if the other end did not.
        error = fut_out.exception() or fut_in.exception()
        link_out = None if fut_out.exception() else fut_out.result()
        link_in = None if fut_in.exception() else fut_in.result()
        with self._cv:
            self.link_in = link_in
            self.link_out = link_out
        if error is not None:
            raise error
        logger.info(
            f"snapshot links: -> {succ} (kept={link_out is old_out}), <- {pred} "
            f"(kept={link_in is old_in}) in {time.perf_counter() - start:.2f}s"
        )

    def _link(
        self,
        name: str,
        rank: int,
        old: Any,
        connect: Callable[[dist.Store], Any],
        seq: int,
        timeout: float,
    ) -> Any:
        """Keep ``old`` if the other end agrees, else connect a new link.
        Rank 0 is the writing end."""
        # Each link uses its own client: a blocking wait on a shared TCPStore
        # client stalls the other link's sets and deadlocks the ring.
        store = dist.PrefixStore(f"ftfsdp/link/{seq}/{name}/", self.store.clone())
        mine = "-" if old is None else str(old.seq)
        theirs = None
        try:
            store.set(f"keep/{rank}", mine)
            wait_for_keys(store, [f"keep/{1 - rank}"], timeout, self.recovery_pending)
            theirs = store.get(f"keep/{1 - rank}").decode()
        finally:
            if old is not None and theirs != mine:
                self._close_transport(old.transport)
        if old is not None and theirs == mine:
            return old
        return connect(store)

    def _connect_in(
        self, store: dist.Store, pred: str, seq: int, timeout: float
    ) -> _LinkIn:
        start = time.perf_counter()
        tr = _bootstrap(
            self.transports.get(),
            store,
            rank=1,
            peer_rank=0,
            timeout=timeout,
            abort=self.recovery_pending,
        )
        bootstrap_s = time.perf_counter() - start
        wait_for_keys(store, ["nbytes"], timeout, self.recovery_pending)
        peer_s = time.perf_counter() - start - bootstrap_s
        nbytes = int(store.get("nbytes"))
        if nbytes != self.replica_nbytes:
            logger.warning(
                f"predecessor {pred} snapshot is {nbytes} bytes, not "
                f"{self.replica_nbytes}; reallocating replicas"
            )
            for h in self.replica_headers:
                h.zero_()
            self.replica_bodies = self._alloc_replicas(nbytes)
            self.replica_nbytes = nbytes
        headers = [tr.register_memory(t) for t in self.replica_headers]
        bodies = [tr.register_memory(t) for t in self.replica_bodies]
        register_s = time.perf_counter() - start - bootstrap_s - peer_s
        logger.info(
            f"snapshot link <- {pred}: bootstrap {bootstrap_s:.2f}s, wait peer "
            f"{peer_s:.2f}s, alloc+register {register_s:.2f}s"
        )
        store.set(
            "replica",
            # pyrefly: ignore [bad-argument-type]
            pickle.dumps(
                (
                    [m.to_remote_buffer().serialize() for m in headers],
                    [m.to_remote_buffer().serialize() for m in bodies],
                )
            ),
        )
        return _LinkIn(peer=pred, seq=seq, transport=tr)

    def _connect_out(
        self, store: dist.Store, succ: str, seq: int, timeout: float
    ) -> _LinkOut:
        # pyrefly: ignore [missing-import]
        from torch.distributed._transport.nixl._memory import NIXLRemoteBuffer

        start = time.perf_counter()
        tr = _bootstrap(
            self.transports.get(),
            store,
            rank=0,
            peer_rank=1,
            timeout=timeout,
            abort=self.recovery_pending,
        )
        bootstrap_s = time.perf_counter() - start
        body_mems = [tr.register_memory(s.body) for s in self.slots]
        header_src = torch.zeros(_HEADER_WORDS, dtype=torch.int64, pin_memory=True)
        header_mem = tr.register_memory(header_src)
        register_s = time.perf_counter() - start - bootstrap_s
        store.set("nbytes", str(self.nbytes))
        wait_for_keys(store, ["replica"], timeout, self.recovery_pending)
        headers, bodies = pickle.loads(store.get("replica"))
        logger.info(
            f"snapshot link -> {succ}: bootstrap {bootstrap_s:.2f}s, register "
            f"{register_s:.2f}s, wait peer "
            f"{time.perf_counter() - start - bootstrap_s - register_s:.2f}s"
        )
        return _LinkOut(
            peer=succ,
            seq=seq,
            transport=tr,
            body_mems=body_mems,
            header_src=header_src,
            header_mem=header_mem,
            remote_headers=[NIXLRemoteBuffer.deserialize(b) for b in headers],
            remote_bodies=[NIXLRemoteBuffer.deserialize(b) for b in bodies],
        )

    @staticmethod
    def _close_transport(tr) -> bool:
        """Close ``tr`` and return whether it succeeded.

        NIXL keeps the registrations if work is still pending, so an
        abandoned transfer can still land. A late write is harmless: the
        successor closes its receiving transport, which has no pending work,
        before a new link can write, so the write cannot reach a reused
        replica. A late read lands in a local slot, so ``fetch_replica``
        requires a restart if the close fails.
        """
        try:
            tr.close(timeout=5.0)
        except Exception as e:
            logger.warning(f"closing snapshot transport failed: {e}")
            return False
        return True
