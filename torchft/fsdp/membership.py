# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host membership for fault tolerant FSDP.

Every trainer process initializes one ``nccl2`` process group over all hosts
(active and spare) with ``enable_reconfigure=True``. A lighthouse quorum
decides which hosts own the ``num_active_hosts`` slots; the process group is
then reconfigured so that slot ``s``, local rank ``l`` becomes global rank
``s * procs_per_host + l``. Survivors keep their slot, so their rank and
their snapshot stay valid across reconfigurations.

Known limitations:

- A spare whose quorum request timed out can leave a stale lighthouse
  entry, so the next quorum may include a process that never reconfigures.
- Once the coordinator has exited, a restarted host retries connecting
  until its node agent gives up.
- If rank 0 dies at gen 0 before publishing ``LATEST_KEY``, a restarted
  host is not known to be one, so the split brain check does not apply.
- Timeouts are measured on the lighthouse clock (quorum creation times).
"""

import dataclasses
import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import Future, wait as wait_futures
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import TypeVar

import torch.distributed as dist

logger = logging.getLogger(__name__)

DONE_KEY = "ftfsdp/done"
"""Set once training ends, to ``DONE_OK`` or ``DONE_FAILED``."""
DONE_OK = "ok"
DONE_FAILED = "failed"
STARTED_KEY = "ftfsdp/started"
LATEST_KEY = "ftfsdp/latest"
"""JSON of the latest reconfigured quorum: gen, seq and slot holder hosts."""

# With a store timeout (https://github.com/pytorch/pytorch/pull/200384) store
# ops raise these instead of hanging and reconnect on the next op.
STORE_ERRORS = (dist.DistNetworkError, dist.DistStoreError)
_MAX_BACKOFF = 10.0
# Minimum seconds between logs of quorums without an assignment.
_LOG_INTERVAL = 30.0

T = TypeVar("T")


class TrainingFinishedError(Exception):
    """Training completed; spares should exit cleanly."""


class RestartRequiredError(Exception):
    """This process cannot continue and must be restarted."""


class EvictedError(RestartRequiredError):
    """This host lost its slot and must restart as a spare."""


class UnrecoverableError(Exception):
    """The training state is lost; restarting processes does not help."""


@dataclass(frozen=True)
class MemberInfo:
    """What a process reports to the lighthouse when joining a quorum."""

    host: str
    local_rank: int
    uid: int
    ident: str
    """Unique per process incarnation."""
    handle: str
    slot: int
    gen: int
    latest_gen: int = -1
    """Gen of ``LATEST_KEY`` when joining, -1 if no quorum was published."""
    restarted: bool = False
    """Holds no slot but its host is a slot holder of ``latest_gen``, i.e.
    the host's previous process is gone."""
    since: float = -1.0
    """Lighthouse time (s) of the first quorum this process joined while
    waiting for an assignment, -1 before that."""
    initialized: bool = False
    """Has run the caller's init collectives, i.e. reached ``attach``. A
    process that adopted a slot but failed to reconfigure has not."""


@dataclass(frozen=True)
class Assignment:
    """Result of slot assignment for one quorum.

    ``ranks`` is ordered by new global rank and is empty if there were not
    enough complete spare hosts to fill every slot or a new slot's successor
    slot has no survivor.
    """

    gen: int
    ranks: tuple[MemberInfo, ...]
    host_slots: dict[str, int]
    evict: frozenset[str]
    new_hosts: frozenset[str]
    seq: int = -1
    """Lighthouse creation time of the quorum in ns. Unique per quorum, unlike
    ``gen`` and the lighthouse quorum id, and increasing across job relaunches.
    Keys per-recovery store data."""

    @property
    def initial(self) -> bool:
        return self.gen == 0


def assign_slots(
    members: list[MemberInfo],
    *,
    num_slots: int,
    procs_per_host: int,
    now: float = 0.0,
    incomplete_timeout: float = math.inf,
) -> Assignment:
    """Deterministically assign hosts to slots.

    The previous gen is the newest of the published one and the slot
    holders' gens; there is none on a fresh start. A host keeps its slot if
    all of its processes are present, agree on the slot, and are at the
    previous gen. A host holding a slot that fails these checks is evicted;
    a host that only misses processes is evicted once its first process has
    waited ``incomplete_timeout`` seconds at lighthouse time ``now``, and
    until then no assignment is made. Free slots go to complete spare hosts
    in host name order.

    Raises ``UnrecoverableError`` if a free slot's successor slot stays
    without a survivor for ``incomplete_timeout``, e.g. after adjacent hosts
    failed, or if hosts claim the same slot.
    """
    by_host: dict[str, list[MemberInfo]] = defaultdict(list)
    for m in members:
        by_host[m.host].append(m)
    prev_gen = max(
        [m.latest_gen for m in members] + [m.gen for m in members if m.slot >= 0],
        default=-1,
    )
    restarted = {
        host
        for host, ms in by_host.items()
        if all(m.restarted and m.latest_gen == prev_gen for m in ms)
    }

    def complete(ms: list[MemberInfo]) -> bool:
        return sorted(m.local_rank for m in ms) == list(range(procs_per_host))

    def waited(ms: list[MemberInfo]) -> float:
        return now - min(now if m.since < 0 else m.since for m in ms)

    kept: dict[str, int] = {}
    evict: set[str] = set()
    candidates: list[str] = []
    waiting = False
    for host, ms in sorted(by_host.items()):
        if all(m.slot < 0 for m in ms):
            if complete(ms):
                candidates.append(host)
            continue
        slots = {m.slot for m in ms}
        gens = {m.gen for m in ms}
        if len(slots) != 1 or gens != {prev_gen}:
            evict.add(host)
            continue
        if not complete(ms):
            # Its other processes may still be detecting the failure.
            if waited(ms) < incomplete_timeout:
                waiting = True
            else:
                evict.add(host)
            continue
        (slot,) = slots
        if slot >= num_slots:
            raise UnrecoverableError(f"host {host} claims slot {slot} >= {num_slots}")
        kept[host] = slot

    owners: dict[int, str] = {}
    for host, slot in kept.items():
        if slot in owners:
            raise UnrecoverableError(
                f"hosts {owners[slot]} and {host} both claim slot {slot}"
            )
        owners[slot] = host

    free = [s for s in range(num_slots) if s not in owners]
    # A new host restores from the replica held by the next slot. Without a
    # surviving successor the shard is lost, which usually means a survivor
    # has not detected the failure yet, so wait for it. Once every slot
    # holder of the previous gen restarted, e.g. after a job relaunch, none
    # can survive; the restore then picks a fresh start or fails.
    missing_successor = (
        prev_gen >= 0
        and len(restarted) < num_slots
        and any((s + 1) % num_slots not in owners for s in free)
    )
    # Give up once survivors waited as long as for an incomplete host.
    survivors = [m for m in members if m.host in kept or m.host in restarted]
    if (
        missing_successor
        and not waiting
        and survivors
        and waited(survivors) >= incomplete_timeout
    ):
        lost = [s for s in free if (s + 1) % num_slots not in owners]
        raise UnrecoverableError(f"no surviving successor for slots {lost}")
    if waiting or missing_successor or len(candidates) < len(free):
        return Assignment(
            gen=prev_gen,
            ranks=(),
            host_slots={},
            evict=frozenset(evict),
            new_hosts=frozenset(),
        )
    new_hosts = candidates[: len(free)]
    for slot, host in zip(free, new_hosts):
        owners[slot] = host

    ranks: list[MemberInfo] = []
    for slot in range(num_slots):
        ms = sorted(by_host[owners[slot]], key=lambda m: m.local_rank)
        ranks.extend(ms)
    return Assignment(
        gen=prev_gen + 1,
        ranks=tuple(ranks),
        host_slots={host: slot for slot, host in owners.items()},
        evict=frozenset(evict),
        new_hosts=frozenset(new_hosts),
    )


def reconfigure_uuid(run_id: str, seq: int) -> int:
    digest = hashlib.sha256(f"{run_id}/{seq}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _quorum_seq(quorum) -> int:
    return quorum.created.seconds * 10**9 + quorum.created.nanos


def _backoff(delay: float) -> float:
    return min(max(delay * 2, 0.5), _MAX_BACKOFF)


def retry_store(fn: Callable[[], T], what: str) -> T:
    """Run ``fn``, retrying transient store errors with backoff. Waits can
    last hours on spares, so this never gives up."""
    delay = 0.2
    while True:
        try:
            return fn()
        except STORE_ERRORS as e:
            logger.warning(f"store {what} failed, retrying in {delay:.1f}s: {e}")
            time.sleep(delay)
            delay = min(delay * 2, _MAX_BACKOFF)


class Membership:
    """Joins lighthouse quorums and reconfigures the default process group."""

    def __init__(
        self,
        *,
        store: dist.Store,
        pg_store: dist.Store,
        lighthouse_addr: str,
        host: str,
        host_index: int,
        local_rank: int,
        num_hosts: int,
        num_slots: int,
        procs_per_host: int,
        run_id: str,
        pg_timeout: timedelta,
        quorum_timeout: timedelta,
        spare_quorum_timeout: timedelta,
        incomplete_timeout: timedelta,
        heartbeat_interval: float,
    ) -> None:
        from torchft._torchft import LighthouseClient

        if not 0 <= host_index < num_hosts:
            raise ValueError(f"host index {host_index} not in [0, {num_hosts})")
        if not 0 <= local_rank < procs_per_host:
            raise ValueError(f"local rank {local_rank} not in [0, {procs_per_host})")
        self.store = store
        self.host = host
        self.local_rank = local_rank
        self.num_slots = num_slots
        self.procs_per_host = procs_per_host
        self.run_id = run_id
        self.pg_timeout = pg_timeout
        self.quorum_timeout = quorum_timeout
        self.spare_quorum_timeout = spare_quorum_timeout
        self.incomplete_timeout = incomplete_timeout
        self.uid = host_index * procs_per_host + local_rank
        self.replica_id = f"{host}/{local_rank}"
        self.ident = f"{self.replica_id}/{uuid.uuid4().hex[:12]}"
        self.slot = -1
        self.gen = -1
        self.seq = -1
        # Set by FaultTolerance.attach.
        self.initialized = False

        # TCPStore clients serialize ops, so a PG thread blocked on the store
        # after an abort would stall membership. ``pg_store`` is a separate
        # client.
        dist.init_process_group(
            "nccl2",
            store=dist.PrefixStore("ftfsdp/pg", pg_store),
            rank=self.uid,
            world_size=num_hosts * procs_per_host,
            enable_reconfigure=True,
            timeout=pg_timeout,
        )
        self.client = LighthouseClient(lighthouse_addr, timedelta(seconds=60))
        self._stop = threading.Event()
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_interval,),
            name="ftfsdp-heartbeat",
            daemon=True,
        )
        self._heartbeat.start()

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.client.heartbeat(self.replica_id)
            except Exception as e:
                logger.warning(f"lighthouse heartbeat failed: {e}")

    def close(self) -> None:
        self._stop.set()

    def recovery_pending(self) -> bool:
        """Whether the current recovery should be abandoned: a process of the
        current quorum requested recovery, or training ended."""
        return self.store.check([_recovery_key(self.seq)]) or self.store.check(
            [DONE_KEY]
        )

    def _latest(self) -> dict | None:
        if not self.store.check([LATEST_KEY]):
            return None
        return json.loads(self.store.get(LATEST_KEY))

    def _check_done(self) -> None:
        """Raise once training finished or failed elsewhere."""
        if not self.store.check([DONE_KEY]):
            return
        if self.store.get(DONE_KEY).decode() == DONE_FAILED:
            raise UnrecoverableError("training failed on another process")
        raise TrainingFinishedError()

    def _recovery_requested(self, latest: dict | None) -> bool:
        """Whether a quorum is wanted now.

        Spares only join once a slot holder of the latest quorum asks for
        recovery. A spare that sits in the lighthouse queue would satisfy
        min_replicas the moment the first survivor joins and form a quorum
        without the remaining survivors.
        """
        if self.slot >= 0 or latest is None:
            return True
        key = _recovery_key(latest["seq"])
        if self.host in latest["hosts"]:
            # This host's previous process held a slot and is gone, e.g. after
            # a restart or a job relaunch, so request recovery for it.
            self.store.set(key, "1")
            return True
        return self.store.check([key])

    def _wait_for_published(self, seq: int) -> None:
        """Wait until quorum ``seq`` or a later one is published, its
        reconfigure failed, or it has timed out.

        A spare left out of a quorum would otherwise see the previous
        quorum's recovery key and queue a new request. The lighthouse keeps
        that request after the spare stops waiting on it, so a later quorum
        includes a process that never reconfigures.
        """
        deadline = time.monotonic() + self.pg_timeout.total_seconds()
        while time.monotonic() < deadline:
            try:
                # A failed reconfigure requests recovery for ``seq``.
                if self.store.check([DONE_KEY]) or self.store.check(
                    [_recovery_key(seq)]
                ):
                    return
                latest = self._latest()
                if latest is not None and latest["seq"] >= seq:
                    return
            except STORE_ERRORS as e:
                logger.warning(f"store poll failed, retrying: {e}")
            time.sleep(0.2)

    def _publish_latest(self, latest: dict) -> None:
        """Set ``LATEST_KEY`` unless it holds a newer quorum, e.g. when a
        stalled rank 0 of an old quorum publishes late."""
        value = json.dumps(latest)
        expected = ""
        while True:
            cur = self.store.compare_set(LATEST_KEY, expected, value).decode()
            if cur == value or json.loads(cur)["seq"] >= latest["seq"]:
                return
            expected = cur

    def _quorum(self, data: dict, timeout: timedelta):
        """Run one quorum request, giving up early once training is done so
        a waiting spare does not outlive the job."""
        fut: Future = Future()

        def run() -> None:
            try:
                fut.set_result(
                    self.client.quorum(
                        replica_id=self.replica_id, timeout=timeout, data=data
                    )
                )
            except BaseException as e:
                fut.set_exception(e)

        # A daemon thread so an abandoned request does not block exit.
        threading.Thread(target=run, name="ftfsdp-quorum", daemon=True).start()
        while True:
            # Not fut.result(timeout): on Python 3.11+ its TimeoutError is
            # the builtin one, which a failed request also raises.
            if wait_futures([fut], timeout=1.0).done:
                return fut.result()
            try:
                self._check_done()
            except STORE_ERRORS as e:
                logger.warning(f"store poll failed, retrying: {e}")

    def next_assignment(self) -> Assignment:
        """Block until this process is assigned a slot and the process group
        has been reconfigured to the new membership.

        Raises ``TrainingFinishedError`` once training is done,
        ``UnrecoverableError`` once it failed and ``EvictedError`` if this
        host lost its slot.
        """
        if self.slot >= 0:
            self._request_recovery(self.seq)
        since = -1.0
        delay = 0.0
        last_log = -math.inf
        while True:
            retry_store(self._check_done, "poll")
            latest = retry_store(self._latest, "get")
            if not retry_store(lambda: self._recovery_requested(latest), "poll"):
                time.sleep(0.2)
                continue
            latest_gen = -1 if latest is None else latest["gen"]
            info = MemberInfo(
                host=self.host,
                local_rank=self.local_rank,
                uid=self.uid,
                ident=self.ident,
                handle=dist._get_reconfigure_handle(),
                slot=self.slot,
                gen=self.gen,
                latest_gen=latest_gen,
                restarted=(
                    self.slot < 0
                    and latest is not None
                    and self.host in latest["hosts"]
                ),
                since=since,
                initialized=self.initialized,
            )
            # The lighthouse keeps a request after the client gives up on it,
            # so a quorum can include a process that never sees the result.
            # Spares therefore wait in a single long request.
            timeout = (
                self.quorum_timeout if self.slot >= 0 else self.spare_quorum_timeout
            )
            try:
                quorum = self._quorum(asdict(info), timeout)
            except (TrainingFinishedError, UnrecoverableError):
                raise
            except Exception as e:
                # E.g. the lighthouse refuses connections or timed out.
                delay = _backoff(delay)
                logger.info(f"quorum request failed, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
                continue
            members = [MemberInfo(**p.data) for p in quorum.participants]
            seq = _quorum_seq(quorum)
            if since < 0:
                since = seq / 1e9
            assignment = assign_slots(
                members,
                num_slots=self.num_slots,
                procs_per_host=self.procs_per_host,
                now=seq / 1e9,
                incomplete_timeout=self.incomplete_timeout.total_seconds(),
            )
            now = time.monotonic()
            if self.local_rank == 0 and (
                assignment.ranks or now - last_log >= _LOG_INTERVAL
            ):
                last_log = now
                procs = Counter(m.host for m in members)
                incomplete = sorted(
                    h for h, n in procs.items() if n != self.procs_per_host
                )
                logger.info(
                    f"quorum {quorum.quorum_id}: {len(members)} procs on "
                    f"{len(procs)} hosts, incomplete={incomplete}, "
                    f"assigned={bool(assignment.ranks)} gen {assignment.gen}, "
                    f"new_hosts={sorted(assignment.new_hosts)} "
                    f"evict={sorted(assignment.evict)}"
                )
            if self.host in assignment.evict:
                raise EvictedError(
                    f"host {self.host} evicted at quorum {quorum.quorum_id}"
                )
            if self.host not in assignment.host_slots:
                if assignment.ranks:
                    self._wait_for_published(seq)
                    # The wait is not time spent waiting for a quorum.
                    since = -1.0
                    continue
                # E.g. waiting for an incomplete host or a successor. The
                # lighthouse re-forms the quorum at once, so back off.
                delay = _backoff(delay)
                time.sleep(delay)
                continue
            assignment = dataclasses.replace(assignment, seq=seq)
            # Adopted before reconfiguring: every process derives the same
            # assignment, and peers that reconfigured publish it, so a
            # process whose reconfigure fails must not fall behind.
            self.slot = assignment.host_slots[self.host]
            self.gen = assignment.gen
            self.seq = seq
            reconf_id = reconfigure_uuid(self.run_id, seq)
            start = time.perf_counter()
            try:
                dist._reconfigure(
                    reconf_id,
                    [m.handle for m in assignment.ranks],
                    timeout=self.pg_timeout,
                ).wait()
            except Exception as e:
                logger.warning(f"reconfigure failed, rejoining quorum: {e}")
                # Peers that reconfigured and spares left out wait on this
                # key, not a timeout.
                self._request_recovery(seq)
                since = -1.0
                continue
            if dist.get_rank() == 0:
                published = {
                    "gen": self.gen,
                    "seq": seq,
                    "hosts": sorted(assignment.host_slots),
                }
                retry_store(lambda: self._publish_latest(published), "set")
            logger.info(
                f"gen {self.gen}: slot {self.slot} rank {dist.get_rank()}/"
                f"{dist.get_world_size()} new_hosts={sorted(assignment.new_hosts)} "
                f"evict={sorted(assignment.evict)} "
                f"reconfigure {time.perf_counter() - start:.2f}s"
            )
            return assignment

    def _request_recovery(self, seq: int) -> None:
        retry_store(lambda: self.store.set(_recovery_key(seq), "1"), "set")

    def request_recovery_once(self) -> None:
        """Ask the current quorum to recover if this process holds a slot,
        e.g. before it exits. Best effort: a store error is only logged."""
        if self.slot < 0:
            return
        try:
            self.store.set(_recovery_key(self.seq), "1")
        except Exception as e:
            logger.warning(f"requesting recovery failed: {e}")


def _recovery_key(seq: int) -> str:
    return f"ftfsdp/recover/{seq}"
