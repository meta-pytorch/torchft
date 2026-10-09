# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fault tolerance for an FSDP training loop.

Training runs plain FSDP over the active hosts. After each step the sharded
parameters and optimizer state are copied asynchronously to pinned CPU memory
and replicated to the next host. When a host fails, survivors catch the
communication error, the lighthouse quorum assigns a hot spare to the failed
slot, the ``nccl2`` process group is reconfigured in place, and every rank
restores the newest step that all ranks hold (the spare reads its shard from
its successor's replica).

Usage::

    # Before building the model: spares block here until they get a slot
    # and exit once training ends.
    ft = FaultTolerance(
        FTFSDPConfig(num_active_hosts=8, num_hosts=9), exit_on_done=True
    )
    model = build_and_shard_model()  # fully_shard over the default group
    optimizer = torch.optim.AdamW(model.parameters())
    try:
        step = ft.attach(
            model_parts=[model],
            optimizers=[optimizer],
            states={"dataloader": loader},
        )
        data = iter(loader)
        while step < num_steps:
            try:
                batch = next(data)
            except StopIteration:
                break
            try:
                train_step(model, optimizer, batch)
            except dist.DistError as e:
                # Only failures caused by peers or comms; let bugs raise.
                step = ft.recover(e)
                data = iter(loader)
                continue
            step += 1
            ft.step(step)
        ft.finish()
    except TrainingFinishedError:
        pass  # peers finished the last steps while this rank recovered
    finally:
        loader.close()  # the caller's cleanup
        ft.close()  # exits with EXIT_OK, EXIT_RESTART or EXIT_FAILED

Requirements on the training loop:

- All collectives use the default process group, the only one reconfigured.
  Timeouts are managed with ``dist.set_timeout``, which only applies to the
  default group.
- A step must not issue host side work that depends on a failed collective's
  result before raising (e.g. ``torch._assert_async`` on the loss): a device
  side assert poisons the CUDA context and forces a process restart.
- When a rank fails on its own (e.g. an exception in Python), its peers only
  notice once their next collective hits ``train_timeout_seconds``.
"""

import json
import logging
import os
import pickle
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, NoReturn

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ErrorType
from torch.distributed.checkpoint.state_dict import _init_optim_state
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FSDPModule
from torchft.fsdp.membership import (
    Assignment,
    DONE_FAILED,
    DONE_KEY,
    DONE_OK,
    Membership,
    RestartRequiredError,
    retry_store,
    STARTED_KEY,
    TrainingFinishedError,
    UnrecoverableError,
)
from torchft.fsdp.snapshot import (
    collect_state_tensors,
    Snapshotter,
    TransportPool,
    wait_for_keys,
)

logger = logging.getLogger(__name__)

# Exit codes of ``FaultTolerance.close`` with ``exit_on_done``.
EXIT_OK = 0
"""Training finished."""
EXIT_RESTART = 1
"""This host must be restarted, e.g. it was evicted."""
EXIT_FAILED = 3
"""Training failed for good; do not restart."""


@dataclass(kw_only=True, slots=True)
class FTFSDPConfig:
    """Fault tolerant FSDP settings."""

    num_active_hosts: int = 2
    """Hosts that train concurrently. Remaining hosts are hot spares."""

    num_hosts: int = 3
    """Total hosts including spares."""

    procs_per_host: int = 1
    """Trainer processes per host."""

    snapshot_interval: int = 1
    """Take an in-memory snapshot every N steps."""

    num_local_snapshots: int = 4
    """Pinned CPU snapshot slots per rank. Needs one slot in flight plus enough
    committed slots to cover the lag between local commit and replication."""

    meta_capacity_bytes: int = 4 << 20
    """Bytes reserved per snapshot for pickled CPU state (user state,
    optimizer scalars)."""

    quorum_timeout_seconds: float = 60.0
    """Timeout for one lighthouse quorum request. Requests are retried. Must
    exceed the lighthouse join timeout, otherwise a heartbeating host that
    never joins (e.g. hung in CUDA) blocks every quorum."""

    recovery_timeout_seconds: float = 600.0
    """Timeout for store waits during recovery. Must cover a spare building
    the model."""

    store_timeout_seconds: float = 30.0
    """Timeout of a single store operation."""

    init_timeout_seconds: float = 300.0
    """Process group timeout until the first step after each recovery
    completes, which may include lazy init and compilation on new hosts."""

    train_timeout_seconds: float = 100.0
    """Process group timeout during training. Bounds failure detection."""

    incomplete_host_timeout_seconds: float | None = None
    """How long a slot holder that misses processes in a quorum is waited
    for before it is evicted. Defaults to ``train_timeout_seconds`` plus 30s,
    enough for its remaining processes to detect the failure."""

    heartbeat_interval_seconds: float = 1.0
    """Lighthouse heartbeat period."""

    max_recoveries: int = 100
    """Give up after this many recoveries."""

    max_failures_per_step: int = 3
    """Give up when training fails this many times without progressing past
    the step it last failed at, e.g. on a deterministic error."""

    def __post_init__(self) -> None:
        if self.num_active_hosts < 2:
            raise ValueError("num_active_hosts must be >= 2")
        if self.num_hosts < self.num_active_hosts:
            raise ValueError("num_hosts must be >= num_active_hosts")
        if self.max_failures_per_step < 1:
            raise ValueError("max_failures_per_step must be >= 1")


def _event(name: str, **fields: Any) -> None:
    """Log a machine readable timing event."""
    logger.info(
        "FTFSDP_EVENT " + json.dumps({"event": name, "t": time.time(), **fields})
    )


def _exit_process(code: int) -> NoReturn:
    """Exit without interpreter teardown, during which NIXL can crash."""
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _connect_store(addr: str, run_id: str, timeout: timedelta) -> dist.Store:
    host, port = addr.rsplit(":", 1)
    store = dist.TCPStore(
        host.strip("[]"),
        int(port),
        is_master=False,
        timeout=timeout,
        wait_for_workers=False,
        use_libuv=True,
    )
    # The coordinator can outlive a trainer job; keep each run's keys apart.
    return dist.PrefixStore(run_id, store)


def init_optim_state(optimizer: torch.optim.Optimizer) -> None:
    """Materialize lazily created optimizer state so it can be snapshotted
    before the first step."""
    params = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
    if not any(optimizer.state.values()):
        if any(p.grad is not None for p in params):
            # _init_optim_state silently does nothing then.
            raise RuntimeError(
                "cannot initialize optimizer state: parameters already have "
                "gradients; call attach() before the first backward"
            )
        _init_optim_state(optimizer)
        # The step on zero gradients still moved the step counters and, with
        # weight decay, the moments. Reset them to a fresh optimizer's,
        # assuming lazily created state starts at zero as for Adam and SGD.
        # SGD with dampening still differs: its first step skips dampening.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    value.zero_()
                elif key == "step":
                    state[key] = 0
    # Stateless optimizers, e.g. SGD without momentum, create no state.
    if not any(optimizer.state.values()):
        return
    missing = sum(not optimizer.state.get(p) for p in params)
    if missing:
        raise RuntimeError(
            f"optimizer has no state for {missing} of {len(params)} "
            "parameters; state created after attach() is not snapshotted"
        )


def choose_restore_step(avail: list[dict[str, Any]], procs_per_host: int) -> int | None:
    """Pick the step every rank restores from the exchanged ``avail``, or
    ``None`` for a fresh start (no rank may have trained).

    A rank without state (e.g. a new host, also when its first recovery
    failed) reads its shard from the replica held by the next host.
    """
    if not any(a["started"] for a in avail):
        return None
    if not any(a["local"] or a["replica"] for a in avail):
        raise UnrecoverableError(
            "training already started in this run but no rank holds a "
            "snapshot, e.g. the job was relaunched; use a new run id"
        )
    world = len(avail)
    common: set[int] | None = None
    for r, a in enumerate(avail):
        if a["has_state"]:
            steps = set(a["local"])
        else:
            steps = {int(s) for s in avail[(r + procs_per_host) % world]["replica"]}
        common = steps if common is None else common & steps
    if not common:
        raise UnrecoverableError(
            "no snapshot step is held by every rank; adjacent hosts likely "
            f"failed together: {avail}"
        )
    return max(common)


class FaultTolerance:
    """Drives membership, snapshots and recovery for one trainer process.

    Construct it before building the model: it initializes the default
    ``nccl2`` process group and blocks spares until they are assigned a slot.
    Arguments left as ``None`` are read from the environment:

    - ``store_addr``: ``FTFSDP_STORE_ADDR``, ``host:port`` of the shared
      TCPStore (see :mod:`torchft.fsdp.coordinator`).
    - ``lighthouse_addr``: ``TORCHFT_LIGHTHOUSE``.
    - ``host``: ``FTFSDP_HOST_NAME``, defaults to the hostname. Unique per
      host.
    - ``host_index``: ``FTFSDP_HOST_INDEX``, in ``[0, num_hosts)``.
    - ``local_rank``: ``LOCAL_RANK``.
    - ``run_id``: ``FTFSDP_RUN_ID``, defaults to ``"ftfsdp"``. Shared by all
      hosts of a run.

    ``attach``, ``recover`` and ``finish`` raise
    :class:`TrainingFinishedError` when training finished elsewhere,
    :class:`RestartRequiredError` when this host must restart (e.g. it was
    evicted or its CUDA context is unusable) and :class:`UnrecoverableError`
    when training failed for good. Failures are published so spares and the
    coordinator exit.

    If ``exit_on_done`` is set, the constructor exits the process instead of
    raising, which covers spares waiting for a slot before any trainer state
    exists, and ``close`` exits with ``EXIT_OK`` (0) once training finished,
    ``EXIT_FAILED`` (3) once it failed, here or elsewhere, and
    ``EXIT_RESTART`` (1) otherwise.
    """

    def __init__(
        self,
        config: FTFSDPConfig,
        *,
        store_addr: str | None = None,
        lighthouse_addr: str | None = None,
        host: str | None = None,
        host_index: int | None = None,
        local_rank: int | None = None,
        run_id: str | None = None,
        exit_on_done: bool = False,
    ) -> None:
        env = os.environ
        local_rank = int(env["LOCAL_RANK"]) if local_rank is None else local_rank
        run_id = env.get("FTFSDP_RUN_ID", "ftfsdp") if run_id is None else run_id
        store_addr = store_addr or env["FTFSDP_STORE_ADDR"]
        self.config = config
        self.exit_on_done = exit_on_done
        self.finished = False
        self.failed = False
        self.num_recoveries = 0
        self.current_step = 0
        self._failed_step = -1
        self._num_failures_at_step = 0
        # Whether this process has passed a restore barrier, i.e. may have
        # trained.
        self._started = False
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(self.device)
        # A recovery connects at most two links; spares create theirs while
        # they wait for a slot.
        self.transports = TransportPool(2)
        recovery_timeout = timedelta(seconds=config.recovery_timeout_seconds)
        self.store = _connect_store(
            store_addr, run_id, timedelta(seconds=config.store_timeout_seconds)
        )
        self.model_parts: list[torch.nn.Module] = []
        self.optimizers: list[torch.optim.Optimizer] = []
        self.snapshotter: Snapshotter | None = None
        self._states: dict[str, Stateful] = {}
        self._replay_init: Callable[[], None] | None = None
        self._shorten_timeout = False
        incomplete_timeout = config.incomplete_host_timeout_seconds
        if incomplete_timeout is None:
            incomplete_timeout = (
                max(config.train_timeout_seconds, config.init_timeout_seconds) + 30.0
            )
        self.membership = Membership(
            store=self.store,
            pg_store=_connect_store(store_addr, run_id, recovery_timeout),
            lighthouse_addr=lighthouse_addr or env["TORCHFT_LIGHTHOUSE"],
            host=host or env.get("FTFSDP_HOST_NAME") or socket.gethostname(),
            host_index=(
                int(env["FTFSDP_HOST_INDEX"]) if host_index is None else host_index
            ),
            local_rank=local_rank,
            num_hosts=config.num_hosts,
            num_slots=config.num_active_hosts,
            procs_per_host=config.procs_per_host,
            run_id=run_id,
            # Every reconfiguration resets to the init timeout so lazy init
            # and compilation on new hosts fit in the first step.
            pg_timeout=timedelta(seconds=config.init_timeout_seconds),
            quorum_timeout=timedelta(seconds=config.quorum_timeout_seconds),
            spare_quorum_timeout=recovery_timeout,
            incomplete_timeout=timedelta(seconds=incomplete_timeout),
            heartbeat_interval=config.heartbeat_interval_seconds,
        )
        _event("process_start", host=self.membership.host, uid=self.membership.uid)
        try:
            self._assignment: Assignment | None = self._next_assignment()
        except (TrainingFinishedError, RestartRequiredError, UnrecoverableError):
            # No trainer state exists yet, so exiting here skips no cleanup.
            if exit_on_done:
                self.close()
            raise

    def attach(
        self,
        *,
        model_parts: list[torch.nn.Module],
        optimizers: list[torch.optim.Optimizer],
        states: dict[str, Stateful] | None = None,
        replay_init: Callable[[], None] | None = None,
    ) -> int:
        """Register the training state, run the initial recovery and return
        the step to resume from.

        - ``model_parts``/``optimizers``: sharded parameters, persistent
          buffers and optimizer state are snapshotted. Optimizer state is
          materialized here. All of them must be updated in place, e.g. no
          ``optimizer.load_state_dict`` after this call.
        - ``states``: other per-rank state restored with a snapshot
          (dataloader, LR schedulers, RNG, ...), as for DCP. Its state dicts
          are pickled.
        - ``replay_init``: run by processes that called ``attach`` when a
          rank of the new quorum has not, e.g. a new host. It must issue
          exactly the collectives, in the same order, that a new host issues
          between constructing this object and calling ``attach`` (e.g.
          DTensor weight init). Its results are overwritten by the restore.
          Model init must use seeded RNG rather than broadcast a seed.
        """
        # The caller's init collectives are done, so later quorums replay
        # them for other processes instead of waiting for this one to run
        # them.
        self.membership.initialized = True
        self.model_parts = model_parts
        self.optimizers = optimizers
        self._states = states or {}
        self._replay_init = replay_init
        cfg = self.config
        # These errors are deterministic, so restarting does not help.
        try:
            if self._assignment is None:
                raise RuntimeError("attach() must be called once")
            for opt in optimizers:
                init_optim_state(opt)
            # Capture would fail every recovery attempt.
            meta_len = len(self._meta())
            if meta_len > cfg.meta_capacity_bytes:
                raise ValueError(
                    f"snapshot metadata is {meta_len} bytes, larger than "
                    f"meta_capacity_bytes={cfg.meta_capacity_bytes}"
                )
        except Exception:
            self._mark_failed()
            raise
        device_tensors, _ = collect_state_tensors(model_parts, optimizers)
        self.snapshotter = Snapshotter(
            device_tensors,
            transports=self.transports,
            device=self.device,
            ident=self.membership.ident,
            store=self.store,
            num_local=cfg.num_local_snapshots,
            meta_capacity=cfg.meta_capacity_bytes,
            interval=cfg.snapshot_interval,
            procs_per_host=cfg.procs_per_host,
            timeout=cfg.recovery_timeout_seconds,
            comm_failed=self._comm_failed,
            recovery_pending=self.membership.recovery_pending,
        )
        self.snapshotter.register_optimizer_hooks(optimizers)
        assignment, self._assignment = self._assignment, None
        assert assignment is not None
        return self._recover_loop(assignment, None)

    def step(self, step: int) -> None:
        """Call after each completed optimizer step, ``step`` counting from
        1."""
        snap = self._snap()
        self.current_step = step
        if self._shorten_timeout:
            self._set_pg_timeouts(self.config.train_timeout_seconds)
            self._shorten_timeout = False
        if step % self.config.snapshot_interval == 0:
            snap.capture(step, self._meta())

    def recover(self, err: Exception) -> int:
        """Recover from a failed step and return the restored step.

        Rejoins the quorum if recovery itself fails, up to
        ``max_recoveries`` in total. Raises ``err`` once training failed
        ``max_failures_per_step`` times without progressing.
        """
        if self.current_step != self._failed_step:
            self._failed_step = self.current_step
            self._num_failures_at_step = 0
        self._num_failures_at_step += 1
        if self._num_failures_at_step > self.config.max_failures_per_step:
            logger.error(f"training failed repeatedly after step {self._failed_step}")
            self._mark_failed()
            raise err
        return self._recover_loop(None, err)

    def finish(self) -> None:
        """Mark training as done so spares exit."""
        self.finished = True
        # Every rank, so spares exit even if rank 0 dies now. Never
        # overwrites a failed outcome.
        retry_store(lambda: self.store.compare_set(DONE_KEY, "", DONE_OK), "set")
        self._snap().flush()
        _event("training_done", step=self.current_step)

    def close(self) -> None:
        """Release snapshot transports and stop the heartbeat. With
        ``exit_on_done`` this logs an in-flight exception and exits the
        process, skipping interpreter teardown, during which NIXL can
        crash."""
        if self.snapshotter is not None:
            self.snapshotter.close()
        self.transports.close()
        self.membership.close()
        if not self.exit_on_done:
            return
        err = sys.exc_info()[1]
        if self.failed or isinstance(err, UnrecoverableError):
            code = EXIT_FAILED
        elif self.finished or isinstance(err, TrainingFinishedError):
            code = EXIT_OK
        else:
            code = EXIT_RESTART
        if err is not None and not isinstance(err, TrainingFinishedError):
            logger.error(f"exiting with status {code}", exc_info=err)
        else:
            logger.info(f"exiting with status {code}")
        _exit_process(code)

    # Internals.

    def _snap(self) -> Snapshotter:
        if self.snapshotter is None:
            raise RuntimeError("attach() has not been called")
        return self.snapshotter

    def _mark_failed(self) -> None:
        """Publish the failed outcome so spares and the coordinator exit."""
        if self.failed:
            return
        self.failed = True
        _event("training_failed", step=self.current_step)
        try:
            # Never overwrites a finished outcome.
            self.store.compare_set(DONE_KEY, "", DONE_FAILED)
        except Exception as e:
            logger.warning(f"publishing the failed outcome failed: {e}")

    def _recover_loop(
        self, assignment: Assignment | None, err: BaseException | None
    ) -> int:
        """Restore with ``assignment``, or rejoin the quorum after ``err``,
        retrying failed recoveries."""
        while True:
            if assignment is None:
                assert err is not None
                self.num_recoveries += 1
                if self.num_recoveries > self.config.max_recoveries:
                    logger.error(
                        f"giving up after {self.config.max_recoveries} recoveries"
                    )
                    self._mark_failed()
                    raise err
            try:
                if assignment is None:
                    assert err is not None
                    assignment = self._handle_failure(err)
                self._recover(assignment)
                return self.current_step
            except TrainingFinishedError:
                raise
            except RestartRequiredError:
                # Otherwise peers wait for this process in the current
                # recovery until it times out.
                self.membership.request_recovery_once()
                raise
            except UnrecoverableError:
                self._mark_failed()
                raise
            except Exception as e:
                logger.exception("recovery failed; rejoining quorum")
                err, assignment = e, None

    def _next_assignment(self) -> Assignment:
        try:
            assignment = self.membership.next_assignment()
        except TrainingFinishedError:
            self.finished = True
            raise
        except UnrecoverableError:
            self._mark_failed()
            raise
        _event("assigned", gen=assignment.gen, rank=dist.get_rank())
        return assignment

    def _cpu_tensors(self) -> list[torch.Tensor]:
        return collect_state_tensors(self.model_parts, self.optimizers)[1]

    def _meta(self) -> bytes:
        state = {
            "step": self.current_step,
            "states": {k: v.state_dict() for k, v in self._states.items()},
            "param_groups": [
                [{k: v for k, v in g.items() if k != "params"} for g in o.param_groups]
                for o in self.optimizers
            ],
            "cpu_tensors": [t.clone() for t in self._cpu_tensors()],
        }
        return pickle.dumps(state)

    def _load_meta(self, meta: bytes) -> None:
        state = pickle.loads(meta)
        self.current_step = state["step"]
        for k, v in self._states.items():
            v.load_state_dict(state["states"][k])
        for opt, groups in zip(self.optimizers, state["param_groups"], strict=True):
            for group, saved in zip(opt.param_groups, groups, strict=True):
                group.update(saved)
        for t, saved in zip(self._cpu_tensors(), state["cpu_tensors"], strict=True):
            t.copy_(saved)

    def _comm_failed(self) -> bool:
        # Polled instead of a Python abort hook: the watchdog needs the GIL to
        # run one, while an autograd thread can hold the GIL spinning in a
        # kernel launch queued behind the hung collective. The hook then
        # never returns and the comm is never revoked.
        pg = dist.distributed_c10d._get_default_group()
        return pg._get_backend(self.device).get_error() != ErrorType.SUCCESS

    def _set_pg_timeouts(self, seconds: float) -> None:
        # Only the default group; collectives on other groups keep theirs.
        dist.set_timeout(timedelta(seconds=seconds))

    def _handle_failure(self, err: BaseException) -> Assignment:
        snap = self._snap()
        _event("failure_detected", step=self.current_step, error=str(err)[:200])
        self._dump_flight_recorder()
        snap.abort()
        for part in self.model_parts:
            if not isinstance(part, FSDPModule):
                continue
            try:
                part.reset_iter_state()
            except Exception as e:
                logger.warning(f"FSDP reset_iter_state failed: {e}")
        for opt in self.optimizers:
            opt.zero_grad()
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise RestartRequiredError(f"CUDA unusable after failure: {e}") from e
        snap.pause()
        return self._next_assignment()

    def _dump_flight_recorder(self) -> None:
        """Write this rank's nccl2 flight recorder trace for each failure.

        The automatic dump on collective failure fires once per process, so
        later failures would otherwise leave no trace.
        """
        prefix = os.environ.get("TORCH_FR_DUMP_TEMP_FILE")
        if not prefix:
            return
        try:
            trace = torch._C._distributed_c10d._dump_fr_trace(
                True, False, False, "nccl2"
            )
            with open(f"{prefix}{dist.get_rank()}.f{self.num_recoveries}", "wb") as f:
                f.write(trace)
        except Exception:
            logger.exception("flight recorder dump failed")

    def _recover(self, assignment: Assignment) -> None:
        start = time.perf_counter()
        cfg = self.config
        store = self.store
        snap = self._snap()
        rank, world = dist.get_rank(), dist.get_world_size()
        seq = assignment.seq
        timeout = cfg.recovery_timeout_seconds
        is_new = self.membership.host in assignment.new_hosts
        # Not from new_hosts: a process whose reconfigure failed in the
        # constructor keeps its slot but has not run init.
        replay = assignment.ranks[rank].initialized and not all(
            m.initialized for m in assignment.ranks
        )

        if not assignment.initial:
            # The first step after recovery may compile on new hosts.
            # step() shortens it again.
            self._set_pg_timeouts(cfg.init_timeout_seconds)
            if replay and self._replay_init is not None:
                # Known limitation: replay_init does not poll
                # recovery_pending, so if a new host fails meanwhile,
                # survivors only notice at the init timeout.
                replay_start = time.perf_counter()
                self._replay_init()
                _event("init_replayed", seconds=time.perf_counter() - replay_start)
        snap.update_links(
            rank=rank,
            ident_of_rank=[m.ident for m in assignment.ranks],
            seq=seq,
            timeout=timeout,
        )
        links_s = time.perf_counter() - start

        avail = self._exchange_avail(seq)
        # Every rank decides from the same exchanged data.
        step = choose_restore_step(avail, cfg.procs_per_host)
        if step is None:
            # pause() after a failed attempt discards captures until resumed.
            snap.resume()
            snap.capture(self.current_step, self._meta(), replicate=False)
            snap.flush()
            # Not by the worker, whose transfers do not poll recovery_pending
            # and whose errors are only logged.
            snap.replicate_now(self.current_step)
        else:
            self._restore_from_snapshots(step, avail)
        restore_s = time.perf_counter() - start - links_s

        # Per-rank keys rather than a counter so a retried set is idempotent.
        barrier = f"ftfsdp/restored/{seq}"
        store.set(f"{barrier}/{rank}", "1")
        wait_for_keys(
            store,
            [f"{barrier}/{r}" for r in range(world)],
            timeout,
            self.membership.recovery_pending,
        )
        if not self._started:
            self._started = True
            # Covers quorums of only new processes, e.g. after a job restart
            # against a coordinator that outlived it.
            store.set(STARTED_KEY, "1")
        snap.resume()
        self._shorten_timeout = True
        _event(
            "recovered",
            gen=assignment.gen,
            rank=rank,
            step=self.current_step,
            new=is_new,
            links_s=links_s,
            restore_s=restore_s,
            total_s=time.perf_counter() - start,
        )

    def _exchange_avail(self, seq: int) -> list[dict[str, Any]]:
        """Share which snapshots each rank holds, whether training started and
        whether this rank holds its training state."""
        store = self.store
        snap = self._snap()
        rank, world = dist.get_rank(), dist.get_world_size()
        store.set(
            f"ftfsdp/avail/{seq}/{rank}",
            json.dumps(
                {
                    "started": self._started or store.check([STARTED_KEY]),
                    "has_state": self._started,
                    "local": snap.committed_steps(),
                    "replica": {str(s): v for s, v in snap.replica_steps().items()},
                }
            ),
        )
        keys = [f"ftfsdp/avail/{seq}/{r}" for r in range(world)]
        wait_for_keys(
            store,
            keys,
            self.config.recovery_timeout_seconds,
            self.membership.recovery_pending,
        )
        return [json.loads(v) for v in store.multi_get(keys)]

    def _restore_from_snapshots(self, step: int, avail: list[dict[str, Any]]) -> None:
        snap = self._snap()
        rank, world = dist.get_rank(), dist.get_world_size()
        g = self.config.procs_per_host
        fetch = not avail[rank]["has_state"]
        if fetch:
            index, meta_len = avail[(rank + g) % world]["replica"][str(step)]
            fetch_start = time.perf_counter()
            snap.fetch_replica(step, index, meta_len)
            _event(
                "replica_fetched", step=step, seconds=time.perf_counter() - fetch_start
            )
        lost = None if fetch else self.current_step - step
        self._load_meta(snap.restore(step))
        snap.replicate_now(step, fetched=fetch)
        logger.info(f"restored step {step} (lost {lost} steps)")
