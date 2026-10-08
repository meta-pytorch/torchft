# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import threading
import time
import unittest
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch.distributed as dist
from torchft.fsdp import membership
from torchft.fsdp.membership import (
    assign_slots,
    DONE_KEY,
    LATEST_KEY,
    MemberInfo,
    Membership,
    reconfigure_uuid,
    TrainingFinishedError,
    UnrecoverableError,
)


def _host(
    name: str,
    *,
    slot: int = -1,
    gen: int = -1,
    procs: int = 2,
    skip=(),
    latest_gen: int = -1,
    restarted: bool = False,
    since: float = -1.0,
):
    return [
        MemberInfo(
            host=name,
            local_rank=l,
            uid=hash((name, l)),
            ident=f"{name}/{l}",
            handle=f"{name}/{l}",
            slot=slot,
            gen=gen,
            latest_gen=latest_gen,
            restarted=restarted,
            since=since,
        )
        for l in range(procs)
        if l not in skip
    ]


class AssignSlotsTest(unittest.TestCase):
    def test_initial(self) -> None:
        members = _host("c") + _host("a") + _host("b")
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.gen, 0)
        self.assertTrue(a.initial)
        self.assertEqual(a.host_slots, {"a": 0, "b": 1})
        self.assertEqual([m.handle for m in a.ranks], ["a/0", "a/1", "b/0", "b/1"])
        self.assertEqual(a.new_hosts, {"a", "b"})
        self.assertEqual(a.evict, set())

    def test_survivors_keep_slots(self) -> None:
        members = _host("z", slot=1, gen=3) + _host("a") + _host("y", slot=2, gen=3)
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        self.assertEqual(a.gen, 4)
        self.assertFalse(a.initial)
        self.assertEqual(a.host_slots, {"a": 0, "z": 1, "y": 2})
        self.assertEqual(a.new_hosts, {"a"})
        self.assertEqual(
            [m.handle for m in a.ranks], ["a/0", "a/1", "z/0", "z/1", "y/0", "y/1"]
        )

    def test_not_enough_spares(self) -> None:
        members = _host("a", slot=0, gen=1) + _host("s", skip=(1,))
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.ranks, ())
        self.assertEqual(a.host_slots, {})
        self.assertEqual(a.gen, 1)

    def test_waits_for_incomplete(self) -> None:
        members = (
            _host("a", slot=0, gen=2, skip=(0,), since=100.0)
            + _host("b", slot=2, gen=1)
            + _host("c", slot=1, gen=2, since=50.0)
            + _host("s1")
            # Spares wait across recoveries.
            + _host("s2", since=0.0)
        )
        a = assign_slots(
            members, num_slots=3, procs_per_host=2, now=120.0, incomplete_timeout=30
        )
        self.assertEqual(a.ranks, ())
        self.assertEqual(a.evict, {"b"})

    def test_evicts_incomplete_and_stale(self) -> None:
        members = (
            _host("a", slot=0, gen=2, skip=(0,), since=100.0)
            + _host("b", slot=2, gen=1)
            + _host("c", slot=1, gen=2)
            + _host("d", slot=3, gen=2)
            + _host("s1")
            + _host("s2")
        )
        a = assign_slots(
            members, num_slots=4, procs_per_host=2, now=130.0, incomplete_timeout=30
        )
        self.assertEqual(a.evict, {"a", "b"})
        self.assertEqual(a.host_slots, {"s1": 0, "c": 1, "s2": 2, "d": 3})
        self.assertEqual(a.gen, 3)

    def test_inconsistent_host_slots_evicted(self) -> None:
        members = (
            [
                MemberInfo("a", 0, 0, "a/0", "a/0", 0, 1),
                MemberInfo("a", 1, 1, "a/1", "a/1", 1, 1),
            ]
            + _host("b", slot=1, gen=1)
            + _host("s")
        )
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.evict, {"a"})
        self.assertEqual(a.host_slots, {"s": 0, "b": 1})

    def test_duplicate_slot_raises(self) -> None:
        members = _host("a", slot=0, gen=1) + _host("b", slot=0, gen=1)
        with self.assertRaisesRegex(UnrecoverableError, "both claim slot 0"):
            assign_slots(members, num_slots=2, procs_per_host=2)

    def test_slot_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(UnrecoverableError, "claims slot 5"):
            assign_slots(_host("a", slot=5, gen=0), num_slots=2, procs_per_host=2)

    def test_extra_spares_unassigned(self) -> None:
        members = _host("a", slot=0, gen=0) + _host("s1") + _host("s2")
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.host_slots, {"a": 0, "s1": 1})
        self.assertNotIn("s2", a.host_slots)

    def test_order_independent(self) -> None:
        members = _host("b", slot=1, gen=0) + _host("x") + _host("y")
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        b = assign_slots(list(reversed(members)), num_slots=3, procs_per_host=2)
        self.assertEqual(a, b)

    def test_adjacent_free_slots_rejected(self) -> None:
        # Slots 0 and 1 are lost, e.g. a rack failed. Wait in case a holder
        # has not detected the failure yet, then give up.
        def assign(since: float):
            members = _host("c", slot=2, gen=1, since=since) + _host("s1") + _host("s2")
            return assign_slots(
                members, num_slots=3, procs_per_host=2, now=100, incomplete_timeout=30
            )

        a = assign(since=80)
        self.assertEqual(a.ranks, ())
        self.assertEqual(a.gen, 1)
        with self.assertRaisesRegex(UnrecoverableError, r"slots \[0\]"):
            assign(since=70)

    def test_only_spares_wait(self) -> None:
        members = _host("s1", latest_gen=1, since=0) + _host("s2", latest_gen=1)
        a = assign_slots(
            members, num_slots=2, procs_per_host=2, now=100, incomplete_timeout=30
        )
        self.assertEqual(a.ranks, ())

    def test_free_slot_needs_successor(self) -> None:
        members = _host("a", slot=0, gen=1) + _host("b", slot=1, gen=1) + _host("s")
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        self.assertEqual(a.host_slots, {"a": 0, "b": 1, "s": 2})
        members = _host("a", slot=0, gen=1) + _host("c", slot=2, gen=1) + _host("s")
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        self.assertEqual(a.host_slots, {"a": 0, "s": 1, "c": 2})

    def test_restarted_holder_waits_for_survivor(self) -> None:
        # A held slot 0 of gen 3 and restarted while B, holding slot 1, is
        # still blocked in a collective. Without B the shard of slot 0 is
        # lost, so {A', S} must not start a new training run.
        members = _host("a", latest_gen=3, restarted=True) + _host("s", latest_gen=3)
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.ranks, ())
        self.assertEqual(a.gen, 3)

        # B never joins, e.g. it failed too.
        with self.assertRaises(UnrecoverableError):
            assign_slots(
                _host("a", latest_gen=3, restarted=True, since=0)
                + _host("s", latest_gen=3),
                num_slots=2,
                procs_per_host=2,
                now=100,
                incomplete_timeout=30,
            )

        members += _host("b", slot=1, gen=3, latest_gen=3)
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.gen, 4)
        self.assertFalse(a.initial)
        self.assertEqual(a.host_slots, {"a": 0, "b": 1})

    def test_all_holders_restarted(self) -> None:
        # E.g. a job relaunch: no survivor can join, so the restore decides.
        members = (
            _host("a", latest_gen=3, restarted=True)
            + _host("b", latest_gen=3, restarted=True)
            + _host("s", latest_gen=3)
        )
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.gen, 4)
        self.assertEqual(a.new_hosts, {"a", "b"})

    def test_holder_behind_published_gen_evicted(self) -> None:
        members = _host("c", slot=1, gen=3, latest_gen=4) + _host("s1") + _host("s2")
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.evict, {"c"})
        self.assertEqual(a.ranks, ())

    def test_uuid(self) -> None:
        u = reconfigure_uuid("run", 7)
        self.assertEqual(u, reconfigure_uuid("run", 7))
        self.assertNotEqual(u, reconfigure_uuid("run", 8))
        self.assertTrue(0 <= u < 1 << 63)


def _publish(store: dist.Store, seq: int, hosts=()) -> None:
    store.set(LATEST_KEY, json.dumps({"gen": 0, "seq": seq, "hosts": list(hosts)}))


class WaitForPublishedTest(unittest.TestCase):
    def _membership(self, timeout: float) -> Membership:
        m = object.__new__(Membership)
        m.store = dist.HashStore()
        m.pg_timeout = timedelta(seconds=timeout)
        return m

    def test_returns_when_published(self) -> None:
        m = self._membership(10)
        _publish(m.store, 1)
        threading.Timer(0.3, lambda: _publish(m.store, 2)).start()
        start = time.monotonic()
        m._wait_for_published(2)
        self.assertLess(time.monotonic() - start, 5)

    def test_times_out(self) -> None:
        m = self._membership(0.5)
        start = time.monotonic()
        m._wait_for_published(1)
        self.assertGreaterEqual(time.monotonic() - start, 0.5)

    def test_returns_on_failed_reconfigure(self) -> None:
        m = self._membership(10)
        threading.Timer(
            0.3, lambda: m.store.set(membership._recovery_key(2), "1")
        ).start()
        start = time.monotonic()
        m._wait_for_published(2)
        self.assertLess(time.monotonic() - start, 5)

    def test_publish_latest_never_goes_back(self) -> None:
        m = self._membership(10)

        def seq() -> int:
            return json.loads(m.store.get(LATEST_KEY))["seq"]

        m._publish_latest({"gen": 1, "seq": 5, "hosts": []})
        self.assertEqual(seq(), 5)
        # A stalled rank 0 of an older quorum.
        m._publish_latest({"gen": 0, "seq": 3, "hosts": []})
        self.assertEqual(seq(), 5)
        m._publish_latest({"gen": 2, "seq": 9, "hosts": []})
        self.assertEqual(seq(), 9)

    def test_request_recovery_once(self) -> None:
        m = self._membership(10)
        m.slot, m.seq = -1, 4
        m.request_recovery_once()
        self.assertFalse(m.store.check([membership._recovery_key(4)]))
        m.slot = 0
        m.request_recovery_once()
        self.assertTrue(m.store.check([membership._recovery_key(4)]))
        # Store errors are not retried or raised.
        m.store = mock.Mock(set=mock.Mock(side_effect=dist.DistNetworkError("x")))
        m.request_recovery_once()
        m.store.set.assert_called_once()


class _CountingStore:
    """HashStore that counts ``check`` calls."""

    def __init__(self) -> None:
        self.store = dist.HashStore()
        self.checks = 0

    def check(self, keys: list[str]) -> bool:
        self.checks += 1
        return self.store.check(keys)

    def __getattr__(self, name: str):
        return getattr(self.store, name)


def _lighthouse(*results) -> mock.Mock:
    """Fake client whose quorum returns or raises ``results`` in order, then
    blocks forever."""
    it = iter(results)

    def quorum(**kwargs):
        result = next(it, None)
        if result is None:
            threading.Event().wait()
        if isinstance(result, BaseException):
            raise result
        return result

    return mock.Mock(quorum=quorum)


class _FlakyStore:
    """HashStore whose first ``check`` raises a transient store error."""

    def __init__(self) -> None:
        self.store = dist.HashStore()
        self.failures = 0

    def check(self, keys: list[str]) -> bool:
        if self.failures == 0:
            self.failures += 1
            raise dist.DistNetworkError("connection reset")
        return self.store.check(keys)

    def __getattr__(self, name: str):
        return getattr(self.store, name)


def _quorum(quorum_id: int, created_ns: int, members: list[MemberInfo]):
    return SimpleNamespace(
        quorum_id=quorum_id,
        created=SimpleNamespace(seconds=created_ns // 10**9, nanos=created_ns % 10**9),
        participants=[SimpleNamespace(data=asdict(m)) for m in members],
    )


class NextAssignmentTest(unittest.TestCase):
    """Two hosts with one process each and two slots."""

    def setUp(self) -> None:
        self.reconfigure = mock.Mock()
        self.sleep = mock.Mock()
        patches = [
            mock.patch.object(dist, "_get_reconfigure_handle", return_value="h"),
            mock.patch.object(dist, "_reconfigure", self.reconfigure),
            mock.patch.object(dist, "get_rank", return_value=0),
            mock.patch.object(dist, "get_world_size", return_value=2),
            mock.patch.object(membership.time, "sleep", self.sleep),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _membership(self, store, host: str = "a") -> Membership:
        m = object.__new__(Membership)
        m.store = store
        m.host = host
        m.local_rank = 0
        m.num_slots = 2
        m.procs_per_host = 1
        m.run_id = "run"
        m.pg_timeout = timedelta(seconds=1)
        m.quorum_timeout = timedelta(seconds=1)
        m.spare_quorum_timeout = timedelta(seconds=1)
        m.incomplete_timeout = timedelta(seconds=1)
        m.uid = 0
        m.replica_id = f"{host}/0"
        m.ident = f"{host}/0/{id(m)}"
        m.slot = m.gen = m.seq = -1
        m.initialized = False
        return m

    @staticmethod
    def _spare(host: str) -> MemberInfo:
        return MemberInfo(host, 0, 0, f"{host}/0/x", "h", -1, -1)

    def _members(self) -> list[MemberInfo]:
        return [self._spare("a"), self._spare("b")]

    def test_keys_unique_per_quorum(self) -> None:
        store = dist.HashStore()
        first = self._membership(store)
        with mock.patch.object(
            first, "_quorum", return_value=_quorum(1, 10**9, self._members())
        ):
            a = first.next_assignment()
        self.assertEqual(a.seq, 10**9)
        self.assertEqual(first.seq, 10**9)
        self.assertEqual(json.loads(store.get(LATEST_KEY))["hosts"], ["a", "b"])
        store.set(membership._recovery_key(a.seq), "1")

        # A relaunch with the same run id: the lighthouse reuses the quorum
        # id since the replica ids did not change. Every slot holder
        # restarted, so the assignment does not wait for survivors.
        relaunched = self._membership(store)
        members = [
            MemberInfo(h, 0, 0, f"{h}/0/y", "h", -1, -1, latest_gen=0, restarted=True)
            for h in ("a", "b")
        ]
        with mock.patch.object(
            relaunched, "_quorum", return_value=_quorum(1, 5 * 10**9, members)
        ):
            b = relaunched.next_assignment()
        self.assertEqual(b.gen, a.gen + 1)
        self.assertNotEqual(b.seq, a.seq)
        self.assertNotEqual(
            self.reconfigure.call_args_list[0].args[0],
            self.reconfigure.call_args_list[1].args[0],
        )
        self.assertFalse(relaunched.recovery_pending())
        self.assertEqual(json.loads(store.get(LATEST_KEY))["seq"], b.seq)

    def test_restarted_slot_holder_requests_recovery(self) -> None:
        store = dist.HashStore()
        _publish(store, 7, hosts=["a", "b"])
        latest = json.loads(store.get(LATEST_KEY))
        self.assertFalse(self._membership(store, "s")._recovery_requested(latest))
        self.assertTrue(self._membership(store, "a")._recovery_requested(latest))
        self.assertTrue(store.check([membership._recovery_key(7)]))
        self.assertTrue(self._membership(store, "s")._recovery_requested(latest))

    def test_restarted_slot_holder_waits_for_survivor(self) -> None:
        store = dist.HashStore()
        store.set(LATEST_KEY, json.dumps({"gen": 3, "seq": 7, "hosts": ["a", "b"]}))
        m = self._membership(store, "a")
        spare = MemberInfo("s", 0, 1, "s/0/x", "h", -1, -1, latest_gen=3)
        survivor = MemberInfo("b", 0, 2, "b/0/x", "h", 1, 3, latest_gen=3)
        quorums = []

        def quorum(data, timeout):
            me = MemberInfo(**data)
            self.assertTrue(me.restarted)
            # B joins once it detects the failure.
            others = [spare] if not quorums else [spare, survivor]
            quorums.append(others)
            return _quorum(len(quorums), (8 + len(quorums)) * 10**9, [me] + others)

        with mock.patch.object(m, "_quorum", side_effect=quorum):
            a = m.next_assignment()
        self.assertEqual(len(quorums), 2)
        self.assertEqual(self.reconfigure.call_count, 1)
        self.assertEqual((a.gen, m.slot), (4, 0))
        self.assertEqual(json.loads(store.get(LATEST_KEY))["gen"], 4)

    def test_failed_reconfigure_requests_recovery(self) -> None:
        store = dist.HashStore()
        m = self._membership(store)
        self.reconfigure.side_effect = [RuntimeError("timeout"), mock.Mock()]
        survivor = MemberInfo("b", 0, 1, "b/0/x", "h", 1, 0, latest_gen=0)
        sent = []

        def quorum(data, timeout):
            sent.append(MemberInfo(**data))
            if len(sent) == 1:
                return _quorum(1, 10**9, self._members())
            # B reconfigured and published gen 0.
            return _quorum(2, 2 * 10**9, [sent[-1], survivor])

        with mock.patch.object(m, "_quorum", side_effect=quorum):
            a = m.next_assignment()
        # The failed process adopted gen 0, so it is not evicted as stale.
        self.assertEqual((sent[1].slot, sent[1].gen), (0, 0))
        # The reconfigure time does not count as waiting.
        self.assertEqual(sent[1].since, -1.0)
        self.assertEqual((a.seq, a.gen), (2 * 10**9, 1))
        self.assertTrue(store.check([membership._recovery_key(10**9)]))

    def test_reports_initialized(self) -> None:
        m = self._membership(dist.HashStore())
        sent = []

        def quorum(data, timeout):
            sent.append(MemberInfo(**data))
            return _quorum(len(sent), len(sent) * 10**9, self._members())

        with mock.patch.object(m, "_quorum", side_effect=quorum):
            m.next_assignment()
            m.initialized = True
            m.next_assignment()
        self.assertEqual([i.initialized for i in sent], [False, True])

    def test_since_is_lighthouse_time(self) -> None:
        m = self._membership(dist.HashStore())
        sent = []

        def quorum(data, timeout):
            sent.append(MemberInfo(**data))
            if len(sent) < 3:
                return _quorum(len(sent), (50 + len(sent)) * 10**9, [self._spare("a")])
            return _quorum(3, 60 * 10**9, self._members())

        with mock.patch.object(m, "_quorum", side_effect=quorum):
            m.next_assignment()
        self.assertEqual([i.since for i in sent], [-1.0, 51.0, 51.0])

    def test_since_reset_after_left_out(self) -> None:
        m = self._membership(dist.HashStore(), host="c")
        sent = []

        def quorum(data, timeout):
            sent.append(MemberInfo(**data))
            if len(sent) == 1:
                return _quorum(1, 50 * 10**9, self._members() + [self._spare("c")])
            return _quorum(2, 60 * 10**9, [sent[-1], self._spare("d")])

        with (
            mock.patch.object(m, "_quorum", side_effect=quorum),
            mock.patch.object(m, "_wait_for_published") as wait,
        ):
            m.next_assignment()
        wait.assert_called_once_with(50 * 10**9)
        self.assertEqual([i.since for i in sent], [-1.0, -1.0])

    def test_recovery_pending_on_done(self) -> None:
        store = dist.HashStore()
        m = self._membership(store)
        self.assertFalse(m.recovery_pending())
        store.set(DONE_KEY, membership.DONE_FAILED)
        self.assertTrue(m.recovery_pending())

    def test_connection_errors_back_off(self) -> None:
        m = self._membership(dist.HashStore())
        errors = [RuntimeError("connection refused")] * 3 + [TimeoutError("deadline")]
        quorums = errors + [_quorum(1, 10**9, self._members())]
        with mock.patch.object(m, "_quorum", side_effect=quorums):
            m.next_assignment()
        self.assertEqual(
            [c.args[0] for c in self.sleep.call_args_list], [0.5, 1.0, 2.0, 4.0]
        )

    def test_lighthouse_timeout_does_not_spin(self) -> None:
        # The lighthouse raises the builtin TimeoutError at its deadline,
        # which on Python 3.11+ is also concurrent.futures.TimeoutError.
        store = _CountingStore()
        m = self._membership(store)
        m.client = _lighthouse(TimeoutError("deadline exceeded"))
        errors = []

        def run() -> None:
            try:
                m._quorum({}, timedelta(seconds=1))
            except TimeoutError as e:
                errors.append(e)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertLess(store.checks, 3)

        # next_assignment backs off and retries.
        m.client = _lighthouse(
            TimeoutError("deadline exceeded"), _quorum(1, 10**9, self._members())
        )
        self.assertEqual(m.next_assignment().seq, 10**9)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [0.5])

    def test_no_assignment_backs_off(self) -> None:
        m = self._membership(dist.HashStore())
        alone = _quorum(1, 10**9, [self._spare("a")])
        quorums = [alone] * 3 + [_quorum(2, 2 * 10**9, self._members())]
        with (
            mock.patch.object(m, "_quorum", side_effect=quorums),
            self.assertLogs(membership.logger, "INFO") as logs,
        ):
            m.next_assignment()
        self.assertEqual(
            [c.args[0] for c in self.sleep.call_args_list], [0.5, 1.0, 2.0]
        )
        # Unassigned quorums are logged at most every _LOG_INTERVAL.
        quorum_logs = [l for l in logs.output if "quorum " in l]
        self.assertEqual(len(quorum_logs), 2)

    def test_done_outcome(self) -> None:
        store = dist.HashStore()
        m = self._membership(store)
        store.set(DONE_KEY, membership.DONE_FAILED)
        with self.assertRaises(UnrecoverableError):
            m.next_assignment()
        m.client = _lighthouse()
        with self.assertRaises(UnrecoverableError):
            m._quorum({}, timedelta(seconds=1))
        store.set(DONE_KEY, membership.DONE_OK)
        with self.assertRaises(TrainingFinishedError):
            m.next_assignment()

    def test_store_error_retried(self) -> None:
        store = _FlakyStore()
        m = self._membership(store)
        with mock.patch.object(
            m, "_quorum", return_value=_quorum(1, 10**9, self._members())
        ):
            m.next_assignment()
        self.assertEqual(store.failures, 1)


if __name__ == "__main__":
    unittest.main()
