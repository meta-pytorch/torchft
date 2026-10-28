# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torchft.fsdp import fault_tolerance
from torchft.fsdp.fault_tolerance import (
    choose_restore_step,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_RESTART,
    FaultTolerance,
    FTFSDPConfig,
    init_optim_state,
)
from torchft.fsdp.membership import (
    Assignment,
    DONE_FAILED,
    DONE_KEY,
    DONE_OK,
    MemberInfo,
    RestartRequiredError,
    STARTED_KEY,
    TrainingFinishedError,
    UnrecoverableError,
)


class InitOptimStateTest(unittest.TestCase):
    def test_matches_fresh_optimizer(self) -> None:
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4)
        ref = torch.nn.Linear(4, 4)
        ref.load_state_dict(model.state_dict())
        opt = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=0.1, weight_decay=0.1)

        init_optim_state(opt)
        self.assertEqual(len(opt.state), 2)
        for p, q in zip(model.parameters(), ref.parameters()):
            self.assertTrue(torch.equal(p, q))
            self.assertIsNone(p.grad)
            self.assertEqual(opt.state[p]["step"].item(), 0)
        self.assertEqual(opt.param_groups[0]["lr"], 0.1)

        for m, o in ((model, opt), (ref, ref_opt)):
            m(torch.ones(2, 4)).sum().backward()
            o.step()
        for p, q in zip(model.parameters(), ref.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_keeps_existing_state(self) -> None:
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters())
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        init_optim_state(opt)
        for p in model.parameters():
            self.assertEqual(opt.state[p]["step"].item(), 1)

    def test_rejects_existing_grads(self) -> None:
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters())
        model(torch.ones(2, 4)).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "before the first backward"):
            init_optim_state(opt)

    def test_rejects_partial_state(self) -> None:
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters())
        model.weight.grad = torch.ones_like(model.weight)
        opt.step()
        with self.assertRaisesRegex(RuntimeError, "1 of 2"):
            init_optim_state(opt)

    def test_matches_fresh_with_weight_decay(self) -> None:
        for make in (
            lambda ps: torch.optim.Adam(ps, lr=0.1, weight_decay=0.1),
            lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.1),
        ):
            torch.manual_seed(0)
            model = torch.nn.Linear(4, 4)
            ref = torch.nn.Linear(4, 4)
            ref.load_state_dict(model.state_dict())
            opt, ref_opt = make(model.parameters()), make(ref.parameters())
            init_optim_state(opt)
            for _ in range(2):
                for m, o in ((model, opt), (ref, ref_opt)):
                    o.zero_grad()
                    m(torch.ones(2, 4)).sum().backward()
                    o.step()
            for p, q in zip(model.parameters(), ref.parameters()):
                torch.testing.assert_close(p, q)

    def test_stateless_optimizer(self) -> None:
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        init_optim_state(opt)
        self.assertFalse(any(opt.state.values()))


class FTFSDPConfigTest(unittest.TestCase):
    def test_validation(self) -> None:
        FTFSDPConfig(num_active_hosts=2, num_hosts=2)
        with self.assertRaisesRegex(ValueError, "num_active_hosts"):
            FTFSDPConfig(num_active_hosts=1, num_hosts=2)
        with self.assertRaisesRegex(ValueError, "num_hosts"):
            FTFSDPConfig(num_active_hosts=3, num_hosts=2)
        with self.assertRaisesRegex(ValueError, "max_failures_per_step"):
            FTFSDPConfig(max_failures_per_step=0)


def _bare_ft(**config) -> FaultTolerance:
    ft = object.__new__(FaultTolerance)
    ft.config = FTFSDPConfig(**config)
    ft.current_step = 0
    ft.num_recoveries = 0
    ft._failed_step = -1
    ft._num_failures_at_step = 0
    ft.store = dist.HashStore()
    ft.finished = False
    ft.failed = False
    ft.exit_on_done = False
    ft.snapshotter = None
    ft.transports = mock.Mock()
    ft.membership = mock.Mock()
    ft._assignment = mock.Mock()
    return ft


def _done(ft: FaultTolerance) -> str | None:
    if not ft.store.check([DONE_KEY]):
        return None
    return ft.store.get(DONE_KEY).decode()


class RecoverTest(unittest.TestCase):
    def test_gives_up_without_progress(self) -> None:
        ft = _bare_ft(max_failures_per_step=2)
        err = RuntimeError("boom")
        with mock.patch.object(ft, "_recover_loop", return_value=0) as loop:
            ft.current_step = 5
            ft.recover(err)
            ft.recover(err)
            self.assertIsNone(_done(ft))
            with self.assertRaises(RuntimeError):
                ft.recover(err)
            self.assertEqual(loop.call_count, 2)
            self.assertEqual(_done(ft), DONE_FAILED)
            ft.current_step = 6
            ft.recover(err)
            self.assertEqual(loop.call_count, 3)

    def test_retries_failed_recovery(self) -> None:
        ft = _bare_ft(max_recoveries=3)
        assignment = mock.Mock()
        attempts = []

        def recover(a) -> None:
            attempts.append(a)
            if len(attempts) < 3:
                raise RuntimeError("store timeout")

        with (
            mock.patch.object(ft, "_recover", side_effect=recover),
            mock.patch.object(ft, "_handle_failure", return_value="next"),
        ):
            self.assertEqual(ft._recover_loop(assignment, None), 0)
        self.assertEqual(attempts, [assignment, "next", "next"])
        self.assertEqual(ft.num_recoveries, 2)

    def test_retry_limit(self) -> None:
        ft = _bare_ft(max_recoveries=1)
        with (
            mock.patch.object(ft, "_recover", side_effect=RuntimeError("x")),
            mock.patch.object(ft, "_handle_failure", return_value="next"),
        ):
            with self.assertRaisesRegex(RuntimeError, "x"):
                ft._recover_loop(mock.Mock(), None)
        self.assertEqual(ft.num_recoveries, 2)
        self.assertEqual(_done(ft), DONE_FAILED)

    def test_unrecoverable_not_retried(self) -> None:
        ft = _bare_ft(max_recoveries=3)
        with mock.patch.object(
            ft, "_recover", side_effect=UnrecoverableError("lost")
        ) as recover:
            with self.assertRaises(UnrecoverableError):
                ft._recover_loop(mock.Mock(), None)
        self.assertEqual(recover.call_count, 1)
        self.assertEqual(_done(ft), DONE_FAILED)

    def test_restart_not_marked_failed(self) -> None:
        ft = _bare_ft()
        ft.membership = membership = mock.Mock()
        with mock.patch.object(
            ft, "_recover", side_effect=RestartRequiredError("cuda")
        ):
            with self.assertRaises(RestartRequiredError):
                ft._recover_loop(mock.Mock(), None)
        self.assertIsNone(_done(ft))
        # Peers abandon the recovery instead of waiting for this process.
        membership.request_recovery_once.assert_called_once()

    def test_restore_keeps_fetched_replica(self) -> None:
        for has_state in (True, False):
            ft = _bare_ft()
            ft._states = {}
            ft.model_parts = []
            ft.optimizers = []
            ft.snapshotter = snap = mock.Mock()
            snap.restore.return_value = b"meta"
            succ = _avail()
            succ["replica"] = {"4": [0, 2]}
            avail = [_avail(has_state=has_state), succ]
            with (
                mock.patch.object(dist, "get_rank", return_value=0),
                mock.patch.object(dist, "get_world_size", return_value=2),
                mock.patch.object(ft, "_load_meta"),
            ):
                ft._restore_from_snapshots(4, avail)
            self.assertEqual(snap.fetch_replica.called, not has_state)
            snap.replicate_now.assert_called_once_with(4, fetched=not has_state)

    def test_oversized_meta_fails_fast(self) -> None:
        ft = _bare_ft(meta_capacity_bytes=16)
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, "meta_capacity_bytes"):
            ft.attach(model_parts=[model], optimizers=[opt])
        self.assertEqual(_done(ft), DONE_FAILED)

    def test_attach_errors_mark_failed(self) -> None:
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters())
        ft = _bare_ft()
        ft._assignment = None
        with self.assertRaisesRegex(RuntimeError, "called once"):
            ft.attach(model_parts=[model], optimizers=[opt])
        self.assertEqual(_done(ft), DONE_FAILED)

        ft = _bare_ft()
        model(torch.ones(2, 4)).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "first backward"):
            ft.attach(model_parts=[model], optimizers=[opt])
        self.assertEqual(_done(ft), DONE_FAILED)

    def test_fresh_start_resumes_before_capture(self) -> None:
        ft = _bare_ft()
        ft._started = False
        ft._states = {}
        ft.model_parts = []
        ft.optimizers = []
        ft.snapshotter = snap = mock.Mock()
        ft.membership = mock.Mock(host="a")
        assignment = mock.Mock(
            initial=True, seq=1, ranks=[_member("a")], new_hosts=set(), gen=0
        )
        with (
            mock.patch.object(dist, "get_rank", return_value=0),
            mock.patch.object(dist, "get_world_size", return_value=1),
            mock.patch.object(
                ft, "_exchange_avail", return_value=[_avail(started=False)]
            ),
        ):
            ft._recover(assignment)
        calls = [c[0] for c in snap.method_calls]
        self.assertLess(calls.index("resume"), calls.index("capture"))
        # Replicated on this thread so the transfer polls recovery_pending.
        self.assertFalse(snap.capture.call_args.kwargs["replicate"])
        self.assertLess(calls.index("flush"), calls.index("replicate_now"))
        snap.replicate_now.assert_called_once_with(0)


def _member(host: str, initialized: bool = False) -> MemberInfo:
    return MemberInfo(host, 0, 0, f"{host}/0", "h", 0, 1, initialized=initialized)


class ReplayInitTest(unittest.TestCase):
    """Two hosts with one process each; this process is rank 0."""

    def _replays(self, me: bool, peer: bool, new_hosts=()) -> bool:
        ft = _bare_ft()
        ft._started = True
        ft.snapshotter = mock.Mock()
        ft.membership = mock.Mock(host="a", recovery_pending=lambda: False)
        ft._replay_init = replay = mock.Mock()
        assignment = Assignment(
            gen=2,
            ranks=(_member("a", me), _member("b", peer)),
            host_slots={"a": 0, "b": 1},
            evict=frozenset(),
            new_hosts=frozenset(new_hosts),
            seq=5,
        )
        ft.store.set("ftfsdp/restored/5/1", "1")
        with (
            mock.patch.object(dist, "get_rank", return_value=0),
            mock.patch.object(dist, "get_world_size", return_value=2),
            mock.patch.object(dist, "set_timeout"),
            mock.patch.object(ft, "_exchange_avail", return_value=[]),
            mock.patch.object(fault_tolerance, "choose_restore_step", return_value=3),
            mock.patch.object(ft, "_restore_from_snapshots"),
        ):
            ft._recover(assignment)
        return replay.called

    def test_survivor_replays_for_new_host(self) -> None:
        self.assertTrue(self._replays(True, False, new_hosts=["b"]))

    def test_replays_for_uninitialized_slot_holder(self) -> None:
        # b kept its slot after its reconfigure failed in the constructor.
        self.assertTrue(self._replays(True, False))

    def test_uninitialized_runs_real_init(self) -> None:
        self.assertFalse(self._replays(False, True, new_hosts=["a"]))
        # a failed to reconfigure before, so it is not a new host.
        self.assertFalse(self._replays(False, True))

    def test_no_replay_once_all_initialized(self) -> None:
        # E.g. b's first recovery failed after attach.
        self.assertFalse(self._replays(True, True, new_hosts=["b"]))

    def test_attach_marks_initialized(self) -> None:
        ft = _bare_ft(meta_capacity_bytes=16)
        ft.membership = mock.Mock(initialized=False)
        model = torch.nn.Linear(4, 4)
        with self.assertRaises(ValueError):
            ft.attach(
                model_parts=[model], optimizers=[torch.optim.AdamW(model.parameters())]
            )
        self.assertTrue(ft.membership.initialized)


class ExitTest(unittest.TestCase):
    def _close(self, ft: FaultTolerance, err: BaseException | None = None) -> int:
        ft.exit_on_done = True
        with (
            mock.patch.object(ft.membership, "close") as close_membership,
            mock.patch.object(
                fault_tolerance, "_exit_process", side_effect=SystemExit
            ) as exit_process,
        ):
            try:
                try:
                    if err is not None:
                        raise err
                finally:
                    ft.close()
            except SystemExit:
                pass
        close_membership.assert_called_once()
        return exit_process.call_args.args[0]

    def test_codes(self) -> None:
        self.assertEqual(self._close(_bare_ft()), EXIT_RESTART)
        self.assertEqual(self._close(_bare_ft(), TrainingFinishedError()), EXIT_OK)
        self.assertEqual(self._close(_bare_ft(), UnrecoverableError()), EXIT_FAILED)
        failed = _bare_ft()
        failed.failed = True
        self.assertEqual(self._close(failed, RuntimeError("boom")), EXIT_FAILED)
        finished = _bare_ft()
        finished.finished = True
        self.assertEqual(self._close(finished), EXIT_OK)

    def test_logs_in_flight_exception(self) -> None:
        with self.assertLogs(fault_tolerance.logger, "ERROR") as logs:
            code = self._close(_bare_ft(), RuntimeError("boom"))
        self.assertEqual(code, EXIT_RESTART)
        self.assertIn("RuntimeError: boom", logs.output[0])

    def test_no_exit_by_default(self) -> None:
        ft = _bare_ft()
        ft.transports = transports = mock.Mock()
        with mock.patch.object(fault_tolerance, "_exit_process") as exit_process:
            ft.close()
        exit_process.assert_not_called()
        transports.close.assert_called_once()

    def test_finish_sets_ok(self) -> None:
        ft = _bare_ft()
        ft.snapshotter = mock.Mock()
        # Set before the flush, which can hang if a peer died.
        ft.snapshotter.flush.side_effect = lambda: self.assertEqual(_done(ft), DONE_OK)
        ft.finish()
        self.assertEqual(_done(ft), DONE_OK)
        self.assertTrue(ft.finished)

    def test_finish_keeps_failed(self) -> None:
        ft = _bare_ft()
        ft.snapshotter = mock.Mock()
        ft.store.set(DONE_KEY, DONE_FAILED)
        ft.finish()
        self.assertEqual(_done(ft), DONE_FAILED)

    def test_finish_retries_store(self) -> None:
        ft = _bare_ft()
        ft.snapshotter = mock.Mock()
        store = ft.store
        calls = []

        def compare_set(*args):
            calls.append(args)
            if len(calls) == 1:
                raise dist.DistNetworkError("connection reset")
            return store.compare_set(*args)

        ft.store = mock.Mock(compare_set=compare_set)
        ft.finish()
        self.assertEqual(len(calls), 2)
        self.assertEqual(store.get(DONE_KEY).decode(), DONE_OK)

    def test_mark_failed_keeps_ok(self) -> None:
        ft = _bare_ft()
        ft.store.set(DONE_KEY, DONE_OK)
        ft._mark_failed()
        self.assertTrue(ft.failed)
        self.assertEqual(_done(ft), DONE_OK)


def _avail(started=True, has_state=True, local=(), replica=()) -> dict:
    return {
        "started": started,
        "has_state": has_state,
        "local": list(local),
        "replica": {str(s): 1 for s in replica},
    }


class ChooseRestoreStepTest(unittest.TestCase):
    """Two hosts with one process each; rank r holds the replica of r - 1."""

    def test_fresh_start(self) -> None:
        avail = [_avail(False, False), _avail(False, False)]
        self.assertIsNone(choose_restore_step(avail, 1))

    def test_started_without_snapshots(self) -> None:
        # A relaunch: STARTED_KEY is set but no process holds a snapshot.
        avail = [_avail(True, False), _avail(True, False)]
        with self.assertRaisesRegex(UnrecoverableError, "relaunched"):
            choose_restore_step(avail, 1)

    def test_survivor_and_new_host(self) -> None:
        avail = [
            _avail(has_state=False),
            _avail(local=[4, 5], replica=[3, 4]),
        ]
        self.assertEqual(choose_restore_step(avail, 1), 4)

    def test_new_host_retry(self) -> None:
        # Rank 0's first recovery failed after it fetched step 4, so it is no
        # longer a new host but still has no training state.
        avail = [
            _avail(has_state=False, local=[4], replica=[5]),
            _avail(local=[4, 5], replica=[3, 4]),
        ]
        self.assertEqual(choose_restore_step(avail, 1), 4)

    def test_no_common_step(self) -> None:
        avail = [
            _avail(has_state=False),
            _avail(local=[5], replica=[3]),
        ]
        with self.assertRaisesRegex(UnrecoverableError, "no snapshot step"):
            choose_restore_step(avail, 1)


class ExchangeAvailTest(unittest.TestCase):
    def _exchange(self, store: dist.Store, started: bool) -> dict:
        ft = _bare_ft()
        ft.store = store
        ft._started = started
        ft.membership = mock.Mock(recovery_pending=lambda: False)
        snap = mock.Mock(committed_steps=lambda: [], replica_steps=lambda: {})
        with (
            mock.patch.object(ft, "_snap", return_value=snap),
            mock.patch.object(dist, "get_rank", return_value=0),
            mock.patch.object(dist, "get_world_size", return_value=1),
        ):
            (avail,) = ft._exchange_avail(7)
        return avail

    def test_fresh_start(self) -> None:
        avail = self._exchange(dist.HashStore(), started=False)
        self.assertFalse(avail["started"])
        self.assertFalse(avail["has_state"])

    def test_started_key(self) -> None:
        store = dist.HashStore()
        store.set(STARTED_KEY, "1")
        avail = self._exchange(store, started=False)
        self.assertTrue(avail["started"])
        self.assertFalse(avail["has_state"])
        self.assertTrue(store.check(["ftfsdp/avail/7/0"]))


if __name__ == "__main__":
    unittest.main()
