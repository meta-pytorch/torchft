# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import threading
import time
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torchft.fsdp import snapshot
from torchft.fsdp.membership import RestartRequiredError


class _FakeTransport:
    _ids = itertools.count()

    def __init__(self) -> None:
        self.id = next(self._ids)
        self.peer: bytes | None = None
        self.closed = False

    def bind(self, *, timeout: float) -> bytes:
        return f"agent{self.id}".encode()

    def connect(self, peer_url: bytes, *, timeout: float) -> int:
        self.peer = peer_url
        return 0

    def close(self, *, timeout: float) -> None:
        self.closed = True


class TransportPoolTest(unittest.TestCase):
    def test_get_refills(self) -> None:
        created = []

        def new():
            created.append(_FakeTransport())
            return created[-1]

        with mock.patch.object(snapshot, "_new_nixl_transport", new):
            pool = snapshot.TransportPool(2)
            first = pool.get()
            second = pool.get()
            pool._ready.get().result()
        self.assertIsNot(first, second)
        self.assertEqual(created[:2], [first, second])
        # Each get queues a replacement.
        self.assertEqual(len(created), 4)

    def test_close_closes_unused(self) -> None:
        created = []
        release = threading.Event()

        def new():
            if len(created) == 2:
                # Still being created when the pool closes.
                release.wait(5.0)
            created.append(_FakeTransport())
            return created[-1]

        with mock.patch.object(snapshot, "_new_nixl_transport", new):
            pool = snapshot.TransportPool(2)
            taken = pool.get()
            while len(created) < 2:
                time.sleep(0.01)
            pool.close()
            release.set()
            pool._executor.shutdown(wait=True)
        self.assertFalse(taken.closed)
        self.assertEqual(len(created), 3)
        self.assertTrue(all(t.closed for t in created[1:]))

    def test_invalid_size(self) -> None:
        with self.assertRaises(ValueError):
            snapshot.TransportPool(0)

    def test_creation_error_raised_on_get(self) -> None:
        def new():
            raise RuntimeError("no NICs")

        with mock.patch.object(snapshot, "_new_nixl_transport", new):
            pool = snapshot.TransportPool(1)
            with self.assertRaisesRegex(RuntimeError, "no NICs"):
                pool.get()


class BootstrapTest(unittest.TestCase):
    def test_exchanges_metadata(self) -> None:
        store = dist.HashStore()
        a, b = _FakeTransport(), _FakeTransport()
        t = threading.Thread(
            target=snapshot._bootstrap,
            args=(b, store),
            kwargs={
                "rank": 1,
                "peer_rank": 0,
                "timeout": 5.0,
                "abort": lambda: False,
            },
        )
        t.start()
        snapshot._bootstrap(
            a, store, rank=0, peer_rank=1, timeout=5.0, abort=lambda: False
        )
        t.join()
        self.assertEqual(a.peer, b.bind(timeout=1.0))
        self.assertEqual(b.peer, a.bind(timeout=1.0))

    def test_closes_on_timeout(self) -> None:
        tr = _FakeTransport()
        with self.assertRaises(TimeoutError):
            snapshot._bootstrap(
                tr,
                dist.HashStore(),
                rank=0,
                peer_rank=1,
                timeout=0.1,
                abort=lambda: False,
            )
        self.assertTrue(tr.closed)

    def test_closes_on_abort(self) -> None:
        tr = _FakeTransport()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            snapshot._bootstrap(
                tr,
                dist.HashStore(),
                rank=0,
                peer_rank=1,
                timeout=60.0,
                abort=lambda: True,
            )
        self.assertTrue(tr.closed)


class WaitForKeysTest(unittest.TestCase):
    def test_returns_when_present(self) -> None:
        store = dist.HashStore()
        store.set("a", "1")
        snapshot.wait_for_keys(store, ["a"], 1.0, lambda: True)

    def test_timeout(self) -> None:
        with self.assertRaises(TimeoutError):
            snapshot.wait_for_keys(dist.HashStore(), ["a"], 0.1, lambda: False)


class UpdateLinksTest(unittest.TestCase):
    """Two ranks on one host each, linked both ways."""

    def setUp(self) -> None:
        self.closes: dict[str, mock.Mock] = {}

    def _snap(self, store: dist.Store, ident: str) -> snapshot.Snapshotter:
        s = object.__new__(snapshot.Snapshotter)
        s.procs_per_host = 1
        s._cv = threading.Condition()
        s._replicating = False
        s.timeout = 5.0
        s.store = store
        s.ident = ident
        s.recovery_pending = lambda: False
        s.link_out = s.link_in = None
        # Records the transports s closes.
        self.closes[ident] = s._close_transport = mock.Mock()
        s._connect_out = lambda store, succ, seq, timeout: snapshot._LinkOut(
            succ, seq, f"out{seq}", [], torch.empty(0), None, [], []
        )
        s._connect_in = lambda store, pred, seq, timeout: snapshot._LinkIn(
            pred, seq, f"in{seq}"
        )
        return s

    def _recover(self, snaps: list[snapshot.Snapshotter], seq: int) -> None:
        idents = [s.ident for s in snaps]
        threads = [
            threading.Thread(
                target=s.update_links,
                kwargs={"rank": r, "ident_of_rank": idents, "seq": seq, "timeout": 5.0},
            )
            for r, s in enumerate(snaps)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _transports(self, snaps: list[snapshot.Snapshotter]) -> list[object]:
        links = [link for s in snaps for link in (s.link_out, s.link_in)]
        return [getattr(link, "transport", None) for link in links]

    def test_kept_when_both_agree(self) -> None:
        store = dist.HashStore()
        snaps = [self._snap(store, "a"), self._snap(store, "b")]
        self._recover(snaps, 1)
        self._recover(snaps, 2)
        self.assertEqual(self._transports(snaps), ["out1", "in1"] * 2)
        self.closes["a"].assert_not_called()
        self.closes["b"].assert_not_called()

    def test_rebuilt_when_peer_lacks_link(self) -> None:
        store = dist.HashStore()
        snaps = [self._snap(store, "a"), self._snap(store, "b")]
        self._recover(snaps, 1)
        # b's receive side failed to connect in the previous recovery.
        snaps[1].link_in = None
        self._recover(snaps, 2)
        self.assertEqual(self._transports(snaps), ["out2", "in1", "out1", "in2"])
        self.closes["a"].assert_called_once_with("out1")
        self.closes["b"].assert_not_called()

    def test_rebuilt_when_broken(self) -> None:
        store = dist.HashStore()
        snaps = [self._snap(store, "a"), self._snap(store, "b")]
        self._recover(snaps, 1)
        link = snaps[0].link_out
        assert link is not None
        link.broken = True
        self._recover(snaps, 2)
        self.assertEqual(self._transports(snaps), ["out2", "in1", "out1", "in2"])
        self.closes["a"].assert_called_once_with("out1")
        self.closes["b"].assert_called_once_with("in1")

    def test_rebuilt_without_waiting_for_write(self) -> None:
        store = dist.HashStore()
        snaps = [self._snap(store, "a"), self._snap(store, "b")]
        self._recover(snaps, 1)
        # An abandoned write; the worker closes its link once it exits.
        snaps[0].timeout = 60.0
        snaps[0]._replicating = True
        start = time.monotonic()
        self._recover(snaps, 2)
        self.assertLess(time.monotonic() - start, 30)
        self.assertEqual(self._transports(snaps), ["out2", "in1", "out1", "in2"])
        self.closes["a"].assert_not_called()
        self.closes["b"].assert_called_once_with("in1")


def _bare_snap(timeout: float = 60.0) -> snapshot.Snapshotter:
    s = object.__new__(snapshot.Snapshotter)
    s.timeout = timeout
    s.aborted = threading.Event()
    s.comm_failed = lambda: False
    s.recovery_pending = lambda: False
    s.link_out = None
    s._cv = threading.Condition()
    return s


class WaitCopyLaunchedTest(unittest.TestCase):
    def _pending(self, s: snapshot.Snapshotter) -> threading.Event:
        job = snapshot._Job(step=3, slot=0, meta=b"", ready=mock.Mock(), epoch=0)
        s._pending = job
        return job.launched

    def test_returns_when_launched(self) -> None:
        s = _bare_snap()
        threading.Timer(0.1, self._pending(s).set).start()
        s.wait_copy_launched()
        self.assertIsNone(s._pending)

    def test_raises_on_recovery_elsewhere(self) -> None:
        s = _bare_snap()
        self._pending(s)
        pending = threading.Event()
        s.recovery_pending = pending.is_set
        threading.Timer(0.1, pending.set).start()
        start = time.monotonic()
        with self.assertRaisesRegex(dist.DistError, "abandoned for recovery"):
            s.wait_copy_launched()
        self.assertLess(time.monotonic() - start, 30)

    def test_raises_on_abort(self) -> None:
        s = _bare_snap()
        self._pending(s)
        s.abort()
        with self.assertRaises(snapshot.SnapshotStalledError):
            s.wait_copy_launched()

    def test_timeout_is_dist_error(self) -> None:
        s = _bare_snap(timeout=0.1)
        self._pending(s)
        with self.assertRaisesRegex(dist.DistError, "not launched"):
            s.wait_copy_launched()


def _work(polls: int | None) -> mock.Mock:
    """Work that completes after ``polls`` polls, or never if None."""
    count = itertools.count()
    return mock.Mock(is_completed=lambda: polls is not None and next(count) >= polls)


def _link_out(s: snapshot.Snapshotter, polls: int | None = None) -> snapshot._LinkOut:
    transport = mock.Mock(
        write=mock.Mock(side_effect=lambda *a, **kw: _work(polls)),
        read=mock.Mock(side_effect=lambda *a, **kw: _work(polls)),
    )
    link = snapshot._LinkOut(
        "b",
        1,
        transport,
        [mock.Mock()],
        torch.zeros(snapshot._HEADER_WORDS, dtype=torch.int64),
        mock.Mock(),
        [None, None],
        [None, None],
    )
    s.link_out = link
    return link


class TransferTest(unittest.TestCase):
    def _write(self, s: snapshot.Snapshotter, link: snapshot._LinkOut) -> None:
        s._transfer(link, link.transport.write, None, None, s._worker_abort)

    def test_comm_failure_aborts(self) -> None:
        s = _bare_snap()
        link = _link_out(s)
        failed = threading.Event()
        s.comm_failed = failed.is_set
        threading.Timer(0.1, failed.set).start()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            self._write(s, link)
        # The abandoned write may still own the transport.
        self.assertTrue(link.broken)

    def test_abort(self) -> None:
        s = _bare_snap()
        link = _link_out(s)
        s.abort()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            self._write(s, link)
        self.assertTrue(link.broken)

    def test_stale_link(self) -> None:
        s = _bare_snap()
        link = _link_out(s)
        s.link_out = None
        with self.assertRaises(snapshot._StaleLinkError):
            self._write(s, link)
        self.assertFalse(link.broken)


class RecoveryTransferTest(unittest.TestCase):
    """Transfers during recovery, while snapshots are still aborted."""

    def _snap(self) -> snapshot.Snapshotter:
        s = _bare_snap()
        s.interval = 2
        s.nbytes = 8
        s.slots = [snapshot._Slot(torch.zeros(8, dtype=torch.uint8), step=4)]
        s.abort()
        return s

    def test_replicate_now_while_aborted(self) -> None:
        s = self._snap()
        link = _link_out(s, polls=3)
        s.replicate_now(4)
        # Invalid header, body, valid header, then the other replica's header.
        self.assertEqual(link.transport.write.call_count, 4)
        self.assertFalse(link.broken)

    def test_replicate_now_fetched_keeps_replica(self) -> None:
        s = self._snap()
        link = _link_out(s, polls=3)
        link.remote_headers = ["h0", "h1"]
        s.replicate_now(4, fetched=True)
        # Step 4 is at index 0, so only index 1's header is invalidated.
        link.transport.write.assert_called_once()
        self.assertEqual(link.transport.write.call_args.args[1], "h1")
        self.assertEqual(link.header_src.tolist(), [0, 0, 0, 0])

    def test_replicate_now_recovery_pending(self) -> None:
        s = self._snap()
        link = _link_out(s)
        pending = threading.Event()
        s.recovery_pending = pending.is_set
        threading.Timer(0.1, pending.set).start()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            s.replicate_now(4)
        self.assertTrue(link.broken)

    def test_fetch_replica_while_aborted(self) -> None:
        s = self._snap()
        _link_out(s, polls=3)
        s.fetch_replica(6, 1, 2)
        self.assertEqual((s.slots[0].step, s.slots[0].meta_len), (6, 2))

    def test_fetch_replica_recovery_pending(self) -> None:
        s = self._snap()
        link = _link_out(s)
        s.recovery_pending = lambda: True
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            s.fetch_replica(6, 1, 2)
        self.assertTrue(link.broken)
        self.assertEqual(s.slots[0].step, -1)
        # Closed so the abandoned read cannot land later.
        link.transport.close.assert_called_once()
        self.assertIsNone(s.link_out)

    def test_fetch_replica_restarts_if_read_pending(self) -> None:
        s = self._snap()
        link = _link_out(s)
        link.transport.close.side_effect = RuntimeError("work pending")
        s.recovery_pending = lambda: True
        with self.assertRaises(RestartRequiredError):
            s.fetch_replica(6, 1, 2)
        self.assertEqual(s.slots[0].step, -1)
        self.assertIsNone(s.link_out)

    def test_replicate_now_uncommitted_raises(self) -> None:
        s = self._snap()
        _link_out(s)
        with self.assertRaisesRegex(RuntimeError, "not committed"):
            s.replicate_now(6)


class WorkerTest(unittest.TestCase):
    def _run(self, err: Exception) -> list[str]:
        s = _bare_snap()
        s.device = torch.device("cpu")
        s._busy = False
        s._closed = False
        s.slots = [snapshot._Slot(torch.zeros(1), reserved=True)]
        s._queue = [snapshot._Job(step=3, slot=0, meta=b"", ready=mock.Mock(), epoch=0)]

        def process(job) -> None:
            s._closed = True
            raise err

        s._process = process
        with (
            mock.patch.object(torch.cuda, "set_device"),
            self.assertLogs(snapshot.logger, "INFO") as logs,
        ):
            s._run()
        self.assertFalse(s.slots[0].reserved)
        return logs.output

    def test_abort_logged_without_traceback(self) -> None:
        (out,) = self._run(snapshot._TransferAbortedError("snapshot transfer aborted"))
        self.assertTrue(out.startswith("INFO:"))
        self.assertNotIn("Traceback", out)

    def test_error_logged_with_traceback(self) -> None:
        (out,) = self._run(RuntimeError("boom"))
        self.assertTrue(out.startswith("ERROR:"))
        self.assertIn("Traceback", out)


class CloseTest(unittest.TestCase):
    def _snap(self, replicating: bool) -> snapshot.Snapshotter:
        s = _bare_snap()
        s._replicating = replicating
        _link_out(s)
        s.link_in = snapshot._LinkIn("a", 1, mock.Mock())
        return s

    def test_closes_links(self) -> None:
        s = self._snap(replicating=False)
        out, in_ = s.link_out, s.link_in
        assert out is not None and in_ is not None
        s.close()
        out.transport.close.assert_called_once()
        in_.transport.close.assert_called_once()
        self.assertIsNone(s.link_out)
        self.assertIsNone(s.link_in)

    def test_skips_link_being_written(self) -> None:
        s = self._snap(replicating=True)
        out, in_ = s.link_out, s.link_in
        assert out is not None and in_ is not None
        s.close()
        # The worker's write sees the link replaced and closes it.
        out.transport.close.assert_not_called()
        in_.transport.close.assert_called_once()
        self.assertIsNone(s.link_out)


class CollectStateTensorsTest(unittest.TestCase):
    def test_includes_persistent_buffers(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        scratch = torch.zeros(1)
        model[1].register_buffer("scratch", scratch, persistent=False)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        _, tensors = snapshot.collect_state_tensors([model], [opt])
        ptrs = [t.data_ptr() for t in tensors]
        expected = {t.data_ptr() for t in model.state_dict().values()}
        self.assertEqual(len(ptrs), len(expected))
        self.assertEqual(set(ptrs), expected)
        self.assertNotIn(scratch.data_ptr(), ptrs)


if __name__ == "__main__":
    unittest.main()
