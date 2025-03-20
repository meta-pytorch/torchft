import threading
from datetime import timedelta
from unittest import TestCase, skipUnless

import torch
from torch.futures import Future

from torchft.futures import (
    _TIMEOUT_MANAGER,
    context_timeout,
    future_timeout,
    future_wait,
    stream_timeout,
)


class FuturesTest(TestCase):
    def test_future_wait(self) -> None:
        # pyre-fixme[29]: Future is not a function
        fut = Future()
        with self.assertRaisesRegex(TimeoutError, "future did not complete within"):
            future_wait(fut, timeout=timedelta(seconds=0.01))

        # pyre-fixme[29]: Future is not a function
        fut = Future()
        fut.set_result(1)
        self.assertEqual(future_wait(fut, timeout=timedelta(seconds=1.0)), 1)

        # pyre-fixme[29]: Future is not a function
        fut = Future()
        fut.set_exception(RuntimeError("test"))
        with self.assertRaisesRegex(RuntimeError, "test"):
            future_wait(fut, timeout=timedelta(seconds=1.0))

    def test_future_timeout(self) -> None:
        # pyre-fixme[29]: Future is not a function
        fut = Future()
        timed_fut = future_timeout(fut, timeout=timedelta(seconds=0.01))
        with self.assertRaisesRegex(TimeoutError, "future did not complete within"):
            timed_fut.wait()

    def test_future_timeout_result(self) -> None:
        # pyre-fixme[29]: Future is not a function
        fut = Future()
        timed_fut = future_timeout(fut, timeout=timedelta(seconds=10))
        fut.set_result(1)
        self.assertEqual(timed_fut.wait(), 1)

    def test_future_timeout_exception(self) -> None:
        # pyre-fixme[29]: Future is not a function
        fut = Future()
        timed_fut = future_timeout(fut, timeout=timedelta(seconds=10))
        fut.set_exception(RuntimeError("test"))
        with self.assertRaisesRegex(RuntimeError, "test"):
            timed_fut.wait()

    def test_context_timeout(self) -> None:
        barrier: threading.Barrier = threading.Barrier(2)

        def callback() -> None:
            barrier.wait()

        with context_timeout(callback, timedelta(seconds=0.01)):
            # block until timeout fires
            barrier.wait()

        def fail() -> None:
            self.fail("timeout should be cancelled")

        with context_timeout(fail, timedelta(seconds=10)):
            pass

    # pyre-fixme[56]: Pyre was not able to infer the type of decorator
    @skipUnless(torch.cuda.is_available(), "CUDA is required for this test")
    def test_stream_timeout(self) -> None:
        torch.cuda.synchronize()

        def callback() -> None:
            self.fail()

        stream_timeout(callback, timeout=timedelta(seconds=0.01))

        # make sure event completes
        torch.cuda.synchronize()

        # make sure that event is deleted on the deletion queue
        item = _TIMEOUT_MANAGER._del_queue.get(timeout=10.0)
        _TIMEOUT_MANAGER._del_queue.put(item)
        del item

        self.assertEqual(_TIMEOUT_MANAGER._clear_del_queue(), 1)
