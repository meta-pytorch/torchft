import asyncio
import threading
from datetime import timedelta
from typing import Callable, Optional, TypeVar
from unittest.mock import Mock

import torch
from torch.futures import Future

T = TypeVar("T")


class _TimerHandle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lock.acquire()
        self._timer_handle: Optional[asyncio.TimerHandle] = None

    def set_timer(self, timer_handle: asyncio.TimerHandle) -> None:
        assert self._lock.locked()
        self._timer_handle = timer_handle
        self._lock.release()

    def cancel(self) -> None:
        with self._lock:
            assert self._timer_handle is not None
            self._timer_handle.cancel()
            self._timer_handle = None


class _TimeoutManager:
    """
    This class manages timeouts for futures. It uses a background thread with an
    event loop to schedule the timeouts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_loop_thread: Optional[threading.Thread] = None
        self._next_timer_id = 0

    def _maybe_start_event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Start the event loop if it has not already been started.
        """
        with self._lock:
            if self._event_loop is None:
                self._event_loop = asyncio.new_event_loop()
                self._event_loop_thread = threading.Thread(
                    target=self._event_loop.run_forever,
                    daemon=True,
                    name="TimeoutManager",
                )
                self._event_loop_thread.start()
            # pyre-fixme[7]: optional
            return self._event_loop

    def shutdown(self) -> None:
        """
        Shutdown the event loop and cancel all pending timeouts.
        """
        with self._lock:
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)
                assert self._event_loop_thread is not None
                self._event_loop_thread.join()
                self._event_loop = None
                self._event_loop_thread = None

    def register(self, fut: Future[T], timeout: timedelta) -> Future[T]:
        """
        Registers a future that will be cancelled after the specified timeout.
        """
        # bypass timeout for mock futures
        if isinstance(fut, Mock):
            return fut

        loop = self._maybe_start_event_loop()

        # pyre-fixme[29]: Future is not a function
        timed_fut: Future[T] = Future()
        handle: _TimerHandle = _TimerHandle()
        # pyre-fixme[6]: *args
        loop.call_soon_threadsafe(self._register, loop, timed_fut, timeout, handle)

        def callback(fut: Future[T]) -> None:
            handle.cancel()
            try:
                timed_fut.set_result(fut.wait())
            except Exception as e:
                try:
                    # this can throw if the future is already done
                    # pyre-fixme[6]: e is not T
                    timed_fut.set_exception(e)
                except Exception:
                    pass

        fut.add_done_callback(callback)
        return timed_fut

    def stream_timeout(self, callback: Callable[[], None], timeout: timedelta) -> None:
        loop = self._maybe_start_event_loop()

        event = torch.cuda.Event()
        event.record()

        def handler() -> None:
            if not event.query():
                callback()

        loop.call_soon_threadsafe(self._register_handler, loop, handler, timeout)

    @classmethod
    def _register(
        cls,
        loop: asyncio.AbstractEventLoop,
        fut: Future[T],
        timeout: timedelta,
        handle: _TimerHandle,
    ) -> None:
        timer_handle = loop.call_later(
            timeout.total_seconds(),
            lambda: fut.set_exception(
                # pyre-fixme[6]: e is not T
                TimeoutError(f"future did not complete within {timeout}")
            ),
        )
        handle.set_timer(timer_handle)

    @classmethod
    def _register_handler(
        cls,
        loop,
        handler: Callable[[], None],
        timeout: timedelta,
    ) -> None:
        loop.call_later(
            timeout.total_seconds(),
            handler,
        )


_TIMEOUT_MANAGER = _TimeoutManager()


def future_timeout(fut: Future[T], timeout: timedelta) -> Future[T]:
    """
    Return a Future that completes with the result of the given Future within
    the given timeout or with a TimeoutError.

    Args:
        fut: The Future to wait for
        timeout: The timeout to wait for the Future to complete

    Returns:
        The future with a timeout
    """
    return _TIMEOUT_MANAGER.register(fut, timeout)


def future_wait(fut: Future[T], timeout: timedelta) -> T:
    """
    Wait for a Future to complete up to a timeout.

    Args:
        fut: The Future to wait for
        timeout: The timeout to wait for the Future to complete

    Returns:
        The result of the Future if it completed within the timeout.

    Raises:
        TimeoutError if the Future did not complete within the timeout.
        Any other exception that occurred in the Future.
    """

    event: threading.Event = threading.Event()

    def callback(fut: Future[T]) -> T:
        event.set()
        return fut.wait()

    fut = fut.then(callback)

    if not event.wait(timeout=timeout.total_seconds()):
        raise TimeoutError(f"future did not complete within {timeout}")

    return fut.wait()


def stream_timeout(callback: Callable[[], None], timeout: timedelta) -> None:
    """
    Registers a callback that will be called after the specified timeout if
    the current stream doesn't complete in time.

    This uses a cuda Event to track the completion of the current stream. If
    the stream is not complete after the timeout, the callback is called.

    Args:
        callback: The callback to call if the stream doesn't complete in time.
        timeout: The timeout to wait for the stream to complete.
    """
    _TIMEOUT_MANAGER.stream_timeout(callback, timeout)
