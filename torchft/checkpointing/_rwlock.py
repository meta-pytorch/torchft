# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""rwlock.py

Adapted from: https://github.com/tylerneylon/rwlock/blob/main/rwlock.py

A class to implement read-write locks on top of the standard threading
library.

This is implemented with two mutexes (threading.Lock instances) as per this
wikipedia pseudocode:

https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock#Using_two_mutexes

__________________________
License info (MIT):

*******

Copyright 2023 Tyler Neylon and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*******
"""

from contextlib import contextmanager
from threading import Lock
from typing import Generator


class RWLock(object):
    """RWLock class; this is meant to allow an object to be read from by
    multiple threads, but only written to by a single thread at a time. See:
    https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock

    All operations are timed and will throw TimeoutError if the timeout is
    exceeded.

    Usage:

        from rwlock import RWLock

        my_obj_rwlock = RWLock(timeout=60.0)

        # When reading from my_obj:
        with my_obj_rwlock.r_lock():
            do_read_only_things_with(my_obj)

        # When writing to my_obj:
        with my_obj_rwlock.w_lock():
            mutate(my_obj)
    """

    def __init__(self, timeout: float = -1) -> None:
        self.timeout = timeout

        self._w_lock = Lock()
        self._num_r_lock = Lock()
        self._num_r = 0

    # ___________________________________________________________________
    # Reading methods.

    def r_acquire(self) -> None:
        if not self._num_r_lock.acquire(timeout=self.timeout):
            raise TimeoutError(
                f"Timed out waiting for rlock after {self.timeout} seconds"
            )

        self._num_r += 1
        if self._num_r == 1:
            if not self._w_lock.acquire(timeout=self.timeout):
                self._num_r -= 1
                self._num_r_lock.release()
                raise TimeoutError(
                    f"Timed out waiting for wlock after {self.timeout} seconds"
                )

        self._num_r_lock.release()

    def r_release(self) -> None:
        assert self._num_r > 0
        self._num_r_lock.acquire()
        self._num_r -= 1
        if self._num_r == 0:
            self._w_lock.release()
        self._num_r_lock.release()

    @contextmanager
    def r_lock(self) -> Generator[None, None, None]:
        """This method is designed to be used via the `with` statement."""
        self.r_acquire()
        try:
            yield
        finally:
            self.r_release()

    # ___________________________________________________________________
    # Writing methods.

    def w_acquire(self) -> None:
        if not self._w_lock.acquire(timeout=self.timeout):
            raise TimeoutError(
                f"Timed out waiting for wlock after {self.timeout} seconds"
            )

    def w_release(self) -> None:
        self._w_lock.release()

    @contextmanager
    def w_lock(self) -> Generator[None, None, None]:
        """This method is designed to be used via the `with` statement."""
        self.w_acquire()
        try:
            yield
        finally:
            self.w_release()

    def w_locked(self) -> bool:
        """Returns True if the lock is currently locked for reading."""
        return self._w_lock.locked()
