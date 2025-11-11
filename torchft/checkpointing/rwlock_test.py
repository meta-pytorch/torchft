# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchft.checkpointing._rwlock import RWLock


def test_w_locked() -> None:
    lock = RWLock()

    with lock.w_lock():
        assert lock.w_locked()
    assert not lock.w_locked()


def test_w_lock_timeout() -> None:
    lock = RWLock(timeout=0.01)

    lock.r_acquire()
    lock.r_acquire()

    with pytest.raises(TimeoutError):
        lock.w_acquire()

    with pytest.raises(TimeoutError):
        with lock.w_lock():
            pass

    lock.r_release()
    with pytest.raises(TimeoutError):
        lock.w_acquire()

    lock.r_release()
    with lock.w_lock():
        pass
    lock.w_acquire()


def test_r_lock_timeout() -> None:
    lock = RWLock(timeout=0.01)

    lock.w_acquire()

    with pytest.raises(TimeoutError):
        lock.r_acquire()

    with pytest.raises(TimeoutError):
        with lock.r_lock():
            pass

    lock.w_release()
    with lock.r_lock():
        pass
    lock.r_acquire()
