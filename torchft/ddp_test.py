# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from typing import Optional
from unittest import TestCase
from unittest.mock import create_autospec, MagicMock

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.distributed_c10d import Work
from torch.futures import Future
from torchft.ddp import DistributedDataParallel, PureDistributedDataParallel
from torchft.manager import _ManagedWork, Manager
from torchft.process_group import ProcessGroupBabyGloo, ProcessGroupGloo
from torchft.work import _DummyWork


class TestDDP(TestCase):
    def test_pure_ddp(self) -> None:
        manager = create_autospec(Manager)

        m = nn.Linear(3, 4)
        m = PureDistributedDataParallel(manager, m)

        inp = torch.rand(2, 3)
        out = m(inp)
        loss = out.mean()
        loss.backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)

        self.assertEqual(manager.allreduce.call_count, len(list(m.parameters())))

    def test_ddp(self) -> None:
        manager = create_autospec(Manager)

        call_count = 0

        def allreduce(
            tensor: torch.Tensor,
        ) -> Work:
            nonlocal call_count

            call_count += 1

            work = _DummyWork(tensor)
            return _ManagedWork(manager, work, tensor)

        manager.allreduce = allreduce

        m = nn.Linear(3, 4)
        m = DistributedDataParallel(manager, m)

        inp = torch.rand(2, 3)
        out = m(inp)
        loss = out.mean()
        loss.backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)

        self.assertGreaterEqual(call_count, 1)

    def test_ddp_failed_reduction_rejects_the_step_instead_of_raising(self) -> None:
        """When another replica dies during a step, the work's `wait` raises.
        `_ManagedWork` tells the manager and returns False, and it never builds
        its future chain, so there is no inner future for the hook to return.
        The hook must still give DDP a finished future. If it raised instead,
        backward() would fail and this replica would die too, instead of
        skipping the step and going on."""
        manager = create_autospec(Manager)
        manager._logger = MagicMock()
        manager.wrap_future = lambda fut, default, timeout=None: fut

        class _FailingWork(Work):
            def wait(self, timeout: Optional[timedelta] = None) -> bool:
                raise RuntimeError("Connection closed by peer")

            def get_future(self) -> Future[object]:
                return Future()

        manager.allreduce = lambda tensor: _ManagedWork(manager, _FailingWork(), tensor)

        m = DistributedDataParallel(manager, nn.Linear(3, 4))
        m(torch.rand(2, 3)).mean().backward()

        manager.report_error.assert_called()
        for p in m.parameters():
            self.assertIsNotNone(p.grad)

    def test_ddp_after_the_manager_errored(self) -> None:
        """After the first error, `allreduce` returns a `_DummyWork`. Its future
        is a normal future that is already done, not a managed chain."""
        manager = create_autospec(Manager)
        manager.allreduce = lambda tensor: _DummyWork(tensor)

        m = DistributedDataParallel(manager, nn.Linear(3, 4))
        m(torch.rand(2, 3)).mean().backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)
