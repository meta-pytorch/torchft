# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import unittest
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Tuple
from unittest.mock import Mock

import torch
import torch.distributed as dist
from torch import nn
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    BroadcastOptions,
    ReduceOp,
    _resolve_process_group,
)
from torch.distributed import (
    ReduceOp,
    TCPStore,
    Work,
    _functional_collectives,
    get_world_size,
)
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from torchft.manager import Manager
from torchft.process_group import (
    ManagedProcessGroup,
    ProcessGroupGloo,
    ft_init_device_mesh,
)


class FSDPTest(unittest.TestCase):
    @staticmethod
    def _test_fsdp(
        world_size: int,
        rank: int,
        dp_replicate: int = 2,
        dp_shard: int = 2,
        tp: int = 1,
    ) -> None:
        torch.cuda.set_device(rank)

        group_size = world_size // dp_replicate
        group = rank // group_size
        group_rank = rank % group_size

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(12346 + group)
        os.environ["RANK"] = str(group_rank)
        os.environ["WORLD_SIZE"] = str(group_size)

        manager = Mock(spec=Manager)
        manager._pg = ProcessGroupGloo()
        device_mesh = ft_init_device_mesh(
            device_type="cuda",
            mesh_shape=(dp_replicate, dp_shard, tp),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
            replicate_dim=0,
            manager=manager,
        )
        manager.num_participants.return_value = 1
        model = nn.Linear(128, 128).cuda()
        batch = torch.randn(4, 128).cuda()

        fsdp_mesh = device_mesh["dp_replicate", "dp_shard"]

        if tp > 1:
            tp_mesh = device_mesh["tp"]
            model = parallelize_module(
                model,
                tp_mesh,
                ColwiseParallel(),
            )
        shard_model = fully_shard(model, mesh=fsdp_mesh)
        shard_model(batch).mean().backward()

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(torch.cuda.device_count() < 4, "Not enough GPUs")
    def test_fsdp(self) -> None:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
            futures = []
            for i in range(4):
                future = executor.submit(self._test_fsdp, 4, i)
                futures.append(future)

            for fut in futures:
                fut.result()

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(torch.cuda.device_count() < 4, "Not enough GPUs")
    def test_fsdp_tp(self) -> None:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
            futures = []
            for i in range(4):
                future = executor.submit(
                    self._test_fsdp, 4, i, dp_replicate=1, dp_shard=2, tp=2
                )
                futures.append(future)

            for fut in futures:
                fut.result()
