# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import Mock

import torch
import torch.distributed as dist
from torch import nn
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

from torchft.manager import Manager
from torchft.process_group import ProcessGroupGloo


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
        pg: ProcessGroupGloo = Mock(spec=ProcessGroupGloo)
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(dp_shard, tp),
            mesh_dim_names=("dp_shard", "tp"),
        )
        manager.num_participants.return_value = 1
        model = nn.Linear(128, 128).cuda()
        batch = torch.randn(4, 128).cuda()

        fsdp_mesh = device_mesh["dp_shard"]

        def all_reduce_hook(output: torch.Tensor) -> None:
            dist.all_reduce(output, group=pg, op=ReduceOp.AVG)

        def apply_set_all_reduce_hook(m: nn.Module) -> None:
            assert isinstance(m, FSDPModule)
            m.set_all_reduce_hook(all_reduce_hook)

        if tp > 1:
            tp_mesh = device_mesh["tp"]
            model = parallelize_module(
                model,
                tp_mesh,
                ColwiseParallel(),
            )
        shard_model = fully_shard(model, mesh=fsdp_mesh)
        shard_model.apply(apply_set_all_reduce_hook)
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
