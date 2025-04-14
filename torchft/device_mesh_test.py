# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
from concurrent.futures import ProcessPoolExecutor
from typing import cast
from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.distributed as dist

from torchft.manager import Manager
from torchft.process_group import (
    ManagedProcessGroup,
    ProcessGroupGloo,
    ft_init_device_mesh,
)


class DeviceMeshTest(TestCase):
    @staticmethod
    def _test_init_device_mesh(world_size: int, rank: int) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(12346)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(4)

        testcase = TestCase()

        manager = Mock(spec=Manager)
        manager._pg = ProcessGroupGloo()
        # Even though we only have 4 workers, we can still initialize (2, 4) mesh.
        # That's because the replicate group is NOT phystically created in the
        # real mesh but is virtually added to the mesh via ManagedDeviceMesh.
        device_mesh = ft_init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, world_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
            replicate_dim=0,
            manager=manager,
        )

        testcase.assertTrue(
            isinstance(device_mesh.get_group("dp_replicate"), ManagedProcessGroup)
        )
        testcase.assertTrue(
            not isinstance(device_mesh.get_group("dp_shard"), ManagedProcessGroup)
        )
        replicate_group = device_mesh.get_group("dp_replicate")
        testcase.assertEqual(
            cast(ManagedProcessGroup, replicate_group)._manager, manager
        )
        replicate_mesh = device_mesh["dp_replicate"]
        testcase.assertEqual(replicate_mesh.get_group(), replicate_group)

        flatten_mesh = device_mesh._flatten("dp")
        manager.num_participants.return_value = 0
        testcase.assertEqual(flatten_mesh.size(), world_size)
        manager.num_participants.return_value = 1
        testcase.assertEqual(flatten_mesh.size(), world_size)
        manager.num_participants.return_value = 2
        testcase.assertEqual(flatten_mesh.size(), world_size * 2)

        testcase.assertEqual(flatten_mesh.get_local_rank(), dist.get_rank())

        device_mesh.get_coordinate()
        buffer = io.BytesIO()
        torch.save(device_mesh, buffer)
        buffer.seek(0)
        torch.load(buffer, weights_only=False)

    def test_init_device_mesh(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                future = executor.submit(self._test_init_device_mesh, 4, i)
                futures.append(future)
            for f in futures:
                f.result()

    def test_repr_hash(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(12346)
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)

        manager = Mock(spec=Manager)
        manager._pg = ProcessGroupGloo()

        for container in [tuple, list]:
            device_mesh = ft_init_device_mesh(
                device_type="cpu",
                mesh_shape=container((1, 1)),
                mesh_dim_names=container((f"dp_replicate_{container}", "dp_shard")),
                replicate_dim=0,
                manager=manager,
            )

            self.assertIsInstance(repr(device_mesh), str)
            self.assertIsInstance(str(device_mesh), str)
            self.assertEqual(hash(device_mesh), hash(device_mesh))
            self.assertIsInstance(hash(device_mesh), int)

            dist.destroy_process_group()
