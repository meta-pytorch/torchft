# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import torch
from torchft.parameter_server import ParameterServer
from torchft.process_group import ProcessGroup, ProcessGroupGloo


class MyParameterServer(ParameterServer):
    def __init__(self) -> None:
        super().__init__(port=0)

    @classmethod
    def new_process_group(cls) -> ProcessGroup:
        return ProcessGroupGloo()

    def forward(self, session_id: str, pg: ProcessGroup) -> None:
        data = torch.zeros(1)
        pg.broadcast_one(data, root=1).wait()

        data += 23

        pg.broadcast_one(data, root=0).wait()


class TestParameterServer(TestCase):
    def test_parameter_server(self) -> None:
        ps = MyParameterServer()

        addr = ps.address()
        pg = MyParameterServer.new_session(addr)

        data = torch.zeros(1)
        data += 12
        # send to server (0) from client (1)
        pg.broadcast_one(data, root=1).wait()

        # recv from server (0) to client (1)
        pg.broadcast_one(data, root=0).wait()
        self.assertEqual(data[0].item(), 12 + 23)
