from datetime import timedelta
from typing import Dict
from unittest import TestCase, skipUnless

import torch
from torch.distributed import TCPStore

from torchft.checkpointing.pg_transport import PGTransport
from torchft.checkpointing.transport import CheckpointTransport
from torchft.checkpointing.transport_test import run_multi_recovery_test
from torchft.process_group import ProcessGroupBabyNCCL, ProcessGroupGloo


class PGTransportTest(TestCase):
    def test_pg_transport_gloo(self) -> None:
        store: TCPStore = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device: torch.device = torch.device("cpu")

        def init(rank: int, world_size: int) -> CheckpointTransport[Dict[str, object]]:
            pg = ProcessGroupGloo()
            pg.configure(
                store_addr=f"localhost:{store.port}/prefix",
                rank=rank,
                world_size=world_size,
            )

            return PGTransport[Dict[str, object]](
                pg, timeout=timedelta(seconds=10), device=device
            )

        run_multi_recovery_test(self, init, device=device)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipUnless(torch.cuda.device_count() >= 3, "need three CUDA devices")
    def test_pg_transport_baby_nccl(self) -> None:
        store: TCPStore = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device: torch.device = torch.device("cuda")

        def init(rank: int, world_size: int) -> CheckpointTransport[Dict[str, object]]:
            torch.cuda.set_device(rank)

            pg = ProcessGroupBabyNCCL()
            pg.configure(
                store_addr=f"localhost:{store.port}/prefix",
                rank=rank,
                world_size=world_size,
            )

            return PGTransport[Dict[str, object]](
                pg, timeout=timedelta(seconds=10), device=device
            )

        run_multi_recovery_test(self, init, device=device)
