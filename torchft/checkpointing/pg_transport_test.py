import sys
from datetime import timedelta
from unittest import TestCase, skipIf, skipUnless

import torch
from torch.distributed import TCPStore

from torchft.checkpointing.pg_transport import PGTransport
from torchft.checkpointing.transport import CheckpointTransport
from torchft.checkpointing.transport_test import (
    make_state_dict,
    run_multi_recovery_test,
)
from torchft.process_group import ProcessGroupBabyNCCL, ProcessGroupGloo


class PGTransportTest(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not passing on mac")
    def test_pg_transport_gloo(self) -> None:
        store: TCPStore = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device: torch.device = torch.device("cpu")

        def init(rank: int, world_size: int) -> CheckpointTransport[dict[str, object]]:
            pg = ProcessGroupGloo()
            pg.configure(
                store_addr=f"localhost:{store.port}/prefix",
                rank=rank,
                world_size=world_size,
            )

            return PGTransport[dict[str, object]](
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
        timeout: timedelta = timedelta(seconds=10)

        def init(rank: int, world_size: int) -> CheckpointTransport[dict[str, object]]:
            torch.cuda.set_device(rank)

            pg = ProcessGroupBabyNCCL(timeout=timeout)
            pg.configure(
                store_addr=f"localhost:{store.port}/prefix",
                rank=rank,
                world_size=world_size,
            )

            return PGTransport[dict[str, object]](pg, timeout=timeout, device=device)

        run_multi_recovery_test(self, init, device=device)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipUnless(torch.cuda.device_count() >= 3, "need three CUDA devices")
    def test_pg_transport_baby_nccl_inplace(self) -> None:
        store: TCPStore = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device: torch.device = torch.device("cuda")
        timeout: timedelta = timedelta(seconds=10)

        def state_dict() -> dict[str, object]:
            return make_state_dict(device)

        def init(rank: int, world_size: int) -> CheckpointTransport[dict[str, object]]:
            torch.cuda.set_device(rank)

            pg = ProcessGroupBabyNCCL(timeout=timeout)
            pg.configure(
                store_addr=f"localhost:{store.port}/prefix",
                rank=rank,
                world_size=world_size,
            )

            return PGTransport[dict[str, object]](
                pg,
                timeout=timeout,
                device=device,
                state_dict=state_dict,
            )

        run_multi_recovery_test(self, init, device=device)
