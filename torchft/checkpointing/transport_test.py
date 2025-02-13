import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Callable, Dict, List
from unittest import TestCase

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, distribute_tensor

from torchft.checkpointing.transport import CheckpointTransport

TIMEOUT_REGEX = r"(Timed out|timed out|timeout|time out)"


def assertStateDictEqual(
    self: TestCase, a: Dict[str, object], b: Dict[str, object]
) -> None:
    for k, v1 in a.items():
        v2 = b[k]
        if isinstance(v1, DTensor) and isinstance(v2, DTensor):
            torch.testing.assert_close(v1._local_tensor, v2._local_tensor)
            self.assertEqual(v1._spec, v2._spec)
        elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            torch.testing.assert_close(v1.cpu(), v2.cpu())
        else:
            self.assertEqual(v1, v2)


def run_multi_recovery_test(
    self: TestCase,
    init_transport: Callable[[int, int], CheckpointTransport[Dict[str, object]]],
    device: torch.device,
) -> None:
    """
    This runs multi node recovery tests for a given transport function.

    This tests send/recv in a 3 node setup, with all and some workers recovering
    and also tests timeout behavior.
    """
    WORLD_SIZE: int = 3

    # barrier is used to simulate quorum/allreduce barriers
    barrier: threading.Barrier = threading.Barrier(WORLD_SIZE)
    metadata: str = ""

    dist.init_process_group(
        backend="gloo", rank=0, world_size=1, store=dist.HashStore()
    )

    device_mesh = DeviceMesh("cpu", 1)
    tensor = torch.randn(4, 4)
    dtensor: DTensor = distribute_tensor(tensor, device_mesh, [])

    def run(rank: int) -> CheckpointTransport[Dict[str, object]]:
        transport = init_transport(rank, WORLD_SIZE)

        if rank == 0:
            nonlocal metadata
            metadata = transport.metadata()

        barrier.wait()

        state_dict: Dict[str, object] = {
            "rank": torch.tensor([1, 2, 3], device=device),
            "str": "str",
            "int": 1234,
            "dtensor": dtensor,
        }

        # 3 node recovery
        if rank == 0:
            transport.send_checkpoint(
                dst_ranks=[1, 2],
                step=1,
                state_dict=state_dict,
                timeout=timedelta(seconds=10),
            )
        else:
            got = transport.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1, timeout=timedelta(seconds=10)
            )
            assertStateDictEqual(self, got, state_dict)

        barrier.wait()
        transport.disallow_checkpoint()

        # 2 node recovery
        if rank == 0:
            transport.send_checkpoint(
                dst_ranks=[2],
                step=2,
                state_dict=state_dict,
                timeout=timedelta(seconds=10),
            )
        elif rank == 2:
            got = transport.recv_checkpoint(
                src_rank=0, metadata=metadata, step=2, timeout=timedelta(seconds=10)
            )
            assertStateDictEqual(self, got, state_dict)

        barrier.wait()
        transport.disallow_checkpoint()

        # timeout test
        if rank == 2:
            with self.assertRaisesRegex(Exception, TIMEOUT_REGEX):
                transport.recv_checkpoint(
                    src_rank=0,
                    metadata=metadata,
                    step=3,
                    timeout=timedelta(milliseconds=10),
                )

            # Make sure send completes quickly.
            # If the transport is async (such as with HTTP) this may just return
            # immediately.
            try:
                transport.send_checkpoint(
                    dst_ranks=[0],
                    step=4,
                    state_dict=state_dict,
                    timeout=timedelta(seconds=10),
                )
            except Exception:
                with self.assertRaisesRegex(Exception, TIMEOUT_REGEX):
                    raise

        return transport

    with ThreadPoolExecutor(max_workers=WORLD_SIZE) as executor:
        results = []
        for i in range(WORLD_SIZE):
            results.append(executor.submit(run, i))

        transports = []

        try:
            for fut in as_completed(results, timeout=10.0):
                transports.append(fut.result())
        except Exception as e:
            print(e)
            raise

        for transport in transports:
            transport.shutdown()

    dist.destroy_process_group()
