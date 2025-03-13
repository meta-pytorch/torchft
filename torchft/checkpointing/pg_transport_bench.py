import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import torch
import torch.distributed as dist

from torchft.checkpointing.pg_transport import PGTransport, _timeit
from torchft.process_group import ProcessGroupBabyNCCL

logger: logging.Logger = logging.getLogger(__name__)


def main(argv: list[str]) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=3_000_000)  # 3MB
    parser.add_argument("--total-size", type=int, default=12_000_000_000)  # 12GB
    args = parser.parse_args(argv)

    CHUNK_SIZE: int = args.chunk_size
    TOTAL_SIZE: int = args.total_size
    INPLACE: bool = args.inplace
    DEVICE: str = args.device

    timeout: timedelta = timedelta(seconds=10)

    store = dist.TCPStore(
        "localhost",
        0,
        is_master=True,
        timeout=timeout,
        wait_for_workers=False,
    )
    store_addr: str = f"localhost:{store.port}"

    def run(rank: int) -> None:
        torch.cuda.set_device(rank)

        device = torch.device(DEVICE)

        with _timeit("init_pg"):
            pg = ProcessGroupBabyNCCL(timeout=timeout)
            pg.configure(store_addr=store_addr, rank=rank, world_size=2)

            t = torch.zeros(10, device=device, dtype=torch.float32)
            pg.allreduce([t], dist.ReduceOp.SUM).wait(timeout=timeout)

        with _timeit("create state_dict"):
            state_dict: dict[str, torch.Tensor] = {}
            for i in range(0, TOTAL_SIZE, CHUNK_SIZE):
                state_dict[f"chunk/{i}"] = torch.zeros(
                    CHUNK_SIZE // 4, dtype=torch.float32, device=device
                )

        def get_state_dict() -> object:
            return state_dict

        transport = PGTransport(
            pg=pg,
            timeout=timeout,
            device=device,
            state_dict=get_state_dict if INPLACE else None,
        )
        metadata = transport.metadata()

        if rank == 0:
            with _timeit("send_checkpoint"):
                transport.send_checkpoint(
                    dst_ranks=[1],
                    step=1,
                    state_dict=state_dict,
                    timeout=timedelta(seconds=60),
                )
        elif rank == 1:
            with _timeit("recv_checkpoint"):
                transport.recv_checkpoint(
                    src_rank=0, metadata=metadata, step=1, timeout=timedelta(seconds=60)
                )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(run, range(2))
        list(results)


if __name__ == "__main__":
    main(sys.argv[1:])
