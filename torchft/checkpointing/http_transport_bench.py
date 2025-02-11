import logging
import sys
from datetime import timedelta
from typing import List

import torch

from torchft.checkpointing.http_transport import HTTPTransport, _time

logger: logging.Logger = logging.getLogger(__name__)


def main(argv: List[str]) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=3_000_000)  # 3MB
    parser.add_argument("--total-size", type=int, default=12_000_000_000)  # 12GB
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    num_chunks: int = args.num_chunks
    CHUNK_SIZE = args.chunk_size
    TOTAL_SIZE = args.total_size

    transport = HTTPTransport(timedelta(seconds=60), num_chunks=num_chunks)
    metadata = transport.metadata()

    logger.info(f"creating state_dict... {CHUNK_SIZE=} {TOTAL_SIZE=}")

    with _time("create state_dict"):
        state_dict = {}
        for i in range(0, TOTAL_SIZE, CHUNK_SIZE):
            state_dict[f"chunk/{i}"] = torch.zeros(
                CHUNK_SIZE // 4, dtype=torch.float32, device=device
            )

    logger.info(f"fetching from {metadata=} {device=} {num_chunks=} {len(state_dict)=}")

    transport.send_checkpoint(
        dst_ranks=[0], step=1, state_dict=state_dict, timeout=timedelta(seconds=60)
    )

    with _time("fetching checkpoint"):
        transport.recv_checkpoint(
            src_rank=1, metadata=metadata, step=1, timeout=timedelta(seconds=60)
        )


if __name__ == "__main__":
    main(sys.argv[1:])
