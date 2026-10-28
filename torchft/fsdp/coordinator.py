# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coordinator for fault tolerant FSDP: TCPStore master plus lighthouse.

Run with ``python -m torchft.fsdp.coordinator``. Settings
come from the environment:

- ``FTFSDP_STORE_PORT`` (default 29600) and ``FTFSDP_LIGHTHOUSE_PORT``
  (default 29510).
- ``FTFSDP_MIN_REPLICAS``: trainer processes needed for a quorum
  (``num_active_hosts * procs_per_host``).
- ``FTFSDP_JOIN_TIMEOUT_MS`` (default 10000) and
  ``FTFSDP_HEARTBEAT_TIMEOUT_MS`` (default 5000).
- ``FTFSDP_RUN_ID`` (default ``ftfsdp``): the run to wait for.

Exits shortly after that run sets the done key, whether training finished
or failed.
"""

import logging
import os
import time
from datetime import timedelta

import torch.distributed as dist
from torchft.fsdp.membership import DONE_KEY

logger = logging.getLogger(__name__)


def main() -> None:
    from torchft._torchft import LighthouseServer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    store_port = int(os.environ.get("FTFSDP_STORE_PORT", "29600"))
    lighthouse_port = int(os.environ.get("FTFSDP_LIGHTHOUSE_PORT", "29510"))
    store = dist.TCPStore(
        "::",
        store_port,
        is_master=True,
        wait_for_workers=False,
        timeout=timedelta(hours=24),
        use_libuv=True,
    )
    lighthouse = LighthouseServer(
        bind=f"[::]:{lighthouse_port}",
        min_replicas=int(os.environ["FTFSDP_MIN_REPLICAS"]),
        join_timeout_ms=int(os.environ.get("FTFSDP_JOIN_TIMEOUT_MS", "10000")),
        quorum_tick_ms=100,
        heartbeat_timeout_ms=int(os.environ.get("FTFSDP_HEARTBEAT_TIMEOUT_MS", "5000")),
    )
    run_store = dist.PrefixStore(os.environ.get("FTFSDP_RUN_ID", "ftfsdp"), store)
    logger.info(f"store on port {store_port}, lighthouse at {lighthouse.address()}")
    # The probe line lets the node agent detect a stalled store server.
    last_probe = 0.0
    outcome = None
    while outcome is None:
        time.sleep(1)
        start = time.monotonic()
        try:
            if run_store.check([DONE_KEY]):
                outcome = run_store.get(DONE_KEY).decode()
        except Exception as e:
            logger.warning(f"store probe failed: {e}")
            continue
        latency = time.monotonic() - start
        if latency > 1.0 or start - last_probe >= 10.0:
            logger.info(f"store probe {latency * 1000:.1f}ms")
            last_probe = start
    logger.info(f"training done ({outcome}); shutting down in 60s")
    time.sleep(60)
    lighthouse.shutdown()


if __name__ == "__main__":
    main()
