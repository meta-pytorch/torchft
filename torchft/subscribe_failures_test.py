import threading  # Import the threading module
import time
from datetime import timedelta
from unittest import TestCase

import pytest
import torch.distributed as dist

from torchft import Manager, ProcessGroupGloo
from torchft._torchft import LighthouseClient, LighthouseServer, Quorum, QuorumMember


class TestSubscribeFailures(TestCase):
    def test_subscribe_failures_notification_manager_failure(self) -> None:
        """
        Test that when two participants are in a quorum and one fails,
        the remaining participant will form a new quorum of size 1.
        """
        server_opt = {
            "bind": "[::]:0",
            "min_replicas": 1,
            "heartbeat_timeout_ms": 200,
            "failure_tick_ms": 100,
            "quorum_tick_ms": 50,
            "join_timeout_ms": 10000,
        }
        lighthouse = None
        manager_A = None
        manager_B = None
        store_A = None
        store_B = None
        manager_B_failure_stream = None

        try:
            lighthouse = LighthouseServer(**server_opt)

            # Initialize Stores for Managers
            # Note: TCPStore is used here as in other tests, assuming it's available
            # or intended despite potential linter warnings about its export.
            store_A = dist.TCPStore(
                host_name="localhost", port=0, is_master=True, wait_for_workers=False
            )
            store_B = dist.TCPStore(
                host_name="localhost", port=0, is_master=True, wait_for_workers=False
            )

            common_manager_pg = ProcessGroupGloo()
            manager_quorum_timeout = timedelta(
                milliseconds=server_opt["join_timeout_ms"] * 3
            )
            manager_heartbeat_interval = timedelta(
                milliseconds=server_opt["heartbeat_timeout_ms"] // 3
            )

            # Initialize Manager A
            manager_A = Manager(
                pg=common_manager_pg,
                min_replica_size=server_opt["min_replicas"],
                load_state_dict=lambda x: None,
                state_dict=lambda: None,
                replica_id="repA",
                store_addr="localhost",
                store_port=store_A.port,
                rank=0,
                world_size=1,
                use_async_quorum=False,
                lighthouse_addr=lighthouse.address(),
                quorum_timeout=manager_quorum_timeout,
                heartbeat_interval=manager_heartbeat_interval,
                connect_timeout=timedelta(seconds=5),
            )

            # Initialize Manager B
            manager_B = Manager(
                pg=common_manager_pg,
                min_replica_size=server_opt["min_replicas"],
                load_state_dict=lambda x: None,
                state_dict=lambda: None,
                replica_id="repB",
                store_addr="localhost",
                store_port=store_B.port,
                rank=0,
                world_size=1,
                use_async_quorum=False,
                lighthouse_addr=lighthouse.address(),
                quorum_timeout=manager_quorum_timeout,
                heartbeat_interval=manager_heartbeat_interval,
                connect_timeout=timedelta(seconds=5),
            )

            # Subscribe manager_B to failure notifications
            if manager_B._lighthouse_client is not None:
                manager_B_failure_stream = (
                    manager_B._lighthouse_client.subscribe_failures(
                        timeout=timedelta(seconds=5)
                    )
                )
            else:
                self.fail("Manager B's lighthouse client was not initialized.")

            # Stage 1: Both managers join and are part of the same quorum
            # Use threading to start both managers' quorum attempts concurrently
            thread_A = threading.Thread(
                target=manager_A.start_quorum, args=(manager_quorum_timeout,)
            )
            thread_B = threading.Thread(
                target=manager_B.start_quorum, args=(manager_quorum_timeout,)
            )

            thread_A.start()
            thread_B.start()

            thread_A.join()
            thread_B.join()

            assert (
                manager_A.num_participants() == 2
            ), f"Initial quorum for repA should have 2 participants, got {manager_A.num_participants()}"
            assert (
                manager_B.num_participants() == 2
            ), f"Initial quorum for repB should have 2 participants, got {manager_B.num_participants()}"

            # Stage 2: Simulate failure of Manager A
            manager_A.shutdown()

            # Wait for failure detection by Lighthouse
            # Sleep duration is based on server configuration to ensure detection occurs
            sleep_duration_stage2 = (
                server_opt["heartbeat_timeout_ms"] + server_opt["failure_tick_ms"] * 2
            ) / 1000.0
            time.sleep(sleep_duration_stage2)

            # Verify Manager B receives failure notification
            if manager_B_failure_stream is not None:
                try:
                    failure_note = next(manager_B_failure_stream)
                    assert (
                        "repA" in failure_note.replica_id
                    ), f"Expected failure notification for repA, but got {failure_note.replica_id}"
                except StopIteration:
                    self.fail("Manager B did not receive a failure notification.")
                except Exception as e:
                    self.fail(f"Error reading failure notification: {e}")
            else:
                self.fail("Failure stream was not initialized for Manager B.")

            # Stage 3: Manager B (healthy) attempts to form a new quorum
            # This call should now form a quorum of size 1 including only Manager B
            manager_B.start_quorum(timeout=manager_quorum_timeout)

            assert (
                manager_B.num_participants() == 1
            ), f"Final quorum for repB should have 1 participant, got {manager_B.num_participants()}."

        finally:
            # Cleanup
            if lighthouse:
                lighthouse.shutdown()
            if manager_A:
                manager_A.shutdown()
            if manager_B:
                manager_B.shutdown()
