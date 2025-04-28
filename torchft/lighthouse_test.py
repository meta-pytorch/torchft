import time
from datetime import timedelta
from unittest import TestCase

import torch.distributed as dist

from torchft import Manager, ProcessGroupGloo
from torchft._torchft import LighthouseClient, LighthouseServer, Quorum, QuorumMember


class TestLighthouse(TestCase):
    def test_join_timeout_behavior(self) -> None:
        """Test that join_timeout_ms affects joining behavior"""
        # To test, we create a lighthouse with 100ms and 400ms join timeouts
        # and measure the time taken to validate the quorum.
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=1,
            join_timeout_ms=100,
        )

        # Create a manager that tries to join
        try:
            store = dist.TCPStore(
                host_name="localhost",
                port=0,
                is_master=True,
                wait_for_workers=False,
            )
            pg = ProcessGroupGloo()
            manager = Manager(
                pg=pg,
                min_replica_size=1,
                load_state_dict=lambda x: None,
                state_dict=lambda: None,
                replica_id=f"lighthouse_test",
                store_addr="localhost",
                store_port=store.port,
                rank=0,
                world_size=1,
                use_async_quorum=False,
                lighthouse_addr=lighthouse.address(),
            )

            start_time = time.time()
            manager.start_quorum()
            time_taken = time.time() - start_time
            assert time_taken < 0.4, f"Time taken to join: {time_taken} > 0.4s"

        finally:
            # Cleanup
            lighthouse.shutdown()
            if "manager" in locals():
                manager.shutdown()

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=1,
            join_timeout_ms=400,
        )

    def test_heartbeat_timeout_ms_sanity(self) -> None:
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=1,
            heartbeat_timeout_ms=100,
        )
        lighthouse.shutdown()

    def test_lighthouse_client_behavior(self) -> None:
        """Test that using LighthouseClient with a generic quorum behavior"""
        # To test, we create a lighthouse with 100ms and 400ms join timeouts
        # and measure the time taken to validate the quorum.
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=1,
            join_timeout_ms=100,
        )

        # Create a manager that tries to join
        try:
            client = LighthouseClient(
                addr=lighthouse.address(),
                connect_timeout=timedelta(seconds=1),
            )
            store = dist.TCPStore(
                host_name="localhost",
                port=0,
                is_master=True,
                wait_for_workers=False,
            )
            result = client.quorum(
                replica_id="lighthouse_test",
                address="localhost",
                store_address=f"localhost:{store.port}",
                step=1,
                world_size=1,
                shrink_only=False,
                timeout=timedelta(seconds=1),
                data={"my_data": 1234},
            )
            assert result is not None
            assert isinstance(result, Quorum)
            assert len(result.participants) == 1
            for member in result.participants:
                assert isinstance(member, QuorumMember)
                assert member.replica_id == "lighthouse_test"
                assert member.data is not None
                assert "my_data" in member.data
                assert member.data["my_data"] == 1234

            # Test the optional args
            result = client.quorum(
                replica_id="lighthouse_test",
                timeout=timedelta(seconds=1),
            )
            assert result is not None
            for member in result.participants:
                assert member.replica_id == "lighthouse_test"

        finally:
            # Cleanup
            lighthouse.shutdown()

    def test_heartbeat_round_trip(self) -> None:
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=1,
            heartbeat_timeout_ms=200,
        )
        try:
            client = LighthouseClient(
                addr=lighthouse.address(),
                connect_timeout=timedelta(seconds=1),
            )

            client.heartbeat("rep0")

            # (Should still be alive, as sleep time is less than timeout)
            time.sleep(0.15)
            q = client.quorum(
                replica_id="rep0",
                timeout=timedelta(milliseconds=500),
            )
            assert any(m.replica_id == "rep0" for m in q.participants)

            # (Wait long enough for timeout to trigger)
            time.sleep(0.25)
            # "Probe" with different replica so we don't revive rep0
            probe = client.quorum(
                replica_id="probe",
                timeout=timedelta(milliseconds=500),
            )
            assert all(m.replica_id != "rep0" for m in probe.participants)

        finally:
            lighthouse.shutdown()
