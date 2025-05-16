from __future__ import annotations

import datetime as _dt
import time

import pytest

import torchft._torchft as ext

_TIMEOUT = _dt.timedelta(seconds=3)

def _client(addr: str, room: str) -> ext.LighthouseClient:
    """Helper: create a LighthouseClient bound to a logical room."""
    return ext.LighthouseClient(addr, _TIMEOUT, room)


@pytest.mark.asyncio
async def test_multi_room_quorums() -> None:
    # 1) Launch one Lighthouse server on any free port
    server = ext.LighthouseServer("[::]:0", min_replicas=1)
    addr: str = server.address()

    # (give the Tokio runtime a tick to bind the listener)
    time.sleep(0.1)

    # 2) Two clients, each in its own room
    cli_a = _client(addr, "jobA")
    cli_b = _client(addr, "jobB")

    # 3) Explicit heart-beats (exercise the RPC path)
    cli_a.heartbeat("a0")
    cli_b.heartbeat("b0")

    # 4) Ask each room for a quorum
    q_a = cli_a.quorum("a0", _TIMEOUT)
    q_b = cli_b.quorum("b0", _TIMEOUT)

    # 5) Assert the rooms are isolated
    assert q_a.quorum_id == q_b.quorum_id == 1
    assert len(q_a.participants) == 1 and q_a.participants[0].replica_id == "a0"
    assert len(q_b.participants) == 1 and q_b.participants[0].replica_id == "b0"

    # 6) Clean shutdown
    server.shutdown()
