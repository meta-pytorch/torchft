"""Exercise the candidate Manager API against the installed native TorchFT ABI."""

from concurrent.futures import Future
from dataclasses import FrozenInstanceError

import pytest


@pytest.fixture
def manager():
    import torchft.manager as module

    instance = module.Manager.__new__(module.Manager)
    instance._quorum_future = Future()
    instance._quorum_future.set_result(None)
    instance._quorum_id = 7
    instance._quorum_member_ids = (0, 1)
    instance._participating_replica_world_size = 2
    instance._step = 4
    instance._batches_committed = 8
    instance._pending_state_dict = None
    instance._healing = False
    instance.errored = lambda: None
    return instance


def test_snapshot_is_immutable_native_progress(manager):
    snapshot = manager.quorum_snapshot()
    assert (
        snapshot.quorum_id,
        snapshot.members,
        snapshot.step,
        snapshot.batches_committed,
    ) == (7, (0, 1), 4, 8)
    with pytest.raises(FrozenInstanceError):
        snapshot.step = 5


@pytest.mark.parametrize("future", [None, Future()])
def test_pending_quorum_is_not_an_observation(manager, future):
    manager._quorum_future = future
    with pytest.raises(RuntimeError, match="has not completed"):
        manager.quorum_snapshot()


def test_failed_future_propagates_before_membership(manager):
    future = Future()
    future.set_exception(RuntimeError("native quorum failed"))
    manager._quorum_future = future
    with pytest.raises(RuntimeError, match="native quorum failed"):
        manager.quorum_snapshot()


@pytest.mark.parametrize("pending,healing", [({}, False), (None, True)])
def test_partial_restore_has_no_safe_observation(manager, pending, healing):
    manager._pending_state_dict = pending
    manager._healing = healing
    with pytest.raises(RuntimeError, match="restoration is incomplete"):
        manager.quorum_snapshot()


def test_failed_configuration_is_not_a_healthy_boundary(manager):
    manager.errored = lambda: RuntimeError("configure failed")
    with pytest.raises(RuntimeError, match="not healthy"):
        manager.quorum_snapshot()


@pytest.mark.parametrize("members", [(), (0,), (0, 0)])
def test_inconsistent_membership_is_rejected(manager, members):
    manager._quorum_member_ids = members
    with pytest.raises(RuntimeError, match="membership and participant count disagree"):
        manager.quorum_snapshot()
