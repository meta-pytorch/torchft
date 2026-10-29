"""Transaction decisions, safe snapshot ownership and interrupted recovery."""

import importlib
from concurrent.futures import Future
from threading import Event, Thread
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(scope="module")
def library():
    import torchft
    import torchft.manager

    # Exercise the installed candidate package without source path injection.
    parent = torchft.manager
    transaction = importlib.import_module("torchft.last_words")
    return parent, transaction


@pytest.fixture
def manager(library):
    _, module = library
    value = module.LastWordsManager.__new__(module.LastWordsManager)
    value.commit_decided = value.commit_authorized = value.commit_unknown = False
    value._checkpoint_transport = Mock()
    value.report_error = Mock()
    return value


@pytest.mark.parametrize("positional", [False, True])
def test_constructor_defaults_to_synchronous_quorum(library, positional):
    parent, module = library
    with patch.object(parent.Manager, "__init__", autospec=True, return_value=None) as initialize:
        if positional:
            module.LastWordsManager(None, None, None, 1, False)
        else:
            module.LastWordsManager(pg=None, load_state_dict=None, state_dict=None, min_replica_size=1)
        from inspect import signature

        bound = signature(initialize).bind(*initialize.call_args.args, **initialize.call_args.kwargs)
        assert bound.arguments["use_async_quorum"] is False


@pytest.mark.parametrize("positional", [False, True])
def test_constructor_rejects_async_before_native_startup(library, positional):
    parent, module = library
    with patch.object(parent.Manager, "__init__", autospec=True, return_value=None) as initialize:
        with pytest.raises(ValueError, match="synchronous quorum"):
            if positional:
                module.LastWordsManager(None, None, None, 1, True)
            else:
                module.LastWordsManager(
                    pg=None, load_state_dict=None, state_dict=None, min_replica_size=1, use_async_quorum=True
                )
        initialize.assert_not_called()


def test_vote_is_issued_once_per_attempt(library, manager, monkeypatch):
    parent, _ = library
    vote = Mock(return_value=True)
    monkeypatch.setattr(parent.Manager, "should_commit", vote)
    assert manager.should_commit() is True
    assert manager.commit_authorized
    with pytest.raises(RuntimeError, match="once per attempt"):
        manager.should_commit()
    assert vote.call_count == 1


def test_commit_timeout_is_unknown_and_disallows_publication(library, manager, monkeypatch):
    parent, module = library
    monkeypatch.setattr(parent.Manager, "should_commit", Mock(side_effect=TimeoutError("native commit")))
    with pytest.raises(module.TorchFTCommitTimeoutError, match="restart the entire replica"):
        manager.should_commit()
    assert manager.commit_unknown and manager.commit_decided
    assert not manager.commit_authorized
    manager._checkpoint_transport.disallow_checkpoint.assert_called_once()
    with pytest.raises(module.TorchFTCommitTimeoutError, match="unknown commit"):
        manager.discard_interrupted_step(RuntimeError("fault"))


def test_authorized_update_cannot_be_discarded(manager):
    manager.commit_authorized = True
    with pytest.raises(RuntimeError, match="Cannot discard an authorized"):
        manager.discard_interrupted_step(RuntimeError("fault"))


def test_terminal_stop_unblocks_wait_without_cancelling_native_future(library, manager):
    _, module = library
    manager._quorum_future = Future()
    manager.last_words_stop_requested = lambda: True
    manager.last_words_terminal_stop_requested = None
    manager.elastic_enabled = False
    manager._last_words_initial_sync_complete = True
    manager._pending_state_dict = manager._recovery_event = None
    manager.last_words_restore_incomplete = False
    manager._snapshot_ready = Event()
    manager._export_snapshot = Mock()
    with pytest.raises(module.TorchFTQuorumStoppedError):
        manager.wait_quorum()
    assert manager._snapshot_ready.is_set()
    assert not manager._quorum_future.cancelled()
    assert not manager._quorum_future.done()
    manager._export_snapshot.assert_not_called()


def test_snapshot_is_prepared_by_waiting_training_thread(library, manager, monkeypatch):
    parent, _ = library
    manager._quorum_future = Future()
    manager._snapshot_requested, manager._snapshot_ready = Event(), Event()
    manager._snapshot_error = None
    manager._healing_snapshot = None
    manager._quorum_timeout = __import__("datetime").timedelta(seconds=2)
    manager.last_words_stop_requested = manager.last_words_terminal_stop_requested = None
    manager.elastic_enabled = True
    threads = []

    def export():
        threads.append(__import__("threading").get_ident())
        manager._quorum_future.set_result(None)
        return {"user": {"step": 3}}

    manager._export_snapshot = export
    monkeypatch.setattr(parent.Manager, "wait_quorum", lambda self: None)
    result = []
    requester = Thread(target=lambda: result.append(manager._manager_state_dict()))
    requester.start()
    manager.wait_quorum()
    requester.join(timeout=3)
    assert not requester.is_alive()
    assert threads == [__import__("threading").get_ident()]
    assert result == [{"user": {"step": 3}}]


def test_partial_restore_stays_ineligible(library, manager, monkeypatch):
    parent, _ = library
    manager.last_words_request = Mock()
    manager.elastic_enabled = True
    manager._last_words_initial_sync_complete = True
    manager._pending_state_dict = {"user": {}}
    manager._recovery_event = None
    manager.last_words_restore_incomplete = False
    monkeypatch.setattr(
        parent.Manager,
        "_apply_pending_state_dict",
        Mock(side_effect=RuntimeError("partial copy")),
    )
    with pytest.raises(RuntimeError, match="partial copy"):
        manager._apply_pending_state_dict()
    assert manager.restoration_incomplete


def test_restore_completion_includes_device_sync_and_application_callback(library, manager, monkeypatch):
    parent, _ = library
    calls = []
    manager.last_words_request = Mock()
    manager.elastic_enabled = True
    manager._last_words_initial_sync_complete = True
    manager._pending_state_dict = {"user": {}}
    manager._recovery_event = None
    manager._synchronize_device = lambda: calls.append("device synchronized")
    manager.after_heal = lambda: calls.append("application restored")
    monkeypatch.setattr(
        parent.Manager,
        "_apply_pending_state_dict",
        lambda self: calls.append("native restored"),
    )
    manager._apply_pending_state_dict()
    assert calls == ["native restored", "device synchronized", "application restored"]
    assert not manager.restoration_incomplete


def test_tensor_validation_precedes_all_live_copies(library):
    import torch

    _, module = library
    original = [torch.tensor([2.0]), torch.tensor([3.0])]
    with pytest.raises(ValueError, match="shape or dtype"):
        module.restore_tensor_leaves([torch.tensor([9.0]), torch.tensor([9.0, 8.0])], original)
    assert [item.item() for item in original] == [2.0, 3.0]
