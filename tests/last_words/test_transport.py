"""Exercise the packaged service over HTTP without importing TorchFT training."""

import subprocess
import sys
import threading

import pytest
from torchft_last_words.client import OwnerClient, WorkerTransport
from torchft_last_words.service import ServiceServer


@pytest.fixture
def session(monkeypatch):
    server = ServiceServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    client = OwnerClient(
        "transport-test",
        address=address,
        num_replicas=2,
        world_size=2,
        heartbeat_timeout=30,
        startup_timeout=60,
        timeout=100,
    )
    owner = client.session
    workers = []
    for replica in range(2):
        for key, value in client.environment(replica).items():
            monkeypatch.setenv(key, value)
        workers.append(
            [WorkerTransport(owner.run_id, replica, rank) for rank in range(2)]
        )
    try:
        yield owner, workers
    finally:
        for group in workers:
            for worker in group:
                worker.client.close()
        owner.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def freeze(worker, step):
    state = dict(step=step, last_words_num_updates=step)
    worker.set_ready(state, "same-model-and-input-contract")
    return worker.report(
        dict(
            phase="frozen",
            state=state,
            world_size=2,
            heartbeat_timeout_seconds=30,
        ),
        True,
    )


def test_all_shards_required_and_newest_complete_replica_wins(session):
    owner, workers = session
    for worker in workers[0]:
        freeze(worker, 3)
    freeze(workers[1][0], 4)
    assert owner.call("owner_tick", {})["grant"] is None
    freeze(workers[1][1], 4)
    grant = owner.call("owner_tick", {})["grant"]
    assert grant["replica_id"] == 1
    assert grant["state"] == dict(step=4, last_words_num_updates=4)


def test_backup_requires_explicit_fencing_confirmation(session):
    owner, workers = session
    for group in workers:
        for worker in group:
            freeze(worker, 3)
    before = owner.call("owner_tick", {})
    writer = before["grant"]["replica_id"]
    pending = owner.call("owner_tick", {str(writer): 1})
    assert pending["phase"] == "FENCING"
    assert pending["grant"] == before["grant"]
    with pytest.raises(RuntimeError, match="fencing not confirmed"):
        owner.call("fenced", False, round=before["grant"]["round"], replica=writer)
    next_round = owner.call(
        "fenced", True, round=before["grant"]["round"], replica=writer
    )
    assert next_round["grant"]["replica_id"] != writer
    assert next_round["grant"]["round"] == before["grant"]["round"] + 1


def test_lost_socket_reconnects_with_the_same_authority(session):
    owner, _ = session
    owner.client.close()
    assert owner.call("owner_tick", {})["instance"] == owner.instance


def test_training_imports_are_not_needed_for_service_or_client():
    code = """
import sys
class RefuseTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'torchft', 'torchtitan'}:
            raise RuntimeError('Unexpected training import: ' + fullname)
sys.meta_path.insert(0, RefuseTrainingImports())
from torchft_last_words.client import ServiceClient, SessionClient, WorkerTransport
from torchft_last_words.service import ServiceServer
server = ServiceServer(('127.0.0.1', 0))
server.server_close()
print('independent')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "independent"


def test_module_entrypoint_is_available():
    result = subprocess.run(
        [sys.executable, "-m", "torchft_last_words", "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--address-file" in result.stdout
