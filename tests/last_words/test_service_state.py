# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Safety contracts of the independent authority, with a controllable clock."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from torchft_last_words import service

MODULE = vars(service)


@pytest.fixture
def state():
    now = [0.0]
    coordinator = MODULE["Coordinator"](lambda: now[0])
    seq = [1]
    registration = dict(
        version=1,
        op="register",
        run_id="run",
        instance="",
        token="o" * 32,
        seq=1,
        data=dict(
            world_size=2,
            replica_tokens=["a" * 32, "b" * 32],
            heartbeat_seconds=10,
            startup_seconds=60,
            timeout_seconds=100,
        ),
    )
    coordinator.command(registration)

    def owner(op="owner_tick", data=None, **kwargs):
        seq[0] += 1
        return coordinator.command(
            dict(
                registration,
                instance=coordinator.instance,
                op=op,
                seq=seq[0],
                data={} if data is None else data,
                **kwargs,
            )
        )

    def report(
        replica,
        rank,
        *,
        update=3,
        sequence=1,
        phase="frozen",
        requested=True,
        signature="model",
    ):
        value = dict(step=update, last_words_num_updates=update)
        return coordinator.command(
            dict(
                version=1,
                op="shard_report",
                run_id="run",
                instance=coordinator.instance,
                token=("a" if replica == 0 else "b") * 32,
                seq=sequence,
                replica=replica,
                rank=rank,
                data=dict(
                    world_size=2,
                    heartbeat_timeout_seconds=10,
                    phase=phase,
                    requested=requested,
                    state=value,
                    ready=dict(state=value, signature=signature)
                    if phase == "frozen"
                    else None,
                ),
            )
        )

    return coordinator, now, owner, report, registration


def ready_all(report, **kwargs):
    for replica in range(2):
        for rank in range(2):
            report(replica, rank, **kwargs)


def test_highest_complete_update_then_lowest_replica_wins(state):
    coordinator, _, owner, report, _ = state
    for replica in range(2):
        for rank in range(2):
            report(replica, rank, update=3 + replica)
    assert owner()["grant"]["replica_id"] == 1
    assert coordinator.sessions["run"].grant["state"]["last_words_num_updates"] == 4


def test_tied_candidates_choose_replica_zero(state):
    _, _, owner, report, _ = state
    ready_all(report)
    assert owner()["grant"]["replica_id"] == 0


def test_no_grant_until_every_shard_of_a_candidate_is_frozen(state):
    _, _, owner, report, _ = state
    report(0, 0)
    report(1, 0)
    assert owner()["grant"] is None


def test_pending_optimizer_gets_prepare_window(state):
    _, now, owner, report, _ = state
    report(0, 0)
    report(0, 1)
    for rank in range(2):
        report(1, rank, phase="updating", requested=False)
    assert owner()["grant"] is None
    now[0] = 9
    for rank in range(2):
        report(1, rank, sequence=2, update=4)
    assert owner()["grant"]["replica_id"] == 1


def test_busy_peer_does_not_extend_original_global_deadline(state):
    coordinator, now, owner, report, _ = state
    report(0, 0, phase="updating")
    report(0, 1, phase="updating")
    report(1, 0, phase="updating")
    report(1, 1, phase="updating")
    for index in range(1, 12):
        now[0] = index * 9
        owner()
        ready_all(report, phase="updating", sequence=index + 1)
    now[0] = 100
    coordinator.tick()
    assert owner()["result"]["error"] == "global save deadline expired"


def test_single_live_replica_can_save_without_training_quorum(state):
    _, _, owner, report, _ = state
    owner(data={"1": 1})
    report(0, 0)
    report(0, 1)
    assert owner()["grant"]["replica_id"] == 0


def test_one_shard_cannot_refresh_dead_peer(state):
    _, now, owner, report, _ = state
    ready_all(report, requested=False)
    now[0] = 9
    owner()
    report(0, 0, sequence=2, requested=False)
    for rank in range(2):
        report(1, rank, sequence=2, requested=False)
    now[0] = 11
    owner()
    # Stale shard excludes replica 0; only complete replica 1 may save.
    now[0] = 18
    owner()
    for rank in range(2):
        report(1, rank, sequence=3)
    now[0] = 21
    snapshot = owner()
    assert snapshot["grant"]["replica_id"] == 1


@pytest.mark.parametrize("phase", ["ineligible", "updating"])
def test_unsafe_replica_is_never_elected(state, phase):
    _, now, owner, report, _ = state
    for rank in range(2):
        report(0, rank, phase=phase)
        report(1, rank)
    if phase == "updating":
        now[0] = 9
        owner()
        for rank in range(2):
            report(0, rank, phase=phase, sequence=2)
            report(1, rank, sequence=2)
        now[0] = 10
    assert owner()["grant"]["replica_id"] == 1


def test_owner_heartbeat_expiry_cannot_be_revived(state):
    coordinator, now, owner, _, _ = state
    now[0] = 11
    coordinator.tick()
    assert owner()["result"]["error"] == "launcher heartbeat expired"


def test_idempotent_requests_do_not_refresh_shard_liveness(state):
    coordinator, now, owner, report, _ = state
    ready_all(report, requested=False)
    now[0] = 9
    owner()
    ready_all(report, requested=False)
    now[0] = 11
    owner()
    assert coordinator.sessions["run"].excluded == {0, 1}


@pytest.mark.parametrize(
    "case", ["version", "token", "incarnation", "worker_mutation", "unknown_field"]
)
def test_protocol_and_actor_isolation(state, case):
    coordinator, _, _, _, registration = state
    command = dict(registration, op="poll", instance=coordinator.instance, seq=2)
    if case == "version":
        command["version"] = 2
    elif case == "token":
        command["token"] = "x" * 32
    elif case == "incarnation":
        command["instance"] = "old"
    elif case == "worker_mutation":
        command.update(token="a" * 32, op="validated")
    else:
        command["extra"] = True
    with pytest.raises(ValueError):
        coordinator.command(command)


def test_shard_disagreement_fails_closed_before_grant(state):
    _, _, owner, report, _ = state
    report(0, 0)
    with pytest.raises(ValueError, match="disagree"):
        report(0, 1, update=4)
    assert owner()["phase"] == "FAILED"
    assert owner()["grant"] is None


def test_round_requires_positive_fencing_before_backup(state):
    _, _, owner, report, _ = state
    ready_all(report)
    assert owner()["grant"]["round"] == 1
    owner(data={"0": 1})
    for data, round in [(True, 0), (False, 1)]:
        with pytest.raises(ValueError):
            owner("fenced", data, round=round, replica=0)
    grant = owner("fenced", True, round=1, replica=0)["grant"]
    assert (grant["replica_id"], grant["round"]) == (1, 2)


def test_writer_budget_and_failover_share_original_deadline(state):
    coordinator, now, owner, report, _ = state
    ready_all(report)
    assert owner()["grant"]["round"] == 1
    for index in range(1, 7):
        now[0] = index * 9
        owner()
        ready_all(report, sequence=index + 1)
    assert owner()["phase"] == "FENCING"
    assert owner("fenced", True, replica=0, round=1)["grant"]["round"] == 2
    assert coordinator.sessions["run"].writer_deadline == 100


def test_completed_session_is_immutable_and_tokens_not_exposed(state):
    coordinator, _, owner, report, _ = state
    ready_all(report)
    owner()
    first = owner("validated", round=1, replica=0)
    assert first["phase"] == "SUCCEEDED"
    assert owner("fail", "late") == first
    raw = json.dumps(first)
    assert all(token not in raw for token in ["o" * 32, "a" * 32, "b" * 32])
    first["grant"]["replica_id"] = 99
    assert coordinator.sessions["run"].grant["replica_id"] == 0


def test_concurrent_shard_reports_produce_one_grant(state):
    coordinator, _, owner, report, _ = state
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda pair: report(*pair), [(0, 0), (0, 1), (1, 0), (1, 1)]))
    assert owner()["grant"]["round"] == 1
    assert coordinator.sessions["run"].round == 1


@pytest.mark.parametrize("value", [0, -1, float("inf"), True])
def test_invalid_budget_is_rejected_without_creating_session(state, value):
    coordinator, _, _, _, registration = state
    command = copy.deepcopy(registration)
    command["run_id"] = "other"
    command["data"]["heartbeat_seconds"] = value
    # Use a fresh service: another active run is independently prohibited.
    coordinator = MODULE["Coordinator"]()
    with pytest.raises(ValueError):
        coordinator.command(command)
    assert not coordinator.sessions
