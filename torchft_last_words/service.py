# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Single-host emergency-save authority, independent of TorchFT and torch.

All shard aggregation and writer decisions are serialized by one lock. Server
incarnations are deliberately ephemeral: losing this process fails the attempt.
Remote execution/fencing and replicated service recovery are not implemented.
"""

import argparse
import copy
import hmac
import json
import math
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

VERSION = 1
MAX_REQUEST_BYTES = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def uint(value):
    return type(value) is int and 0 <= value < 2**64


class Session:
    def __init__(self, command, now):
        self.config = command["data"]
        config = self.config
        require(
            isinstance(config, dict)
            and set(config)
            == {
                "world_size",
                "replica_tokens",
                "heartbeat_seconds",
                "startup_seconds",
                "timeout_seconds",
            },
            "invalid registration fields",
        )
        require(
            uint(config["world_size"]) and 1 <= config["world_size"] <= 64,
            "invalid shard count",
        )
        tokens = config["replica_tokens"]
        require(
            isinstance(tokens, list) and 1 <= len(tokens) <= 64, "invalid replica count"
        )
        require(
            all(
                isinstance(token, str)
                and len(token) >= 32
                and token != command["token"]
                for token in tokens
            ),
            "invalid replica token",
        )
        require(len(set(tokens)) == len(tokens), "duplicate replica token")
        for key in ("heartbeat_seconds", "startup_seconds", "timeout_seconds"):
            value = config[key]
            require(
                type(value) in (float, int) and math.isfinite(value) and value > 0,
                "invalid time budget",
            )
        self.owner = command["token"]
        self.started = self.owner_at = now
        self.owner_seq = command["seq"]
        self.owner_command = command
        self.fault_at = None
        self.reason = ""
        self.phase = "RUNNING"
        self.reports = {}
        self.exits = {}
        self.excluded = set()
        self.grant = None
        self.round = 0
        self.signature = None
        self.writer_deadline = 0
        self.result = None
        self.revision = 1

    def stop(self, now, reason):
        if self.fault_at is None and self.result is None:
            self.fault_at, self.reason, self.phase = now, reason, "PREPARING"
            self.revision += 1

    def finish(self, error=""):
        if self.result is None:
            self.phase = "FAILED" if error else "SUCCEEDED"
            self.result = dict(
                success=not error,
                error=error,
                round=self.round,
                replica_id=self.grant["replica_id"] if self.grant else None,
            )
            self.revision += 1

    def shards(self, replica):
        return [
            self.reports.get((replica, rank), {}).get("data")
            for rank in range(self.config["world_size"])
        ]

    def ready(self, replica):
        shards = self.shards(replica)
        if any(
            not shard or shard["phase"] != "frozen" or shard.get("ready") is None
            for shard in shards
        ):
            return None
        ready = shards[0]["ready"]
        require(
            all(
                shard["ready"] == ready and shard["state"] == ready["state"]
                for shard in shards
            ),
            "ready shards disagree",
        )
        state = ready["state"]
        require(
            isinstance(ready.get("signature"), str)
            and bool(ready["signature"])
            and uint(state.get("step"))
            and uint(state.get("last_words_num_updates")),
            "invalid completed state",
        )
        return ready

    def tick(self, now):
        if self.result is not None:
            return
        if now - self.owner_at > self.config["heartbeat_seconds"]:
            self.finish("launcher heartbeat expired")
            return
        for replica in range(len(self.config["replica_tokens"])):
            if replica in self.excluded or self.exits.get(replica) == 0:
                continue
            for rank in range(self.config["world_size"]):
                report = self.reports.get((replica, rank))
                stale = (
                    now - report["at"] > self.config["heartbeat_seconds"]
                    if report
                    else now - self.started > self.config["startup_seconds"]
                )
                if stale:
                    self.excluded.add(replica)
                    self.stop(now, "missing replica/shard heartbeat")
                    break
        if self.fault_at is None:
            return
        remaining = self.config["timeout_seconds"] - (now - self.fault_at)
        if remaining <= 0:
            self.finish("global save deadline expired")
            return
        if self.grant is not None:
            writer = self.grant["replica_id"]
            if (
                writer in self.exits
                or writer in self.excluded
                or now >= self.writer_deadline
            ) and self.phase != "FENCING":
                self.phase = "FENCING"
                self.revision += 1
            return
        candidates, pending = [], False
        for replica in range(len(self.config["replica_tokens"])):
            if replica in self.excluded or replica in self.exits:
                continue
            ready = self.ready(replica)
            if ready is None:
                pending = True
            else:
                candidates.append((replica, ready))
        if pending and (
            not candidates or now - self.fault_at < self.config["heartbeat_seconds"]
        ):
            return
        if not candidates:
            self.finish("no complete live replica ready")
            return
        signatures = {ready["signature"] for _, ready in candidates}
        require(
            len(signatures) == 1
            and (self.signature is None or self.signature in signatures),
            "replica signatures disagree",
        )
        self.signature = next(iter(signatures))
        writer, ready = max(
            candidates,
            key=lambda item: (
                item[1]["state"]["last_words_num_updates"],
                item[1]["state"]["step"],
                -item[0],
            ),
        )
        self.round += 1
        self.writer_deadline = now + remaining / (2 if len(candidates) > 1 else 1)
        self.grant = dict(round=self.round, replica_id=writer, **ready)
        self.phase = "SAVING"
        self.revision += 1

    def snapshot(self, instance, run_id):
        return copy.deepcopy(
            dict(
                version=VERSION,
                instance=instance,
                run_id=run_id,
                phase=self.phase,
                revision=self.revision,
                stop=self.fault_at is not None,
                reason=self.reason,
                grant=self.grant,
                result=self.result,
                reports={
                    str(replica): dict(shards=self.shards(replica))
                    for replica in range(len(self.config["replica_tokens"]))
                },
            )
        )

    def report(self, command, now):
        rank, replica, data = (
            command.get("rank", 0),
            command.get("replica", 0),
            command["data"],
        )
        require(uint(rank) and rank < self.config["world_size"], "invalid shard rank")
        require(
            isinstance(data, dict)
            and data.get("world_size") == self.config["world_size"]
            and data.get("heartbeat_timeout_seconds")
            == self.config["heartbeat_seconds"],
            "shard topology/budget differs from registration",
        )
        require(
            isinstance(data.get("phase"), str) and bool(data["phase"]),
            "invalid shard phase",
        )
        require(type(data.get("requested")) is bool, "invalid stop request")
        ready = data.get("ready")
        if ready is not None:
            require(
                isinstance(ready, dict)
                and set(ready) == {"state", "signature"}
                and isinstance(ready["state"], dict)
                and isinstance(ready["signature"], str),
                "invalid ready",
            )
        key = (replica, rank)
        old = self.reports.get(key)
        if old:
            require(
                command["seq"] > old["seq"]
                or (command["seq"] == old["seq"] and data == old["data"]),
                "stale or regressed shard report",
            )
            require(
                (old["data"].get("ready") is None or old["data"]["ready"] == ready)
                and (
                    old["data"]["phase"] != "ineligible"
                    or data["phase"] == "ineligible"
                ),
                "frozen state regressed",
            )
            if command["seq"] == old["seq"]:
                return
        self.reports[key] = dict(seq=command["seq"], data=data, at=now)
        if data["phase"] == "ineligible":
            self.excluded.add(replica)
        if data["requested"]:
            self.stop(now, "worker requested stop")
        self.ready(replica)


class ElasticSession(Session):
    """Repeated save events around externally managed TorchFT membership.

    Quorum reports are observations from the training boundary, never a second
    quorum algorithm. Admission rotates credentials but does not start workers.
    """

    def __init__(self, command, now):
        config = copy.deepcopy(command["data"])
        elastic = config.pop("elastic")
        require(
            isinstance(elastic, dict)
            and set(elastic) == {"active_replicas", "recovery_seconds"},
            "invalid elastic registration",
        )
        active = elastic["active_replicas"]
        require(
            isinstance(active, list)
            and bool(active)
            and all(
                uint(replica) and replica < len(config["replica_tokens"])
                for replica in active
            )
            and len(set(active)) == len(active),
            "invalid initial active replicas",
        )
        budget = elastic["recovery_seconds"]
        require(
            type(budget) in (int, float) and math.isfinite(budget) and budget > 0,
            "invalid recovery budget",
        )
        super().__init__(dict(command, data=config), now)
        self.owner_command = command
        self.active = set(active)
        self.initial_replicas = sorted(active)
        self.members = tuple(sorted(active))
        self.generations = [0 for _ in config["replica_tokens"]]
        self.used_tokens = set(config["replica_tokens"]) | {self.owner}
        self.member_started = dict.fromkeys(active, now)
        self.excluded = set(range(len(config["replica_tokens"]))) - self.active
        self.membership_epoch = 0
        self.event_id = 0
        self.event_kind = None
        self.event_result = None
        self.history = []
        self.recovery_at = None
        self.recovery_seconds = budget
        self.resumed = set()
        self.observed_quorum = None

    def recover(self, now, reason):
        if (
            self.result is not None
            or self.fault_at is not None
            or self.recovery_at is not None
        ):
            return
        self.event_id += 1
        self.event_kind = "baseline"
        self.event_result = None
        self.recovery_at = now
        self.reason, self.phase = reason, "RECOVERING"
        self.resumed.clear()
        self.revision += 1

    def stop(self, now, reason):
        if self.result is not None:
            return
        if self.recovery_at is None:
            self.recover(now, reason)
        # Promotion never resets the save deadline of an event already saving.
        self.event_kind = "terminal"
        super().stop(now, reason)

    def _quorum(self):
        observations = {}
        for replica in sorted(self.active - self.excluded - self.exits.keys()):
            for rank in range(self.config["world_size"]):
                shard = self.reports.get((replica, rank), {}).get("data", {})
                quorum = (
                    shard.get("quorum")
                    if shard.get("event_id") == self.event_id
                    else None
                )
                if quorum is not None:
                    observations[replica, rank] = quorum
        for quorum in observations.values():
            members = tuple(quorum["members"])
            if not set(members) <= self.active - self.excluded - self.exits.keys():
                continue
            required = [
                (replica, rank)
                for replica in members
                for rank in range(self.config["world_size"])
            ]
            if all(observations.get(key) == quorum for key in required):
                return quorum
        return None

    def tick(self, now):
        if self.result is not None:
            return
        if now - self.owner_at > self.config["heartbeat_seconds"]:
            self.finish("launcher heartbeat expired")
            return
        for replica in sorted(self.active - self.excluded - self.exits.keys()):
            for rank in range(self.config["world_size"]):
                report = self.reports.get((replica, rank))
                stale = (
                    now - report["at"] > self.config["heartbeat_seconds"]
                    if report
                    else now - self.member_started[replica]
                    > self.config["startup_seconds"]
                )
                if stale:
                    self.excluded.add(replica)
                    if self.phase == "RESUMING" and replica in self.members:
                        self.finish("member failed before baseline release")
                        return
                    self.recover(now, "missing replica/shard heartbeat")
                    break
        if self.phase == "RESUMING":
            if now - self.fault_at >= self.config["timeout_seconds"]:
                self.finish("resume acknowledgement deadline expired")
            return
        if self.fault_at is None:
            quorum = self._quorum()
            if self.recovery_at is not None:
                if quorum is not None:
                    self.members = tuple(quorum["members"])
                    self.membership_epoch += 1
                    self.observed_quorum = quorum
                    self.fault_at, self.phase = now, "PREPARING"
                    self.revision += 1
                elif now - self.recovery_at >= self.recovery_seconds:
                    self.stop(now, "elastic recovery deadline expired")
            elif quorum is not None and self.observed_quorum != quorum:
                if set(quorum["members"]) < set(self.members):
                    self.recover(now, "TorchFT reported reduced membership")
                    return
                if tuple(quorum["members"]) != self.members:
                    self.membership_epoch += 1
                self.members, self.observed_quorum = tuple(quorum["members"]), quorum
                self.revision += 1
            if self.fault_at is None:
                return
        # Inactive/reserved slots and obsolete process generations must never
        # become pending writers. The base election operates on exclusions.
        super().tick(now)
        if self.grant is not None:
            self.grant.update(
                event_id=self.event_id, membership_epoch=self.membership_epoch
            )

    def ready(self, replica):
        if self.event_kind == "baseline" and replica not in self.members:
            return None
        if any(
            shard and shard.get("event_id") != self.event_id
            for shard in self.shards(replica)
        ):
            return None
        ready = super().ready(replica)
        if ready is not None and self.event_kind == "baseline":
            observed = self.observed_quorum
            if observed is None:
                raise ValueError("Baseline event has no recovered quorum")
            require(
                ready["state"] == observed["state"],
                "baseline differs from recovered boundary",
            )
        return ready

    def report(self, command, now):
        data, replica, rank = (
            command["data"],
            command.get("replica", 0),
            command.get("rank", 0),
        )
        require(
            replica in self.active and replica not in self.excluded,
            "replica generation is not active",
        )
        require(uint(data.get("event_id")), "missing save event identity")
        require(data["event_id"] <= self.event_id, "future save event")
        if data["event_id"] < self.event_id:
            return  # Delayed pre-event heartbeats cannot reintroduce old ready state.
        require(
            type(data.get("recoverable", False)) is bool,
            "invalid recoverable fault flag",
        )
        require(
            type(data.get("terminal_requested", False)) is bool,
            "invalid terminal request flag",
        )
        quorum = data.get("quorum")
        if quorum is not None:
            require(
                isinstance(quorum, dict)
                and set(quorum) == {"id", "members", "state", "torchft"}
                and uint(quorum["id"])
                and isinstance(quorum["members"], list)
                and bool(quorum["members"])
                and all(
                    uint(i) and i < len(self.generations) for i in quorum["members"]
                )
                and quorum["members"] == sorted(set(quorum["members"]))
                and replica in quorum["members"]
                and isinstance(quorum["state"], dict)
                and uint(quorum["state"].get("step"))
                and uint(quorum["state"].get("last_words_num_updates"))
                and isinstance(quorum["torchft"], dict)
                and uint(quorum["torchft"].get("step"))
                and uint(quorum["torchft"].get("batches_committed")),
                "invalid TorchFT boundary observation",
            )
            require(
                data["phase"] == "quorum_ready",
                "quorum must be observed before training",
            )
        old = self.reports.get((replica, rank))
        if (
            old is not None
            and old["data"].get("ready") is not None
            and self.event_result is not None
            and (replica, rank) in self.resumed
            and self.phase in {"RESUMING", "RUNNING"}
            and data["phase"] == "completed"
            and data.get("ready") is None
        ):
            require(command["seq"] > old["seq"], "stale release report")
            # Only an authenticated shard which acknowledged this validated
            # event may relinquish its immutable frozen report.
            old["data"] = dict(old["data"], ready=None, phase="completed")
        if old and old["data"].get("ready") is not None:
            require(self.fault_at is not None, "frozen report outside save event")
        if data.get("terminal_requested"):
            self.stop(now, "worker requested terminal save")
        # Requested acknowledges a freeze and can arrive after the final resume
        # acknowledgement released this event. Only the explicit terminal flag
        # can promote it to a terminal save; a delayed heartbeat is not a fault.
        super().report(dict(command, data=dict(data, requested=False)), now)
        if data.get("recoverable"):
            self.recover(now, "worker reported recoverable fault")

    def apply(self, command, now):
        op, data = command["op"], command.get("data")
        if op == "owner_tick":
            require(isinstance(data, dict), "exits must be an object")
            for key, code in data.items():
                require(
                    key.isdigit()
                    and int(key) in self.active
                    and type(code) is int
                    and -(2**63) <= code < 2**63,
                    "invalid replica exit",
                )
                replica = int(key)
                require(
                    replica not in self.exits or self.exits[replica] == code,
                    "exit changed",
                )
                newly_exited = replica not in self.exits
                self.exits[replica] = code
                if code != 0 and newly_exited:
                    self.excluded.add(replica)
                    if self.phase == "RESUMING" and replica in self.members:
                        self.finish("member failed before baseline release")
                        return
                    # A reduced native quorum may have saved and resumed before
                    # torchrun finishes reaping its already excluded peer.
                    if replica in self.members:
                        self.recover(now, "worker exited")
        elif op == "admit":
            require(
                self.phase == "RUNNING" and self.result is None,
                "admission closed during save/recovery",
            )
            require(
                isinstance(data, dict)
                and set(data) == {"replica_id", "generation", "token", "fenced"},
                "invalid admission",
            )
            replica, token = data["replica_id"], data["token"]
            require(
                uint(replica) and replica < len(self.generations),
                "invalid admission slot",
            )
            require(data["fenced"] is True, "old process generation must be fenced")
            require(
                uint(data["generation"])
                and data["generation"] == self.generations[replica] + 1,
                "stale process generation",
            )
            require(
                isinstance(token, str)
                and len(token) >= 32
                and token not in self.used_tokens,
                "admission requires fresh credentials",
            )
            require(
                replica in self.excluded or replica not in self.active,
                "slot is still active",
            )
            self.config["replica_tokens"][replica] = token
            self.used_tokens.add(token)
            self.generations[replica] += 1
            self.excluded.discard(replica)
            self.exits.pop(replica, None)
            self.active.add(replica)
            self.member_started[replica] = now
            for rank in range(self.config["world_size"]):
                self.reports.pop((replica, rank), None)
            self.revision += 1
        elif op == "validated" and self.event_kind == "baseline":
            require(
                self.phase in {"SAVING", "FENCING"}
                and self.grant is not None
                and command.get("round") == self.round
                and command.get("replica", 0) == self.grant["replica_id"],
                "stale baseline validation",
            )
            self.event_result = dict(
                success=True, event_id=self.event_id, grant=copy.deepcopy(self.grant)
            )
            self.history.append(copy.deepcopy(self.event_result))
            self.phase = "RESUMING"
            self.revision += 1
        elif op == "resume_ack":
            replica, rank = command.get("replica", 0), command.get("rank", 0)
            require(
                replica in self.members
                and replica not in self.excluded
                and replica not in self.exits
                and uint(rank)
                and rank < self.config["world_size"],
                "invalid resume shard",
            )
            if (
                self.phase == "RUNNING"
                and data == self.event_id
                and (replica, rank) in self.resumed
            ):
                return  # A bounded HTTP retry may repeat the final release acknowledgement.
            require(self.phase == "RESUMING", "no validated baseline to resume")
            require(data == self.event_id, "stale resume event")
            self.resumed.add((replica, rank))
            if len(self.resumed) == len(self.members) * self.config["world_size"]:
                self.phase = "RUNNING"
                self.fault_at = self.recovery_at = None
                self.grant = None
                self.signature = None
                self.event_kind = None
                self.reason = ""
                # All current-generation reports must be refreshed after release.
                self.reports.clear()
                self.member_started = dict.fromkeys(self.active, now)
                self.revision += 1
        elif op == "recovery_failed":
            self.stop(now, data if isinstance(data, str) else "elastic recovery failed")
        else:
            Coordinator._apply(self, command, now)

    def snapshot(self, instance, run_id):
        snapshot = super().snapshot(instance, run_id)
        snapshot.update(
            elastic=True,
            event_id=self.event_id,
            event_kind=self.event_kind,
            event_result=copy.deepcopy(self.event_result),
            membership_epoch=self.membership_epoch,
            members=list(self.members),
            generations=list(self.generations),
            initial_replicas=list(self.initial_replicas),
            observed_quorum=copy.deepcopy(self.observed_quorum),
            operation_deadline=(
                self.fault_at + self.config["timeout_seconds"]
                if self.fault_at is not None
                else self.recovery_at
                + self.recovery_seconds
                + self.config["timeout_seconds"]
                if self.recovery_at is not None
                else None
            ),
        )
        return snapshot


class Coordinator:
    def __init__(self, clock=time.monotonic):
        self.instance = secrets.token_hex(16)
        self.clock, self.lock, self.sessions = clock, threading.Lock(), {}

    def tick(self):
        with self.lock:
            for session in self.sessions.values():
                try:
                    session.tick(self.clock())
                except (ValueError, KeyError, TypeError):
                    session.finish("invalid replica report")

    def command(self, command):
        require(
            isinstance(command, dict)
            and set(command)
            <= {
                "version",
                "op",
                "run_id",
                "instance",
                "token",
                "seq",
                "replica",
                "rank",
                "round",
                "data",
            },
            "invalid command fields",
        )
        require(command.get("version") == VERSION, "unsupported last words protocol")
        op = command.get("op")
        if op == "capabilities":
            return dict(version=VERSION, instance=self.instance, backend="service")
        require(
            isinstance(command.get("run_id"), str)
            and 0 < len(command["run_id"]) <= 200,
            "invalid run id",
        )
        require(
            isinstance(command.get("token"), str) and len(command["token"]) >= 32,
            "invalid actor",
        )
        require(uint(command.get("seq")) and command["seq"] > 0, "invalid sequence")
        replica = command.get("replica", 0)
        require(uint(replica), "invalid replica")
        with self.lock:
            now, run_id = self.clock(), command["run_id"]
            if op == "register":
                require(
                    command["seq"] == 1 and not command.get("instance"),
                    "invalid registration identity",
                )
                if run_id in self.sessions:
                    session = self.sessions[run_id]
                    require(
                        session.owner_command == command, "run id already registered"
                    )
                else:
                    require(
                        len(self.sessions) < 64
                        and all(s.result is not None for s in self.sessions.values()),
                        "active session or session limit",
                    )
                    session_type = (
                        ElasticSession if "elastic" in command["data"] else Session
                    )
                    self.sessions[run_id] = session_type(copy.deepcopy(command), now)
                return self.sessions[run_id].snapshot(self.instance, run_id)
            require(
                command.get("instance") == self.instance,
                "Coordinator incarnation changed",
            )
            require(run_id in self.sessions, "unknown save session")
            session = self.sessions[run_id]
            owner = hmac.compare_digest(command["token"], session.owner)
            tokens = session.config["replica_tokens"]
            worker = replica < len(tokens) and hmac.compare_digest(
                command["token"], tokens[replica]
            )
            require(owner or worker, "invalid actor")
            require(
                op in {"poll", "shard_report", "resume_ack"} or owner,
                "operation requires launcher",
            )
            require(
                op not in {"shard_report", "resume_ack"} or worker,
                "report requires replica token",
            )
            session.tick(now)  # Expired deadlines cannot be refreshed by late traffic.
            if op == "poll" or session.result is not None:
                return session.snapshot(self.instance, run_id)
            if owner:
                require(
                    command["seq"] > session.owner_seq
                    or (
                        command["seq"] == session.owner_seq
                        and command == session.owner_command
                    ),
                    "stale launcher sequence",
                )
                if command["seq"] == session.owner_seq:
                    return session.snapshot(self.instance, run_id)
            if op == "shard_report" and isinstance(session, ElasticSession):
                # A fenced actor can still send an in-flight heartbeat while
                # torchrun stops its other shard. Reject it before the report
                # mutation guard: it cannot invalidate the live writer's event.
                require(
                    replica in session.active and replica not in session.excluded,
                    "replica generation is not active",
                )
            try:
                if isinstance(session, ElasticSession):
                    session.apply(command, now)
                else:
                    self._apply(session, command, now)
                if owner:
                    session.owner_seq, session.owner_command, session.owner_at = (
                        command["seq"],
                        copy.deepcopy(command),
                        now,
                    )
                session.tick(now)
            except (ValueError, KeyError, TypeError):
                # Malformed authenticated shard data may already have been stored;
                # fail closed so a background tick cannot issue an unsafe grant.
                if op == "shard_report":
                    session.finish("invalid replica report")
                raise
            return session.snapshot(self.instance, run_id)

    @staticmethod
    def _apply(session, command, now):
        op, data, replica = (
            command["op"],
            command.get("data"),
            command.get("replica", 0),
        )
        if op == "owner_tick":
            require(isinstance(data, dict), "exits must be an object")
            exits = {}
            for key, code in data.items():
                require(
                    key.isdigit()
                    and int(key) < len(session.config["replica_tokens"])
                    and type(code) is int
                    and -(2**63) <= code < 2**63,
                    "invalid replica exit",
                )
                key = int(key)
                require(
                    key not in session.exits or session.exits[key] == code,
                    "exit changed",
                )
                exits[key] = code
            session.exits.update(exits)
            if any(exits.values()):
                session.stop(now, "worker exited")
        elif op == "stop":
            session.stop(
                now, data if isinstance(data, str) else "launcher requested stop"
            )
        elif op == "fail":
            session.finish(
                data if isinstance(data, str) and data else "launcher failed"
            )
        elif op == "shard_report":
            session.report(command, now)
        elif op in {"fenced", "validated"}:
            require(session.grant is not None, "no active grant")
            require(
                command.get("round") == session.round
                and session.grant["replica_id"] == replica,
                "stale writer round",
            )
            if op == "validated":
                session.finish()
            else:
                require(
                    session.phase == "FENCING" and data is True, "fencing not confirmed"
                )
                session.excluded.add(replica)
                session.grant, session.phase = None, "PREPARING"
                session.revision += 1
        else:
            raise ValueError("unknown last words operation")


class ServiceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        require(
            address[0] == "127.0.0.1",
            "Only loopback is supported; multi-host fencing is not implemented",
        )
        self.coordinator = Coordinator()
        self.stopped = threading.Event()
        super().__init__(address, Handler)
        self.ticker = threading.Thread(target=self._tick, daemon=True)
        self.ticker.start()

    def _tick(self):
        while not self.stopped.wait(0.1):
            self.coordinator.tick()

    def server_close(self):
        self.stopped.set()
        self.ticker.join(timeout=2)
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, format, *args):
        pass  # Never log tokens, request bodies, or capabilities.

    def respond(self, code, data):
        raw = json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/status":
            self.respond(404, dict(error="unknown endpoint"))
            return
        server = cast("ServiceServer", self.server)
        self.respond(
            200,
            dict(
                version=VERSION, backend="service", instance=server.coordinator.instance
            ),
        )

    def do_POST(self):
        try:
            require(self.path == "/coordinate", "unknown endpoint")
            require(
                not self.headers.get("Transfer-Encoding"),
                "chunked requests unsupported",
            )
            length = int(self.headers.get("Content-Length", "0"))
            require(
                0 < length <= MAX_REQUEST_BYTES, "report exceeds 1 MiB or missing body"
            )
            raw = self.rfile.read(length)
            require(len(raw) == length, "incomplete request")
            command = json.loads(
                raw,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError("nonfinite JSON")
                ),
            )
            result = cast("ServiceServer", self.server).coordinator.command(command)
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            self.close_connection = True
            self.respond(400, dict(error=str(error)))
            return
        self.respond(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1:0")
    parser.add_argument("--address-file", type=Path)
    args = parser.parse_args()
    host, port = args.bind.rsplit(":", 1)
    with ServiceServer((host, int(port))) as server:
        record = json.dumps(
            dict(address=f"http://127.0.0.1:{server.server_port}", backend="service")
        )
        if args.address_file:
            temporary = args.address_file.with_suffix(".tmp")
            temporary.write_text(record)
            temporary.replace(args.address_file)
        print(record, flush=True)
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
