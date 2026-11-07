# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Standard-library HTTP clients for TorchFT emergency-save coordination.

This module must remain standard-library-only: launcher imports cannot initialize
torch/NPU. Control files here are audit copies, never authorization inputs.
"""

import http.client
import json
import os
import secrets
import threading
from urllib.parse import urlsplit

WIRE_VERSION = 1


class ServiceClient:
    """Bounded persistent HTTP transport; imports no tensor or native extension."""

    def __init__(self, address):
        parsed = urlsplit(address)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or not parsed.port
            or parsed.path not in ("", "/")
        ):
            raise ValueError(
                "The single-host coordination service requires a loopback HTTP endpoint"
            )
        self.connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port, timeout=2
        )
        self.lock = threading.Lock()

    def call(self, document) -> str:
        with self.lock:
            for attempt in range(2):
                try:
                    self.connection.request(
                        "POST",
                        "/coordinate",
                        body=document.encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    response = self.connection.getresponse()
                    raw = response.read(8 * 1024 * 1024 + 1)
                    if len(raw) > 8 * 1024 * 1024:
                        raise RuntimeError("Coordinator response exceeds limit")
                    result = raw.decode()
                    if response.status != 200:
                        raise RuntimeError(
                            json.loads(result).get(
                                "error", "Coordinator request failed"
                            )
                        )
                    return result
                except (OSError, http.client.HTTPException):
                    # The server may close an idle keep-alive socket during DCP
                    # validation. Replay exactly the same sequence/document once;
                    # the authority rejects stale generations and deduplicates it.
                    self.connection.close()
                    if attempt:
                        raise
                except BaseException:
                    self.connection.close()
                    raise
        raise RuntimeError("Coordinator transport exhausted its retry budget")

    def close(self):
        with self.lock:
            self.connection.close()


class SessionClient:
    def __init__(self, client, run_id, token, *, instance="", replica=0):
        self.client, self.run_id, self.token = client, run_id, token
        self.instance, self.replica = instance, replica
        self.seq = 0
        self.lock = threading.Lock()

    def call(self, op, data=None, *, round=0, replica=None):
        with self.lock:
            self.seq += 1
            document = dict(
                version=WIRE_VERSION,
                op=op,
                run_id=self.run_id,
                token=self.token,
                instance=self.instance,
                seq=self.seq,
                replica=self.replica if replica is None else replica,
                round=round,
                data=data,
            )
            result = json.loads(self.client.call(json.dumps(document)))
            if result.get("version") != WIRE_VERSION:
                raise RuntimeError("Incompatible Coordinator last words protocol")
            if self.instance and result.get("instance") != self.instance:
                raise RuntimeError(
                    "Coordinator instance changed; terminate this attempt"
                )
            if op != "capabilities" and result.get("run_id") != self.run_id:
                raise RuntimeError("Coordinator returned another save session")
            return result


class OwnerClient:
    """Register sessions and distribute credentials without checkpoint storage policy."""

    def __init__(
        self,
        run_id,
        *,
        address,
        num_replicas,
        world_size,
        heartbeat_timeout,
        startup_timeout,
        timeout,
        elastic=None,
    ):
        self.run_id = run_id
        self.address = address
        self.tokens = [secrets.token_hex(32) for _ in range(num_replicas)]
        self.session = SessionClient(
            ServiceClient(address), run_id, secrets.token_hex(32)
        )
        registration = dict(
            world_size=world_size,
            replica_tokens=self.tokens,
            heartbeat_seconds=heartbeat_timeout,
            startup_seconds=startup_timeout,
            timeout_seconds=timeout,
        )
        if elastic is not None:
            registration["elastic"] = elastic
        try:
            self.snapshot = self.session.call("register", registration)
        except BaseException:
            self.session.client.close()
            raise
        self.session.instance = self.snapshot["instance"]
        self.world_size = world_size
        self.heartbeat_interval = min(1, heartbeat_timeout / 3)

    def environment(self, replica):
        environment = dict(
            LAST_WORDS_COORDINATION_BACKEND="service",
            LAST_WORDS_COORDINATOR_ADDRESS=self.address,
            LAST_WORDS_COORDINATOR_INSTANCE=self.session.instance,
            LAST_WORDS_REPLICA_TOKEN=self.tokens[replica],
            LAST_WORDS_ELASTIC_ENABLED="1" if self.snapshot.get("elastic") else "0",
        )
        if self.snapshot.get("elastic"):
            environment["LAST_WORDS_REPLICA_GENERATION"] = str(
                self.snapshot["generations"][replica]
            )
            environment["LAST_WORDS_INITIAL_REPLICAS"] = json.dumps(
                self.snapshot["initial_replicas"]
            )
        return environment

    def admit_external(self, replica, *, generation, token, fenced):
        """Rotate an external generation only after its owner confirms fencing."""
        self.snapshot = self.session.call(
            "admit",
            dict(replica_id=replica, generation=generation, token=token, fenced=fenced),
        )
        self.tokens[replica] = token

    def close(self):
        self.session.client.close()


class WorkerTransport:
    def __init__(self, run_id, replica, rank):
        self.client = ServiceClient(os.environ["LAST_WORDS_COORDINATOR_ADDRESS"])
        self.identity = dict(
            version=WIRE_VERSION,
            run_id=run_id,
            instance=os.environ["LAST_WORDS_COORDINATOR_INSTANCE"],
            token=os.environ["LAST_WORDS_REPLICA_TOKEN"],
            replica=replica,
            rank=rank,
        )
        self.lock = threading.Lock()
        self.seq = 0
        self.snapshot = None
        self.ready = None
        self.ready_event = None
        self.event_id = 0
        self.elastic_enabled = os.environ.get("LAST_WORDS_ELASTIC_ENABLED") == "1"

    def report(self, status, requested):
        with self.lock:
            self.seq += 1
            status = dict(status, requested=requested, ready=self.ready)
            if self.elastic_enabled:
                status["event_id"] = self.event_id
                if self.ready_event != self.event_id:
                    status["ready"] = None
            document = dict(self.identity, seq=self.seq, op="shard_report", data=status)
            snapshot = json.loads(self.client.call(json.dumps(document)))
            if any(
                snapshot.get(key) != self.identity[key]
                for key in ("version", "run_id", "instance")
            ):
                raise RuntimeError(
                    "Coordinator response belongs to another save session"
                )
            self.snapshot = snapshot
            self.event_id = snapshot.get("event_id", 0)
            return snapshot

    def get(self):
        with self.lock:
            return self.snapshot

    def poll(self):
        with self.lock:
            self.seq += 1
            snapshot = json.loads(
                self.client.call(
                    json.dumps(dict(self.identity, seq=self.seq, op="poll", data=None))
                )
            )
            if any(
                snapshot.get(key) != self.identity[key]
                for key in ("version", "run_id", "instance")
            ):
                raise RuntimeError("Coordinator poll belongs to another save session")
            self.snapshot = snapshot
            self.event_id = snapshot.get("event_id", 0)
            return snapshot

    def set_ready(self, state, signature):
        with self.lock:
            ready = dict(state=state, signature=signature)
            if (
                self.ready is not None
                and self.ready_event == self.event_id
                and self.ready != ready
            ):
                raise RuntimeError("Cannot change frozen ready state")
            self.ready = ready
            self.ready_event = self.event_id

    def resume_ack(self):
        with self.lock:
            self.seq += 1
            snapshot = json.loads(
                self.client.call(
                    json.dumps(
                        dict(
                            self.identity,
                            seq=self.seq,
                            op="resume_ack",
                            data=self.event_id,
                        )
                    )
                )
            )
            if any(
                snapshot.get(key) != self.identity[key]
                for key in ("version", "run_id", "instance")
            ):
                raise RuntimeError("Resume acknowledgement belongs to another session")
            self.snapshot = snapshot
            self.ready = self.ready_event = None
            return snapshot
