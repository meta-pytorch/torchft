# Last-words coordination and Manager hooks

This document describes the experimental split rebased onto upstream main.
It is not documentation for an unmodified published wheel. The candidate adds
Python coordination and Manager APIs while preserving upstream's native
process-group reconfiguration. Build the native extension from this checkout;
earlier results with the original 0.2.0 extension do not validate this branch.
Real-device functional, precision and performance acceptance is incomplete.

## Ownership

Lighthouse and Manager retain ownership of training membership, quorum, commit
decisions and peer healing. The standalone service coordinates application save
sessions, writer grants and generations. A writer grant does not authorize an
optimizer update, and a quorum observation does not authorize a checkpoint save.

The application owns complete-state eligibility, checkpoint format, storage
validation, publication and dataloader cursors. Its supervisor starts and fences
owned processes and validates the published result. The service neither loads
model tensors nor interprets DCP checkpoints or starts training processes.
Device synchronization and communicator recovery remain backend responsibilities.

## Standalone service

The same TorchFT distribution contains the lightweight top-level package
`torchft_last_words`. Starting it does not import `torchft.__init__`, torch or the
native extension:

```bash
python -m torchft_last_words --bind 127.0.0.1:0 --address-file /private/run/address.json
# Equivalent installed command:
torchft_last_words --bind 127.0.0.1:0 --address-file /private/run/address.json
```

The address-file parent must already exist. Port zero selects an available port;
the service prints the actual address and atomically replaces the address file.
The current deployment supports single-host loopback HTTP and Linux application
supervision. It does not provide multi-host orchestration or durable service
failover. An instance change invalidates an existing session.

The HTTP endpoints are `GET /status` and `POST /coordinate`. The service version
and client `WIRE_VERSION` are 1. Operations include `capabilities`, `register`,
`shard_report`, `poll`, `owner_tick`, `stop`, `fail`, `fenced`, `validated`, `admit`,
`resume_ack` and `recovery_failed`. The application's legacy file-backend version
is a separate contract; it is not the HTTP wire version.

Session messages bind run, service instance, replica, sequence and round, with
worker reports also identifying the shard rank. The owner must confirm old writer
exit before fencing and re-election. `validated` records the owner's successful
storage validation; the service does not independently inspect checkpoint bytes.

## Standard-library client APIs

Import these classes from `torchft_last_words.client`:

| Interface | Contract |
| --- | --- |
| `ServiceClient(address)` | Accepts a loopback HTTP address with an explicit port. `call(document)` sends a JSON string and returns the response string. `close()` closes the connection. |
| `SessionClient(client, run_id, token, *, instance="", replica=0)` | `call(op, data=None, *, round=0, replica=None)` supplies the wire envelope and checks the returned version, run and instance. |
| `OwnerClient(run_id, *, address, num_replicas, world_size, heartbeat_timeout, startup_timeout, timeout, elastic=None)` | Registers a session. `environment(replica)` returns worker credentials. `admit_external(replica, *, generation, token, fenced)` requests explicit generation admission. `close()` closes its transport. |
| `WorkerTransport(run_id, replica, rank)` | Reads owner-supplied environment. Provides `report(status, requested)`, `get()`, `poll()`, `set_ready(state, signature)` and `resume_ack()`. |

The transport uses a two-second socket timeout and serializes calls per client.
It replays the same serialized request at most once after a socket/HTTP transport
failure; application code must not generate a new sequence as a substitute for
that replay. Oversized responses, protocol errors and instance/run mismatches
fail explicitly. A transport error is not a new writer grant.

Stop application reporting threads before closing their transport. The owner
closes `OwnerClient`; a `WorkerTransport` caller owns its `client` and closes that
`ServiceClient` after reporting stops. Closing either client does not shut down
the service process. The supervisor owns that process's shutdown and exit check.
Worker environments contain session credentials and belong in private launch
configuration rather than general diagnostic output.

## Manager integration

`torchft.last_words.LastWordsManager` subclasses the existing Manager. Its device
injection points are keyword arguments `synchronize_device`,
`process_group_usable` and `barrier_after_initial_quorum`; ordinary Manager
construction arguments remain applicable. The first two callbacks take no
arguments. Synchronization must complete device work before restoration is
declared complete, and the usability callback returns a boolean.

Bind application hooks before the first quorum:

```python
manager.set_last_words_hooks(
    request=request_stop,
    stop_requested=is_stop_requested,
    terminal_stop_requested=is_terminal_stop_requested,
    training_boundary=on_training_boundary,
    after_heal=on_complete_heal,
    elastic_enabled=True,
)
```

`request(reason)` receives a reason string. The two stop predicates take no
arguments and return booleans. `training_boundary()` and `after_heal()` take no
arguments; recovery completion runs on the training thread after synchronization.
These hooks transfer application policy, not ownership of Manager internals.
Hook exceptions propagate; applications must not turn them into save eligibility.

| API | Required application behavior |
| --- | --- |
| `begin_step()` | Establish quorum/healing before consuming the next attempt's input. The boundary callback still needs the application's completed-state checks. |
| `should_commit(timeout=None)` | Obtain at most one decision per attempt. A true result authorizes the update; it does not prove the application update has completed. |
| `discard_interrupted_step(error)` | Use only for an interrupted attempt whose update was not authorized and whose decision is not unknown. Authorized or unknown attempts cannot be silently discarded or replayed. |
| `register_state_dict_fn(key, load_state_dict, state_dict)` | Register complete, mutually consistent application state. State export occurs on the training thread and snapshots tensor leaves to CPU; restoration preserves existing destination tensor placement. |
| `restoration_incomplete` | Refuse saving while a partial load, device synchronization or after-heal callback remains incomplete or has failed. |
| `quorum_snapshot()` | Read a completed healthy observation without initiating another quorum. Call on the training thread after synchronous quorum/healing. |

The immutable `torchft_last_words.state.QuorumSnapshot` contains `quorum_id`,
`members`, `step` and `batches_committed`. Pending, cancelled, failed or incomplete
quorums do not supply an eligible snapshot. Backend plugins may additionally
validate native rank-to-logical-replica mapping.

`TorchFTCommitTimeoutError` marks an unknown commit result and requires replica
restart rather than local replay. `TorchFTQuorumStoppedError` unwinds a CPU quorum
wait for a terminal application stop; it does not prove the native RPC was
cancelled. Applications must preserve device fencing and process-exit checks.
Use the inherited Manager shutdown API with an appropriate ownership policy;
neither a hook nor HTTP success guarantees that every native operation exited.

## Acceptance boundaries

The split candidate's Manager/service tests cover Python coordination and HTTP
contracts. CPU fixtures do not establish real-device quorum, model compilation or
fault recovery. Applications must separately verify complete checkpoint and cursor
restoration, post-recovery commits and zero-exit process closure. Functional
recovery with deterministic execution disabled does not establish exact training
trajectory equivalence. Model-only compilation does not establish loss compilation.
