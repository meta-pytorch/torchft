# RFC: Last-words save coordination and consistent Manager snapshots

## Summary

We would like to discuss an experimental last-words mechanism for distributed
training: when ordinary replica recovery cannot safely continue, a surviving
replica with complete state may publish a final recoverable checkpoint before
the job exits. Saving is best effort, and refusing an unsafe save is an explicit
outcome.

This RFC proposes the TorchFT portion: an independent coordination service,
lightweight clients, and Manager hooks for observing committed state and
exporting consistent recovery snapshots. A paired TorchTitan proposal owns the
training transaction and checkpoint integration. The implementation is a WIP for
discussion, not a request to stabilize the APIs or merge before validation.

## Motivation

A worker or accelerator failure may make training collectives unusable while
another replica still holds useful, fully committed model and optimizer state.
Reverting to the last periodic checkpoint can discard that work. Conversely,
saving whichever tensors remain accessible can publish inconsistent state if an
optimizer update, healing operation, or data-cursor advancement is incomplete.

The control path therefore needs to distinguish training membership from save
authorization, retain a way to coordinate after training work fails, and avoid
starting a second writer until the previous writer is confirmed fenced.

## Example use case

1. Replica groups train using the existing Lighthouse and Manager quorum.
2. A failure is reported. The application determines whether it can recover the
   interrupted attempt or needs to stop and attempt a final save.
3. Each surviving shard reports the completeness of its last committed state.
4. The service grants a save round to an eligible complete replica. This grant
   does not authorize another optimizer update.
5. The application writes and validates a full checkpoint. The owner records the
   validation outcome and confirms that owned workers have exited.
6. A subsequent job may restore that checkpoint after independent validation.
   If no complete replica or valid publication exists, it reports failure and
   retains diagnostics rather than presenting partial state as recoverable.

## Expected behavior and requirements

### Keep the existing training authority

Lighthouse and Manager continue to decide membership, quorum and commit results.
The service coordinates save sessions and writer ownership; it must not create
an alternative training quorum. `Manager.quorum_snapshot()` is a read-only
observation of a completed healthy quorum. A pending or failed quorum, incomplete
healing, or inconsistent membership cannot supply an eligible observation.

### Protect the commit and snapshot boundaries

`LastWordsManager` records one commit decision per attempt. A timed-out commit
has an unknown outcome and cannot be treated as a rejected update that is safe
to retry. An authorized update cannot be discarded as an uncommitted attempt.

Registered application state is copied on the training thread before background
transport consumes it. Restoring tensor leaves preserves live DTensor placement
and validates structure, shape and dtype before copying. A partial restore or
failed completion callback leaves the replica ineligible for saving.

### Separate service lifetime from training imports

The same distribution provides a lightweight `torchft_last_words` package. Its
HTTP service and clients use the standard library and can start without importing
torch, the native TorchFT extension, or a model registry. The current prototype
uses loopback HTTP and a single-host owner; multi-host orchestration and durable
service failover are outside its implemented scope.

Messages identify the service instance, run, replica, sequence, generation and
save round. Stale credentials and replayed transitions must not affect a newer
session. Service restart invalidates the old instance's session. Credentials
and response files are private deployment artifacts.

### Make writer fencing and outcomes explicit

The owner confirms termination of the old writer before reporting `fenced` and
allowing re-election. The service records `validated` only after the application
owner has checked storage. It does not parse DCP or independently prove that the
checkpoint bytes are correct. The application must also verify worker exits;
successful publication alone is not a clean job termination.

## Scope and proposed interfaces

| Component | Responsibility |
| --- | --- |
| `torchft_last_words.service` | Save-session state machine, writer grants, generations and timeouts |
| `ServiceClient`, `SessionClient`, `OwnerClient`, `WorkerTransport` | HTTP transport, identity envelopes, owner and worker operations |
| `QuorumSnapshot` | Immutable quorum ID, logical members and native progress |
| `LastWordsManager` | Single commit decision, training-thread snapshots, restoration and application callbacks |
| Application / paired TorchTitan integration | Complete-state eligibility, checkpoint schema, storage validation, cursors, supervisor and retention |
| Device backend | Synchronization and recovery of device work and communicators |

The service exposes `GET /status` and `POST /coordinate` with wire version 1.
The candidate Manager hooks are `set_last_words_hooks`, `begin_step`,
`discard_interrupted_step`, and `restoration_incomplete`, alongside existing
state-dict registration and quorum/commit APIs. See [the interface notes](last_words.md)
for signatures and threading constraints.

The initial LastWordsManager contract requires synchronous quorum and healing.
Unsupported asynchronous combinations fail during configuration, before native
startup, rather than exposing a premature training boundary. This restriction
applies to standalone users as well as the paired TorchTitan integration.

The current logical-slot representation uses unique integer suffixes in replica
IDs, as the paired application does. General identity mapping, asynchronous
quorums and fixed-with-spares need further API design and coverage; they must not
be inferred to work from synchronous paired-application tests.

The WIP preserves upstream native process-group reconfiguration. Earlier
candidate runs used an older native TorchFT ABI and are not validation of this
rebased implementation. Control/HTTP tests, Manager unit tests, native-extension
builds and end-to-end recovery need separate results. Accelerator recovery,
numerical equivalence and performance are not accepted for this WIP.

## Questions for discussion

1. Does a standalone save coordinator belong in TorchFT, and should its
   lightweight package ship in the same distribution?
2. Which consistency and threading guarantees should Manager expose without
   requiring application access to private recovery state?
3. Should logical save identities be strings or explicit application slots,
   especially for asynchronous quorum and spare-replica configurations?
4. What evidence should be required before replacing a failed writer?
5. Are the current fail-closed commit-unknown and partial-healing outcomes the
   appropriate initial contract?
6. What native recovery, shutdown and checkpoint-publication tests should be
   mandatory before removing the WIP designation?
