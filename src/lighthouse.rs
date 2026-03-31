// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use core::net::SocketAddr;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;

use anyhow::Result;
use anyhow::anyhow;
use askama::Template;
use axum::Router;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::Html;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::routing::post;
use gethostname::gethostname;
use log::error;
use log::info;
use structopt::StructOpt;
use tokio::sync::Mutex;
use tokio::sync::broadcast;
use tokio::task::JoinSet;
use tokio::time::interval;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::service::Routes;
use tonic::transport::Server;
use tonic::transport::server::TcpIncoming;

use crate::manager::manager_client_new;
use crate::torchftpb::KillRequest;
use crate::torchftpb::LighthouseHeartbeatRequest;
use crate::torchftpb::LighthouseHeartbeatResponse;
use crate::torchftpb::LighthouseQuorumRequest;
use crate::torchftpb::LighthouseQuorumResponse;
use crate::torchftpb::Quorum;
use crate::torchftpb::QuorumMember;
use crate::torchftpb::lighthouse_service_server::LighthouseService;
use crate::torchftpb::lighthouse_service_server::LighthouseServiceServer;

#[derive(Clone)]
struct QuorumMemberDetails {
    joined: Instant,
    member: QuorumMember,
}

struct State {
    channel: broadcast::Sender<Quorum>,
    participants: HashMap<String, QuorumMemberDetails>,
    prev_quorum: Option<Quorum>,
    quorum_id: i64,

    // heartbeat information
    // replica_id -> last heartbeat
    heartbeats: HashMap<String, Instant>,
}

pub struct Lighthouse {
    state: Mutex<State>,
    opt: LighthouseOpt,
    listener: Mutex<Option<tokio::net::TcpListener>>,
    local_addr: SocketAddr,
    change_logger: ChangeLogger,
}

struct ChangeLogger {
    last_reason: std::sync::Mutex<Option<String>>,
}
impl ChangeLogger {
    fn new() -> Self {
        ChangeLogger {
            last_reason: std::sync::Mutex::new(None),
        }
    }
    fn log_if_changed(&self, reason: &str) {
        let mut last_reason = self.last_reason.lock().unwrap();
        if last_reason.as_deref() != Some(reason) {
            info!("Quorum status: {}", reason);
            *last_reason = Some(reason.to_string());
        }
    }
}

#[derive(StructOpt, Debug)]
#[structopt()]
pub struct LighthouseOpt {
    // bind is the address to bind the server to.
    #[structopt(
        long = "bind",
        default_value = "[::]:29510",
        help = "Address to bind the server to"
    )]
    pub bind: String,

    #[structopt(
        long = "join_timeout_ms",
        default_value = "60000",
        help = "How long to wait for heartbeating stragglers to join before issuing quorum"
    )]
    pub join_timeout_ms: u64,

    #[structopt(
        long = "min_replicas",
        help = "Minimum number of replicas to consider a quorum"
    )]
    pub min_replicas: u64,

    #[structopt(
        long = "quorum_tick_ms",
        default_value = "100",
        help = "How frequently to check for quorum when waiting for stragglers."
    )]
    pub quorum_tick_ms: u64,

    #[structopt(
        long = "heartbeat_timeout_ms",
        default_value = "5000",
        help = "How long to wait for a heartbeat before considering a replica dead."
    )]
    pub heartbeat_timeout_ms: u64,

    #[structopt(
        long = "expected_replicas",
        use_delimiter = true,
        value_delimiter = ",",
        help = "Expected number of replicas (comma-separated, strictly increasing, e.g., '1,2,4')"
    )]
    pub expected_replicas: Option<Vec<u64>>,
}

fn quorum_changed(a: &Vec<QuorumMember>, b: &Vec<QuorumMember>) -> bool {
    let a_ids: Vec<&String> = a.iter().map(|p| &p.replica_id).collect();
    let b_ids: Vec<&String> = b.iter().map(|p| &p.replica_id).collect();

    return a_ids != b_ids;
}

// Find the expected replica count based on candidate participant count.
// Returns the largest value in expected_replicas that is <= current_count.
fn find_expected_replica_count(expected_replicas: &[u64], candidate_count: usize) -> Option<usize> {
    expected_replicas
        .iter()
        .filter(|&&n| n <= candidate_count as u64)
        .max()
        .map(|&n| n as usize)
}

// Apply expected_replicas truncation while preserving prev_quorum participants.
// When truncating, this function:
// 1. Preserves all prev_quorum participants (to avoid disrupting running training)
// 2. Fills remaining slots with new participants (sorted by smallest replica IDs)
// 3. Returns the truncated participant list
fn apply_expected_replicas_truncation(
    mut participants: Vec<QuorumMember>,
    expected_replicas: &Option<Vec<u64>>,
    prev_quorum: &Option<Quorum>,
) -> Vec<QuorumMember> {
    if let Some(ref expected) = expected_replicas {
        if let Some(expected_count) = find_expected_replica_count(expected, participants.len()) {
            if expected_count < participants.len() {
                // If prev_quorum exists, preserve its participants
                if let Some(ref prev) = prev_quorum {
                    let prev_replica_ids: HashSet<&String> =
                        prev.participants.iter().map(|p| &p.replica_id).collect();

                    // Separate into prev_quorum participants (already training) and new participants
                    let mut prev_participants: Vec<QuorumMember> = participants
                        .iter()
                        .filter(|p| prev_replica_ids.contains(&p.replica_id))
                        .cloned()
                        .collect();

                    let mut new_participants: Vec<QuorumMember> = participants
                        .iter()
                        .filter(|p| !prev_replica_ids.contains(&p.replica_id))
                        .cloned()
                        .collect();

                    // Calculate how many new participants we can add
                    // If prev_participants exceeds expected_count, we need to shrink prev_participants too
                    if prev_participants.len() > expected_count {
                        // Shrink case: truncate prev_participants to expected_count
                        // Keep participants with smallest replica IDs (consistent with expansion logic)
                        prev_participants.sort_by_key(|p| p.replica_id.clone());
                        prev_participants.truncate(expected_count);
                        participants = prev_participants;
                    } else {
                        // Expansion or stable case: keep all prev_participants and add new ones if there's room
                        let remaining_slots = expected_count - prev_participants.len();

                        // Truncate new participants (they are already sorted by ID)
                        // Keep only the ones with smallest IDs
                        new_participants.truncate(remaining_slots);

                        // Combine: all prev participants + selected new participants
                        participants = prev_participants;
                        participants.extend(new_participants);

                        // Sort again to maintain consistent ordering
                        participants.sort_by_key(|p| p.replica_id.clone());
                    }
                } else {
                    // No prev_quorum, just truncate normally (keep smallest IDs)
                    participants.truncate(expected_count);
                }
            }
        }
    }
    participants
}

// Checks whether the quorum is valid, the new quorum and an explanation for the state.
fn quorum_compute(
    now: Instant,
    state: &State,
    opt: &LighthouseOpt,
) -> (Option<Vec<QuorumMember>>, String) {
    let heartbeats = &state.heartbeats;
    let healthy_replicas: HashSet<&String> = heartbeats
        .iter()
        .filter_map(|(replica_id, last_heartbeat)| {
            if now.duration_since(*last_heartbeat) < Duration::from_millis(opt.heartbeat_timeout_ms)
            {
                return Some(replica_id);
            }
            None
        })
        .collect();

    let healthy_participants: HashMap<&String, &QuorumMemberDetails> = state
        .participants
        .iter()
        .filter(|(replica_id, _details)| healthy_replicas.contains(replica_id))
        .collect();

    let mut candidate_participants: Vec<QuorumMember> = healthy_participants
        .values()
        .map(|details| details.member.clone())
        .collect();

    // Sort by replica ID to get a consistent ordering across runs.
    // This ensures that when truncating based on expected_replicas,
    // we keep participants with smaller IDs and remove those with larger IDs.
    candidate_participants.sort_by_key(|p| p.replica_id.clone());

    let shrink_only = healthy_participants
        .iter()
        .any(|(_, details)| details.member.shrink_only);

    let metadata = format!(
        "[{}/{} participants healthy][{} heartbeating][shrink_only={}]",
        healthy_participants.len(),
        state.participants.len(),
        healthy_replicas.len(),
        shrink_only,
    );

    // Check if we can use the previous quorum.
    // TODO: do we still need this given we have heartbeats?
    if state.prev_quorum.is_some() {
        let prev_quorum = state.prev_quorum.as_ref().unwrap();

        let prev_replica_ids: HashSet<&String> = prev_quorum
            .participants
            .iter()
            .map(|p| &p.replica_id)
            .collect();

        if shrink_only {
            candidate_participants = candidate_participants
                .into_iter()
                .filter(|p| prev_replica_ids.contains(&p.replica_id))
                .collect();
        }

        // Fast quorum is when all previous participants are still in the quorum
        // and we have enough participants to form a quorum.
        let is_fast_quorum = prev_quorum
            .participants
            .iter()
            .all(|prev_member| healthy_participants.contains_key(&prev_member.replica_id));

        if is_fast_quorum {
            // Apply expected_replicas constraint if set
            // Preserve prev_quorum participants (already training) when truncating
            let result_participants = apply_expected_replicas_truncation(
                candidate_participants.clone(),
                &opt.expected_replicas,
                &state.prev_quorum,
            );

            let final_metadata = format!(
                "[{}/{} participants selected][{}/{} participants healthy][{} heartbeating][shrink_only={}]",
                result_participants.len(),
                candidate_participants.len(),
                healthy_participants.len(),
                state.participants.len(),
                healthy_replicas.len(),
                shrink_only,
            );

            return (
                Some(result_participants),
                format!("Fast quorum found! {}", final_metadata),
            );
        }
    }

    // Minimum quorum size check.
    if healthy_participants.len() < opt.min_replicas as usize {
        return (
            None,
            format!(
                "New quorum not ready, only have {} participants, need min_replicas {} {}",
                healthy_participants.len(),
                opt.min_replicas,
                metadata
            ),
        );
    }

    // Avoid split brain by requiring at least half of the known alive workers.
    if healthy_participants.len() <= healthy_replicas.len() / 2 {
        return (
            None,
            format!(
                "New quorum not ready, only have {} participants, need at least half of {} healthy workers {}",
                healthy_participants.len(),
                healthy_replicas.len(),
                metadata
            ),
        );
    }

    let all_healthy_joined = healthy_participants.len() == healthy_replicas.len();

    // Quorum is valid at this point but lets wait for stragglers.
    let first_joined = healthy_participants
        .values()
        .map(|details| details.joined)
        .min()
        .unwrap_or(now);
    if !all_healthy_joined
        && now.duration_since(first_joined) < Duration::from_millis(opt.join_timeout_ms)
    {
        return (
            None,
            format!(
                "Valid quorum with {} participants, waiting for {} healthy but not participating stragglers due to join timeout {}",
                healthy_participants.len(),
                healthy_replicas.len() - healthy_participants.len(),
                metadata
            ),
        );
    }

    // Apply expected_replicas constraint if set
    // Preserve prev_quorum participants (already training) when truncating
    let result_participants = apply_expected_replicas_truncation(
        candidate_participants.clone(),
        &opt.expected_replicas,
        &state.prev_quorum,
    );

    let final_metadata = format!(
        "[{}/{} participants selected][{}/{} participants healthy][{} heartbeating][shrink_only={}]",
        result_participants.len(),
        candidate_participants.len(),
        healthy_participants.len(),
        state.participants.len(),
        healthy_replicas.len(),
        shrink_only,
    );

    (
        Some(result_participants),
        format!("Valid quorum found {}", final_metadata),
    )
}

impl Lighthouse {
    pub async fn new(opt: LighthouseOpt) -> Result<Arc<Self>> {
        // Validate expected_replicas constraints
        if let Some(ref expected) = opt.expected_replicas {
            // Check if strictly increasing
            for i in 1..expected.len() {
                if expected[i] <= expected[i - 1] {
                    return Err(anyhow!("expected_replicas must be strictly increasing"));
                }
            }

            // Check if first element equals min_replicas
            if !expected.is_empty() && expected[0] != opt.min_replicas {
                return Err(anyhow!(
                    "expected_replicas[0] ({}) must equal min_replicas ({})",
                    expected[0],
                    opt.min_replicas
                ));
            }
        }

        let listener = tokio::net::TcpListener::bind(&opt.bind).await?;

        let (tx, _) = broadcast::channel(16);

        Ok(Arc::new(Self {
            state: Mutex::new(State {
                participants: HashMap::new(),
                channel: tx,
                prev_quorum: None,
                quorum_id: 0,
                heartbeats: HashMap::new(),
            }),
            opt,
            local_addr: listener.local_addr()?,
            listener: Mutex::new(Some(listener)),
            change_logger: ChangeLogger::new(),
        }))
    }

    fn _quorum_tick(self: Arc<Self>, state: &mut State) -> Result<()> {
        let (quorum_met, reason) = quorum_compute(Instant::now(), state, &self.opt);
        self.change_logger.log_if_changed(&reason);

        if quorum_met.is_some() {
            let participants = quorum_met.unwrap();

            let commit_failure_replica_ids: Vec<String> = participants
                .iter()
                .filter(|p| p.commit_failures > 0)
                .map(|p| p.replica_id.clone())
                .collect();

            // only increment quorum ID if something about the quorum
            // changed (members/addresses/etc)
            if state.prev_quorum.is_none()
                || quorum_changed(
                    &participants,
                    &state.prev_quorum.as_ref().unwrap().participants,
                )
            {
                state.quorum_id += 1;
                info!(
                    "Detected quorum change, bumping quorum_id to {}",
                    state.quorum_id
                );
            } else if commit_failure_replica_ids.len() > 0 {
                state.quorum_id += 1;
                info!(
                    "Detected commit failures in [{}], bumping quorum_id to {}",
                    commit_failure_replica_ids.join(", "),
                    state.quorum_id
                );
            }

            let quorum = Quorum {
                quorum_id: state.quorum_id,
                participants,
                created: Some(SystemTime::now().into()),
            };

            info!("Quorum! {:?}", quorum);

            state.prev_quorum = Some(quorum.clone());
            state.participants.clear();
            match state.channel.send(quorum) {
                Ok(_) => (),
                Err(e) => error!("failed to send quorum {}", e),
            }
        }
        Ok(())
    }

    async fn _run_quorum(self: Arc<Self>) -> Result<()> {
        let mut interval = interval(Duration::from_millis(self.opt.quorum_tick_ms));
        loop {
            interval.tick().await; // Wait for the next tick
            let mut state = self.state.lock().await;
            self.clone()._quorum_tick(&mut state)?;
        }
    }

    pub fn address(&self) -> String {
        format!(
            "http://{}:{}",
            gethostname().into_string().unwrap(),
            self.local_addr.port()
        )
    }

    async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        info!("Lighthouse listening on: {}", self.address());

        let listener = self.listener.lock().await.take().unwrap();
        let incoming =
            TcpIncoming::from_listener(listener, true, None).map_err(|e| anyhow::anyhow!(e))?;

        // Setup HTTP endpoints
        let app = Router::new()
            .route(
                "/",
                get(|| async { Html(IndexTemplate {}.render().unwrap()) }),
            )
            .route(
                "/status",
                get({
                    let self_clone = self.clone();
                    move || async { self_clone.get_status().await }
                }),
            )
            .route(
                "/replica/:replica_id/kill",
                post({
                    let self_clone = self.clone();
                    move |path| async { self_clone.kill(path).await }
                }),
            );

        // register the GRPC service
        let routes = Routes::from(app).add_service(LighthouseServiceServer::new(self));

        Server::builder()
            // allow non-GRPC connections
            .accept_http1(true)
            .add_routes(routes)
            .serve_with_incoming(incoming)
            .await
            .map_err(|e| e.into())
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut set = JoinSet::new();

        set.spawn(self.clone()._run_quorum());

        set.spawn(self.clone()._run_grpc());

        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    }

    async fn get_status(self: Arc<Self>) -> Html<String> {
        let template = {
            let state = self.state.lock().await;

            let (_, quorum_status) = quorum_compute(Instant::now(), &state, &self.opt);

            let max_step = if let Some(quorum) = &state.prev_quorum {
                quorum
                    .participants
                    .iter()
                    .map(|p| p.step)
                    .max()
                    .unwrap_or(-1)
            } else {
                -1
            };

            let num_participants = if let Some(quorum) = &state.prev_quorum {
                quorum.participants.len() as i64
            } else {
                -1
            };

            StatusTemplate {
                quorum_id: state.quorum_id,
                num_participants,
                prev_quorum: state.prev_quorum.clone(),
                quorum_status,
                max_step,

                heartbeats: state.heartbeats.clone(),
                old_age_threshold: Instant::now()
                    .checked_sub(Duration::from_millis(self.opt.heartbeat_timeout_ms))
                    .unwrap_or(Instant::now()),
            }
        };
        Html(template.render().unwrap())
    }

    async fn kill(self: Arc<Self>, Path(replica_id): Path<String>) -> Result<(), AppError> {
        let addr = 'addr: {
            let state = self.state.lock().await;

            if state.prev_quorum.is_none() {
                return Err(AppError(anyhow!("failed to find replica")));
            }

            for member in state.prev_quorum.clone().unwrap().participants {
                if member.replica_id == replica_id {
                    break 'addr member.address;
                }
            }

            return Err(AppError(anyhow!("failed to find replica")));
        };

        let mut client = manager_client_new(addr, Duration::from_secs(10)).await?;

        let request = tonic::Request::new(KillRequest {
            msg: "killed from dashboard".to_string(),
        });
        let _resp = client.kill(request).await?;

        Ok(())
    }
}

#[tonic::async_trait]
impl LighthouseService for Arc<Lighthouse> {
    async fn quorum(
        &self,
        request: Request<LighthouseQuorumRequest>,
    ) -> Result<Response<LighthouseQuorumResponse>, Status> {
        let req = request.into_inner();
        let requester = req
            .requester
            .ok_or_else(|| return Status::invalid_argument("missing requester"))?;

        info!(
            "Received quorum request for replica {}",
            &requester.replica_id
        );

        let mut rx = {
            let mut state = self.state.lock().await;

            // implicit heartbeat
            state
                .heartbeats
                .insert(requester.replica_id.clone(), Instant::now());

            state.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester.clone(),
                },
            );
            let rx = state.channel.subscribe();

            // proactively run quorum tick
            self.clone()
                ._quorum_tick(&mut state)
                .map_err(|e| Status::from_error(e.into()))?;

            rx
        };

        let quorum = loop {
            let current_quorum = rx.recv().await.map_err(|e| Status::from_error(e.into()))?;

            if current_quorum
                .participants
                .iter()
                .any(|p| p.replica_id == requester.replica_id)
            {
                break current_quorum;
            }

            // Only continue the loop if the replica is not in the quorum
            let mut state = self.state.lock().await;
            state.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester.clone(),
                },
            );
            info!("Replica {} not in quorum, retrying", &requester.replica_id);
        };

        let reply = LighthouseQuorumResponse {
            quorum: Some(quorum),
        };

        Ok(Response::new(reply))
    }

    async fn heartbeat(
        &self,
        request: Request<LighthouseHeartbeatRequest>,
    ) -> Result<Response<LighthouseHeartbeatResponse>, Status> {
        let replica_id = request.into_inner().replica_id;

        {
            let mut state = self.state.lock().await;
            state.heartbeats.insert(replica_id, Instant::now());
        }

        let reply = LighthouseHeartbeatResponse {};
        Ok(Response::new(reply))
    }
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {}

#[derive(Template)]
#[template(path = "status.html")]
struct StatusTemplate {
    prev_quorum: Option<Quorum>,
    quorum_id: i64,
    quorum_status: String,
    num_participants: i64,
    max_step: i64,
    heartbeats: HashMap<String, Instant>,

    // visualization thresholds
    old_age_threshold: Instant,
}

// Make our own error that wraps `anyhow::Error`.
struct AppError(anyhow::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Sub;

    use tonic::transport::Channel;

    use super::*;
    use crate::net::connect;
    use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;

    async fn lighthouse_client_new(addr: String) -> Result<LighthouseServiceClient<Channel>> {
        let conn = connect(addr, Duration::from_secs(10)).await?;
        Ok(LighthouseServiceClient::new(conn))
    }

    #[tokio::test]
    async fn test_quorum_join_timeout() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: None,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
        };

        let now = Instant::now();

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("New quorum not ready, only have 0 participants, need min_replicas 1 [0/0 participants healthy]"), "{}", reason);

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        // all healthy workers participating
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        // add healthy worker but not participating
        state.heartbeats.insert("c".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("join timeout"), "{}", reason);

        // increase elapsed time to pass join timeout
        state.participants.get_mut("a").unwrap().joined = now.sub(Duration::from_hours(10));
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_heartbeats() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 0,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: None,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
        };

        let now = Instant::now();

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        assert!(
            reason.contains("[1/1 participants healthy][1 heartbeating]"),
            "{}",
            reason
        );

        // expired heartbeat
        state
            .heartbeats
            .insert("a".to_string(), now.sub(Duration::from_secs(10)));

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(
            reason.contains("[0/1 participants healthy][0 heartbeating]"),
            "{}",
            reason
        );

        // 1 healthy, 1 expired
        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        let participants = quorum_met.unwrap();
        assert!(participants.len() == 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_fast_prev_quorum() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: None,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
        };

        let now = Instant::now();

        assert!(!quorum_compute(now, &state, &opt).0.is_some());

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        // Not proceeding since one worker is alive but not participating
        state.heartbeats.insert("b".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("need at least half"), "{}", reason);

        state.prev_quorum = Some(Quorum {
            quorum_id: 1,
            participants: vec![QuorumMember {
                replica_id: "a".to_string(),
                address: "".to_string(),
                store_address: "".to_string(),
                step: 1,
                world_size: 1,
                shrink_only: false,
                data: String::new(),
                commit_failures: 0,
            }],
            created: Some(SystemTime::now().into()),
        });

        assert!(quorum_compute(now, &state, &opt).0.is_some());

        // test expanding quorum w/ fast quorum
        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        let participants = quorum_met.unwrap();
        assert!(participants.len() == 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_shrink_only() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: None,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
        };

        let now = Instant::now();

        state.prev_quorum = Some(Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            ],
            created: Some(SystemTime::now().into()),
        });

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: true,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        // insert particpant that was not in prev quorum
        state.participants.insert(
            "c".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "c".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: true,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("c".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        assert!(reason.contains("[shrink_only=true]",), "{}", reason);

        let quorum = quorum_met.unwrap();
        assert!(quorum.len() == 1);
        assert!(quorum[0].replica_id == "a");

        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_e2e() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: None,
        };
        let lighthouse = Lighthouse::new(opt).await?;

        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new(lighthouse.address()).await.unwrap();

        {
            let request = tonic::Request::new(LighthouseHeartbeatRequest {
                replica_id: "foo".to_string(),
            });

            let _response = client.heartbeat(request).await.unwrap();
        }

        {
            let request = tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(QuorumMember {
                    replica_id: "foo".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 10,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                }),
            });

            let response = client.quorum(request).await.unwrap();
            let quorum = response.into_inner().quorum.unwrap();
            assert_eq!(quorum.participants.len(), 1);
        }

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_split_brain() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: None,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
        };

        let now = Instant::now();

        assert!(!quorum_compute(now, &state, &opt).0.is_some());

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        // Not proceeding since one worker is alive but not participating
        state.heartbeats.insert("b".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("New quorum not ready, only have 1 participants, need at least half of 2 healthy workers [1/1 participants healthy][2 heartbeating]"), "{}", reason);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_changed() {
        let a = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];
        let b = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "changed".to_string(),
            store_address: "changed".to_string(),
            step: 1000,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];

        // replica_id is the same
        assert!(!quorum_changed(&a, &b));

        let c = vec![QuorumMember {
            replica_id: "2".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];
        // replica_id changed
        assert!(quorum_changed(&a, &c));
    }

    #[tokio::test]
    async fn test_lighthouse_join_during_shrink() -> Result<()> {
        fn create_member(id: &str, addr_num: &str, step: i64, shrink_only: bool) -> QuorumMember {
            QuorumMember {
                replica_id: id.to_string(),
                address: format!("addr{}", addr_num),
                store_address: format!("store{}", addr_num),
                step,
                world_size: 1,
                shrink_only,
                data: String::new(),
                commit_failures: 0,
            }
        }

        fn create_request(member: &QuorumMember) -> tonic::Request<LighthouseQuorumRequest> {
            tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(member.clone()),
            })
        }

        let opt = LighthouseOpt {
            min_replicas: 2,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1000,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: None,
        };

        // Start the lighthouse service
        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        // Create client to interact with lighthouse
        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        // 1. First quorum
        let mut first_request = create_request(&create_member("replica0", "1", 1, false));
        let mut second_request = create_request(&create_member("replica1", "2", 1, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let first_response = client.quorum(second_request).await?;
        let first_quorum = first_response.into_inner().quorum.unwrap();
        assert_eq!(first_quorum.participants.len(), 2);
        assert_eq!(first_quorum.participants[0].replica_id, "replica0");
        assert_eq!(first_quorum.participants[1].replica_id, "replica1");
        assert_eq!(first_quorum.participants[1].step, 1);

        // 2. Quorum without joiner
        let joining_request = create_request(&create_member("joiner", "3", 1, false));
        let joining_task = tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(joining_request).await }
        });

        // Try to shrink only
        first_request = create_request(&create_member("replica0", "1", 2, true));
        second_request = create_request(&create_member("replica1", "2", 2, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let second_response = client.quorum(second_request).await?;
        let second_quorum = second_response.into_inner().quorum.unwrap();
        assert!(
            second_quorum
                .participants
                .iter()
                .all(|p| p.replica_id != "joiner")
        );
        assert_eq!(second_quorum.participants.len(), 2);
        assert_eq!(second_quorum.participants[0].replica_id, "replica0");
        assert_eq!(second_quorum.participants[1].replica_id, "replica1");
        assert_eq!(second_quorum.participants[1].step, 2);

        // 3. Quorum with joiner
        first_request = create_request(&create_member("replica0", "1", 3, false));
        second_request = create_request(&create_member("replica1", "2", 3, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let third_response = client.quorum(second_request).await?;
        let third_quorum = third_response.into_inner().quorum.unwrap();
        assert!(
            third_quorum
                .participants
                .iter()
                .any(|p| p.replica_id == "joiner")
        );
        assert_eq!(third_quorum.participants.len(), 3);
        assert_eq!(third_quorum.participants[2].step, 3);

        let join_result = joining_task.await?;
        let join_quorum = join_result.unwrap().into_inner().quorum.unwrap();
        assert!(
            join_quorum
                .participants
                .iter()
                .any(|p| p.replica_id == "joiner")
        );
        assert_eq!(join_quorum.participants.len(), 3);
        assert_eq!(join_quorum.participants[2].step, 3);

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_commit_failures() -> Result<()> {
        fn create_member(id: &str, commit_failures: i64) -> QuorumMember {
            QuorumMember {
                replica_id: id.to_string(),
                address: format!("addr{}", id),
                store_address: format!("store{}", id),
                step: 10,
                world_size: 1,
                shrink_only: false,
                data: String::new(),
                commit_failures,
            }
        }

        fn create_request(member: &QuorumMember) -> tonic::Request<LighthouseQuorumRequest> {
            tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(member.clone()),
            })
        }

        let opt = LighthouseOpt {
            min_replicas: 2,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1000,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: None,
        };

        // Start the lighthouse service
        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        // Create client to interact with lighthouse
        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        // First two quorums should be stable
        for _i in 0..2 {
            let first_request = create_request(&create_member("replica0", 0));
            let second_request = create_request(&create_member("replica1", 0));

            tokio::spawn({
                let mut client = client.clone();
                async move { client.quorum(first_request).await }
            });
            let first_response = client.quorum(second_request).await?;
            let first_quorum = first_response.into_inner().quorum.unwrap();
            assert_eq!(first_quorum.quorum_id, 1);
            assert_eq!(first_quorum.participants.len(), 2);
            assert_eq!(first_quorum.participants[0].commit_failures, 0);
            assert_eq!(first_quorum.participants[1].commit_failures, 0);
        }

        // commit_failures should increment quorum_id
        let first_request = create_request(&create_member("replica0", 0));
        let second_request = create_request(&create_member("replica1", 2));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let first_response = client.quorum(second_request).await?;
        let first_quorum = first_response.into_inner().quorum.unwrap();
        assert_eq!(first_quorum.quorum_id, 2);
        assert_eq!(first_quorum.participants.len(), 2);
        assert_eq!(first_quorum.participants[0].commit_failures, 0);
        assert_eq!(first_quorum.participants[1].commit_failures, 2);

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_expected_replicas_truncate() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 0,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: Some(vec![1, 2, 4]),
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
        };

        let now = Instant::now();

        // Add 3 participants: "a", "b", "c"
        for id in ["a", "b", "c"] {
            state.participants.insert(
                id.to_string(),
                QuorumMemberDetails {
                    joined: now,
                    member: QuorumMember {
                        replica_id: id.to_string(),
                        address: format!("addr_{}", id),
                        store_address: format!("store_{}", id),
                        step: 1,
                        world_size: 3,
                        shrink_only: false,
                        data: String::new(),
                        commit_failures: 0,
                    },
                },
            );
            state.heartbeats.insert(id.to_string(), now);
        }

        // With 3 participants and expected_replicas=[1,2,4], should get 2 participants
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        let participants = quorum_met.unwrap();
        assert_eq!(
            participants.len(),
            2,
            "Should have 2 participants (from expected_replicas)"
        );

        // Verify that we keep the participants with smaller IDs ("a" and "b")
        assert_eq!(
            participants[0].replica_id, "a",
            "First participant should be 'a' (smallest ID)"
        );
        assert_eq!(
            participants[1].replica_id, "b",
            "Second participant should be 'b' (second smallest ID)"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_expected_replicas_preserves_prev_quorum() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 0,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: Some(vec![1, 2, 4]),
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 1,
            heartbeats: HashMap::new(),
        };

        let now = Instant::now();

        // Setup prev_quorum with participants "b" and "d" (not sorted, to test that we preserve them)
        state.prev_quorum = Some(Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "b".to_string(),
                    address: "addr_b".to_string(),
                    store_address: "store_b".to_string(),
                    step: 1,
                    world_size: 2,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "d".to_string(),
                    address: "addr_d".to_string(),
                    store_address: "store_d".to_string(),
                    step: 1,
                    world_size: 2,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            ],
            created: Some(SystemTime::now().into()),
        });

        // Add 4 participants: "a", "b", "c", "d" (all healthy)
        // "b" and "d" are from prev_quorum, "a" and "c" are new
        for id in ["a", "b", "c", "d"] {
            state.participants.insert(
                id.to_string(),
                QuorumMemberDetails {
                    joined: now,
                    member: QuorumMember {
                        replica_id: id.to_string(),
                        address: format!("addr_{}", id),
                        store_address: format!("store_{}", id),
                        step: 1,
                        world_size: 4,
                        shrink_only: false,
                        data: String::new(),
                        commit_failures: 0,
                    },
                },
            );
            state.heartbeats.insert(id.to_string(), now);
        }

        // With 4 participants and expected_replicas=[1,2,4], should get 4 participants
        // But this is fast_quorum (all prev_quorum members are healthy)
        // If we had 5 participants with expected_replicas=[1,2,4], we should keep 4
        // Let's add a 5th participant "e"
        state.participants.insert(
            "e".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "e".to_string(),
                    address: "addr_e".to_string(),
                    store_address: "store_e".to_string(),
                    step: 1,
                    world_size: 5,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("e".to_string(), now);

        // With 5 participants and expected_replicas=[1,2,4], should get 4 participants
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        let participants = quorum_met.unwrap();
        assert_eq!(
            participants.len(),
            4,
            "Should have 4 participants (from expected_replicas)"
        );

        // Verify the sorted order: should be ["a", "b", "c", "d"], but "e" should be excluded
        assert_eq!(
            participants[0].replica_id, "a",
            "First participant should be 'a' (sorted order)"
        );
        assert_eq!(
            participants[1].replica_id, "b",
            "Second participant should be 'b' (sorted order)"
        );
        assert_eq!(
            participants[2].replica_id, "c",
            "Third participant should be 'c' (sorted order)"
        );
        assert_eq!(
            participants[3].replica_id, "d",
            "Fourth participant should be 'd' (sorted order)"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_expected_replicas_shrink() -> Result<()> {
        // This test simulates the user's scenario:
        // - Start with 4 replicas (world_size=4)
        // - One replica fails, leaving 3 healthy
        // - With expected_replicas=[1,2,4], should shrink to 2 (not stay at 3)
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 0,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
            expected_replicas: Some(vec![1, 2, 4]),
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 1,
            heartbeats: HashMap::new(),
        };

        let now = Instant::now();

        // Setup prev_quorum with 4 participants (simulating initial quorum)
        state.prev_quorum = Some(Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "replica0".to_string(),
                    address: "addr0".to_string(),
                    store_address: "store0".to_string(),
                    step: 100,
                    world_size: 4,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica1".to_string(),
                    address: "addr1".to_string(),
                    store_address: "store1".to_string(),
                    step: 100,
                    world_size: 4,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica2".to_string(),
                    address: "addr2".to_string(),
                    store_address: "store2".to_string(),
                    step: 100,
                    world_size: 4,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "replica3".to_string(),
                    address: "addr3".to_string(),
                    store_address: "store3".to_string(),
                    step: 100,
                    world_size: 4,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            ],
            created: Some(SystemTime::now().into()),
        });

        // Now only 3 replicas are healthy (replica3 failed)
        for id in ["replica0", "replica1", "replica2"] {
            state.participants.insert(
                id.to_string(),
                QuorumMemberDetails {
                    joined: now,
                    member: QuorumMember {
                        replica_id: id.to_string(),
                        address: format!("addr_{}", &id[7..]), // extract number from "replicaN"
                        store_address: format!("store_{}", &id[7..]),
                        step: 100,
                        world_size: 3,
                        shrink_only: false,
                        data: String::new(),
                        commit_failures: 0,
                    },
                },
            );
            state.heartbeats.insert(id.to_string(), now);
        }

        // With 3 healthy participants and expected_replicas=[1,2,4],
        // should truncate to 2 participants (not keep all 3)
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        let participants = quorum_met.unwrap();
        assert_eq!(
            participants.len(),
            2,
            "Should have 2 participants (from expected_replicas), not 3. Reason: {}",
            reason
        );

        // Should keep replica0 and replica1 (smallest IDs)
        assert_eq!(participants[0].replica_id, "replica0");
        assert_eq!(participants[1].replica_id, "replica1");

        // Verify the reason includes the truncation info
        assert!(
            reason.contains("2/3 participants selected"),
            "Reason should show truncation: {}",
            reason
        );

        Ok(())
    }
}
