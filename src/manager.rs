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

use anyhow::Result;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::sleep;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Channel;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use crate::net::connect;
use crate::timeout::try_parse_grpc_timeout;
use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;
use crate::torchftpb::manager_service_client::ManagerServiceClient;
use crate::torchftpb::{
    manager_service_server::{ManagerService, ManagerServiceServer},
    CheckpointAddressRequest, CheckpointAddressResponse, KillRequest, KillResponse,
    LighthouseHeartbeatRequest, LighthouseQuorumRequest, ManagerQuorumRequest,
    ManagerQuorumResponse, Quorum, QuorumMember, ShouldCommitRequest, ShouldCommitResponse,
};

#[cfg(not(test))]
use log::{info, warn};

#[cfg(test)]
use std::{println as info, println as warn};

struct ManagerState {
    checkpoint_servers: HashMap<i64, String>,
    channel: broadcast::Sender<Quorum>,
    participants: HashSet<i64>,

    should_commit_channel: broadcast::Sender<bool>,
    should_commit_failures: HashSet<i64>,
    should_commit_count: HashSet<i64>,
}

pub struct Manager {
    replica_id: String,
    hostname: String,
    store_address: String,
    world_size: u64,
    state: Mutex<ManagerState>,
    listener: Mutex<Option<tokio::net::TcpListener>>,
    local_addr: SocketAddr,
    heartbeat_interval: Duration,
    lighthouse_client: LighthouseServiceClient<Channel>,
}

pub async fn manager_client_new(
    addr: String,
    connect_timeout: Duration,
) -> Result<ManagerServiceClient<Channel>> {
    info!("ManagerClient: establishing connection to {}", &addr);
    let conn = connect(addr, connect_timeout).await?;
    Ok(ManagerServiceClient::new(conn))
}

pub async fn lighthouse_client_new(
    addr: String,
    connect_timeout: Duration,
) -> Result<LighthouseServiceClient<Channel>> {
    info!("LighthouseClient: establishing connection to {}", &addr);
    let conn = connect(addr, connect_timeout).await?;
    Ok(LighthouseServiceClient::new(conn))
}

impl Manager {
    pub async fn new(
        replica_id: String,
        lighthouse_addr: String,
        hostname: String,
        bind: String,
        store_addr: String,
        world_size: u64,
        heartbeat_interval: Duration,
        connect_timeout: Duration,
    ) -> Result<Arc<Self>> {
        let listener = tokio::net::TcpListener::bind(&bind).await?;
        let local_addr = listener.local_addr()?;

        let (should_commit_tx, _) = broadcast::channel(16);
        let (tx, _) = broadcast::channel(16);

        let client = lighthouse_client_new(lighthouse_addr.clone(), connect_timeout).await?;

        Ok(Arc::new(Self {
            replica_id: replica_id,
            lighthouse_client: client,
            hostname: hostname,
            store_address: store_addr,
            world_size: world_size,
            heartbeat_interval: heartbeat_interval,
            state: Mutex::new(ManagerState {
                checkpoint_servers: HashMap::new(),
                channel: tx,
                participants: HashSet::new(),

                should_commit_channel: should_commit_tx,
                should_commit_count: HashSet::new(),
                should_commit_failures: HashSet::new(),
            }),
            local_addr: local_addr,
            listener: Mutex::new(Some(listener)),
        }))
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut set = JoinSet::new();

        set.spawn(self.clone()._run_heartbeat());

        set.spawn(self.clone()._run_grpc());

        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    }

    pub fn address(&self) -> String {
        format!("http://{}:{}", self.hostname, self.local_addr.port())
    }

    async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        info!(
            "Manager {} listening on {}",
            self.replica_id,
            self.address()
        );

        let listener = self.listener.lock().await.take().unwrap();
        let incoming =
            TcpIncoming::from_listener(listener, true, None).map_err(|e| anyhow::anyhow!(e))?;

        Server::builder()
            .add_service(ManagerServiceServer::new(self))
            .serve_with_incoming(incoming)
            .await
            .map_err(|e| e.into())
    }

    async fn _run_heartbeat(self: Arc<Self>) -> Result<()> {
        let mut client = self.lighthouse_client.clone();
        loop {
            let request = tonic::Request::new(LighthouseHeartbeatRequest {
                replica_id: self.replica_id.clone(),
            });

            let _response = client.heartbeat(request).await;

            sleep(self.heartbeat_interval).await;
        }
    }

    async fn _run_quorum(
        &self,
        state: &mut ManagerState,
        requester: QuorumMember,
        timeout: Duration,
    ) -> Result<(), Status> {
        if (state.participants.len() as u64) < self.world_size {
            return Ok(());
        }

        state.participants.clear();
        info!("all workers joined -- starting quorum");

        // TODO: don't hold the lock during quorum

        let mut client = self.lighthouse_client.clone();

        let mut lighthouse_request = tonic::Request::new(LighthouseQuorumRequest {
            requester: Some(requester),
        });
        lighthouse_request.set_timeout(timeout);

        let response = tokio::time::timeout(timeout, client.quorum(lighthouse_request))
            .await
            .unwrap_or_else(|e| {
                Err(Status::cancelled(format!(
                    "lighthouse quorum timed out: {}",
                    e.to_string()
                )))
            })?;
        let resp = response.into_inner();

        info!("got lighthouse quorum {:?}", resp);

        state
            .channel
            .send(
                resp.quorum
                    .ok_or_else(|| Status::internal("missing quorum"))?,
            )
            .map_err(|e| Status::from_error(e.into()))?;

        Ok(())
    }
}

#[tonic::async_trait]
impl ManagerService for Arc<Manager> {
    async fn quorum(
        &self,
        request: Request<ManagerQuorumRequest>,
    ) -> Result<Response<ManagerQuorumResponse>, Status> {
        let req = request.get_ref();
        let rank = req.rank;

        info!("got quorum request for rank {}", rank);

        let timeout = try_parse_grpc_timeout(&request.metadata())
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "invalid timeout {}",
                    e.to_str().unwrap_or("invalid")
                ))
            })?
            .ok_or_else(|| Status::invalid_argument("missing timeout"))?;

        let mut rx = {
            let mut state = self.state.lock().await;

            // save checkpoint server info for healing process
            // TODO: make separate call to set?
            state
                .checkpoint_servers
                .insert(req.rank, req.checkpoint_server_addr.clone());

            // TODO check step
            state.participants.insert(rank);
            let rx = state.channel.subscribe();

            self._run_quorum(
                &mut state,
                QuorumMember {
                    replica_id: self.replica_id.clone(),
                    address: self.address(),
                    store_address: self.store_address.clone(),
                    step: req.step,
                    world_size: self.world_size,
                    shrink_only: req.shrink_only,
                },
                timeout,
            )
            .await?;

            rx
        };

        let quorum = rx
            .recv()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut participants = quorum.participants.clone();
        participants.sort_by(|a, b| a.replica_id.cmp(&b.replica_id));

        let replica_rank = participants.iter().enumerate().find_map(|(i, p)| {
            if p.replica_id == self.replica_id {
                Some(i)
            } else {
                None
            }
        });
        if replica_rank.is_none() {
            return Err(Status::not_found(format!(
                "replica {} not participating in returned quorum",
                self.replica_id
            )));
        }

        let max_step = participants.iter().map(|p| p.step).max().unwrap();
        let max_participants: Vec<&QuorumMember> =
            participants.iter().filter(|p| p.step == max_step).collect();

        let primary = max_participants[rank as usize % max_participants.len()];

        let mut max_rank = None;
        for (i, p) in max_participants.iter().enumerate() {
            if p.replica_id == self.replica_id {
                max_rank = Some(i as i64);
                break;
            }
        }

        // Decide whether we should be healing:
        // 1. if we're not at the max step
        // 2. if everyone is at the first step and we're not the primary
        let heal = max_step != req.step || max_step == 0 && primary.replica_id != self.replica_id;
        if heal {
            info!(
                "healing is required step={}, max_step={}",
                req.step, max_step
            );
        }

        let reply = ManagerQuorumResponse {
            quorum_id: quorum.quorum_id,
            // address is used for looking up the checkpoint server address.
            address: primary.address.clone(),
            store_address: primary.store_address.clone(),
            max_step: max_step,
            max_rank: max_rank,
            max_world_size: max_participants.len() as i64,
            replica_rank: replica_rank.unwrap() as i64,
            replica_world_size: participants.len() as i64,
            heal: heal,
        };

        info!("returning quorum for rank {}", rank);

        Ok(Response::new(reply))
    }

    async fn checkpoint_address(
        &self,
        request: Request<CheckpointAddressRequest>,
    ) -> Result<Response<CheckpointAddressResponse>, Status> {
        let state = self.state.lock().await;

        let req = request.into_inner();

        let address = state
            .checkpoint_servers
            .get(&req.rank)
            .ok_or_else(|| Status::invalid_argument("rank not found"))?;

        let reply = CheckpointAddressResponse {
            checkpoint_server_address: address.clone(),
        };
        Ok(Response::new(reply))
    }

    async fn should_commit(
        &self,
        request: Request<ShouldCommitRequest>,
    ) -> Result<Response<ShouldCommitResponse>, Status> {
        let req = request.into_inner();
        let rank = req.rank;

        info!(
            "should_commit request from {} should_commit={}",
            rank, req.should_commit
        );

        // TODO: check step count

        let mut rx = {
            let mut state = self.state.lock().await;

            if !req.should_commit {
                state.should_commit_failures.insert(rank);
            }
            state.should_commit_count.insert(rank);

            let rx = state.should_commit_channel.subscribe();

            if state.should_commit_count.len() == self.world_size as usize {
                let decision = state.should_commit_failures.len() == 0;
                info!("should_commit completed should_commit={}", decision);

                state
                    .should_commit_channel
                    .send(decision)
                    .map_err(|e| Status::from_error(e.into()))?;

                // reset state
                state.should_commit_count.clear();
                state.should_commit_failures.clear();
                let (should_commit_tx, _) = broadcast::channel(16);
                state.should_commit_channel = should_commit_tx;
            }

            rx
        };

        let should_commit = rx
            .recv()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let reply = ShouldCommitResponse {
            should_commit: should_commit,
        };
        Ok(Response::new(reply))
    }

    async fn kill(&self, request: Request<KillRequest>) -> Result<Response<KillResponse>, Status> {
        let req = request.into_inner();

        warn!("got kill request: {}", req.msg);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighthouse::{Lighthouse, LighthouseOpt};

    async fn should_commit(rank: i64, should_commit: bool) -> Result<ShouldCommitResponse> {
        let mut client = manager_client_new(
            "http://localhost:29531".to_string(),
            Duration::from_secs(10),
        )
        .await?;

        let request = tonic::Request::new(ShouldCommitRequest {
            rank: rank,
            step: 1,
            should_commit: should_commit,
        });
        let resp = client.should_commit(request).await?;

        Ok(resp.into_inner())
    }

    #[tokio::test]
    async fn test_should_commit() -> Result<()> {
        let lighthouse = Lighthouse::new(LighthouseOpt {
            bind: "[::]:0".to_string(),
            join_timeout_ms: 100,
            min_replicas: 1,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
        })
        .await?;
        let lighthouse_fut = tokio::spawn(lighthouse.clone().run());

        let manager = Manager::new(
            "rep_id".to_string(),
            lighthouse.address(),
            "addr".to_string(),
            "[::]:29531".to_string(),
            "store_addr".to_string(),
            2,                          // world size
            Duration::from_millis(100), // heartbeat interval
            Duration::from_secs(10),    // connect timeout
        )
        .await?;
        let manager_fut = tokio::spawn(manager._run_grpc());

        let fut_a = tokio::spawn(should_commit(0, true));
        let fut_b = tokio::spawn(should_commit(1, true));
        let resp_a = fut_a.await??;
        let resp_b = fut_b.await??;

        assert!(resp_a.should_commit);
        assert!(resp_b.should_commit);

        let fut_a = tokio::spawn(should_commit(0, true));
        let fut_b = tokio::spawn(should_commit(1, false));
        let resp_a = fut_a.await??;
        let resp_b = fut_b.await??;

        assert!(!resp_a.should_commit);
        assert!(!resp_b.should_commit);

        manager_fut.abort();
        lighthouse_fut.abort();

        Ok(())
    }

    #[tokio::test]
    async fn test_get_quorum() -> Result<()> {
        let lighthouse = Lighthouse::new(LighthouseOpt {
            bind: "[::]:0".to_string(),
            join_timeout_ms: 100,
            min_replicas: 1,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
        })
        .await?;
        let lighthouse_fut = tokio::spawn(lighthouse.clone().run());

        let manager = Manager::new(
            "rep_id".to_string(),
            lighthouse.address(),
            "localhost".to_string(),
            "[::]:0".to_string(),
            "store_addr".to_string(),
            1,                          // world size
            Duration::from_millis(100), // heartbeat interval
            Duration::from_secs(10),    // connect timeout
        )
        .await?;
        let manager_fut = tokio::spawn(manager.clone().run());

        let mut client = manager_client_new(manager.address(), Duration::from_secs(10)).await?;

        let mut request = tonic::Request::new(ManagerQuorumRequest {
            rank: 0,
            step: 123,
            checkpoint_server_addr: "addr".to_string(),
            shrink_only: false,
        });
        request.set_timeout(Duration::from_secs(10));
        let resp = client.quorum(request).await?.into_inner();

        manager_fut.abort();
        lighthouse_fut.abort();

        assert_eq!(resp.quorum_id, 1);
        assert_eq!(resp.address, manager.address());
        assert_eq!(resp.store_address, "store_addr".to_string());
        assert_eq!(resp.max_step, 123);
        assert_eq!(resp.max_rank, Some(0));
        assert_eq!(resp.max_world_size, 1);
        assert_eq!(resp.replica_rank, 0);
        assert_eq!(resp.replica_world_size, 1);
        assert_eq!(resp.heal, false);

        Ok(())
    }

    #[tokio::test]
    async fn test_get_quorum_heal_first_step() -> Result<()> {
        let lighthouse = Lighthouse::new(LighthouseOpt {
            bind: "[::]:0".to_string(),
            join_timeout_ms: 100,
            min_replicas: 2,
            quorum_tick_ms: 100,
            heartbeat_timeout_ms: 5000,
        })
        .await?;
        let lighthouse_fut = tokio::spawn(lighthouse.clone().run());

        let mut manager_futs: Vec<tokio::task::JoinHandle<Result<ManagerQuorumResponse>>> =
            Vec::new();

        for replica_id in 0..2 {
            let lighthouse_addr = lighthouse.address();
            manager_futs.push(tokio::spawn(async move {
                let manager = Manager::new(
                    format!("rep_{}", replica_id),
                    lighthouse_addr,
                    "localhost".to_string(),
                    "[::]:0".to_string(),
                    "store_addr".to_string(),
                    1,                          // world size
                    Duration::from_millis(100), // heartbeat interval
                    Duration::from_secs(10),    // connect timeout
                )
                .await?;
                let manager_fut = tokio::spawn(manager.clone().run());

                let mut client =
                    manager_client_new(manager.address(), Duration::from_secs(10)).await?;

                let mut request = tonic::Request::new(ManagerQuorumRequest {
                    rank: 0,
                    step: 0,
                    checkpoint_server_addr: "addr".to_string(),
                    shrink_only: false,
                });
                request.set_timeout(Duration::from_secs(10));

                let result = client.quorum(request).await?.into_inner();

                manager_fut.abort();

                Ok(result)
            }));
        }

        let resp_a = manager_futs.swap_remove(0).await??;
        let resp_b = manager_futs.swap_remove(0).await??;

        lighthouse_fut.abort();

        assert_eq!(resp_a.quorum_id, 1);
        assert_eq!(resp_a.max_step, 0);
        assert_eq!(resp_a.replica_rank, 0);
        assert_eq!(resp_a.replica_world_size, 2);
        assert_eq!(resp_a.heal, false);

        assert_eq!(resp_b.quorum_id, 1);
        assert_eq!(resp_b.max_step, 0);
        assert_eq!(resp_b.replica_rank, 1);
        assert_eq!(resp_b.replica_world_size, 2);
        assert_eq!(resp_b.heal, true);

        Ok(())
    }
}
