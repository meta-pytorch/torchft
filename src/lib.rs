// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod lighthouse;
pub mod manager;

use core::time::Duration;
use std::env;
use std::sync::Arc;

use anyhow::Result;
use pyo3::exceptions::PyRuntimeError;
use structopt::StructOpt;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tonic::transport::Channel;

pub mod torchftpb {
    tonic::include_proto!("torchft");
}

use crate::torchftpb::manager_service_client::ManagerServiceClient;
use crate::torchftpb::{CheckpointAddressRequest, ManagerQuorumRequest, ShouldCommitRequest};
use pyo3::prelude::*;

#[pyclass]
struct Manager {
    handle: JoinHandle<Result<()>>,
    manager: Arc<manager::Manager>,
    _runtime: Runtime,
}

#[pymethods]
impl Manager {
    #[new]
    fn new(
        py: Python<'_>,
        replica_id: String,
        lighthouse_addr: String,
        address: String,
        bind: String,
        store_addr: String,
        world_size: u64,
    ) -> Self {
        py.allow_threads(move || {
            let runtime = Runtime::new().unwrap();
            let manager = runtime
                .block_on(manager::Manager::new(
                    replica_id,
                    lighthouse_addr,
                    address,
                    bind,
                    store_addr,
                    world_size,
                ))
                .unwrap();
            let handle = runtime.spawn(manager.clone().run());
            Self {
                handle: handle,
                manager: manager,
                _runtime: runtime,
            }
        })
    }

    fn address(&self) -> PyResult<String> {
        Ok(self.manager.address().to_string())
    }

    fn shutdown(&self, py: Python<'_>) {
        py.allow_threads(move || {
            self.handle.abort();
        })
    }
}

#[pyclass]
struct ManagerClient {
    runtime: Runtime,
    client: ManagerServiceClient<Channel>,
    timeout: Duration,
}

#[pymethods]
impl ManagerClient {
    #[new]
    fn new(py: Python<'_>, addr: String, timeout: Duration) -> PyResult<Self> {
        py.allow_threads(move || {
            let runtime = Runtime::new().unwrap();
            let client = runtime
                .block_on(manager::manager_client_new(addr, timeout))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(Self {
                runtime: runtime,
                client: client,
                timeout: timeout,
            })
        })
    }

    fn quorum(
        &mut self,
        py: Python<'_>,
        rank: i64,
        step: i64,
        checkpoint_server_addr: String,
    ) -> PyResult<(i64, i64, i64, String, String, i64, Option<i64>, i64, bool)> {
        py.allow_threads(move || {
            let mut request = tonic::Request::new(ManagerQuorumRequest {
                rank: rank,
                step: step,
                checkpoint_server_addr: checkpoint_server_addr,
            });
            // This notifies the server about the timeout but doesn't affect the
            // endpoint timeout which we set on client creation.
            request.set_timeout(self.timeout);

            let response = self
                .runtime
                .block_on(self.client.quorum(request))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let resp = response.into_inner();
            Ok((
                resp.quorum_id,
                resp.replica_rank,
                resp.replica_world_size,
                resp.address,
                resp.store_address,
                resp.max_step,
                resp.max_rank,
                resp.max_world_size,
                resp.heal,
            ))
        })
    }

    fn checkpoint_address(&mut self, py: Python<'_>, rank: i64) -> PyResult<String> {
        py.allow_threads(move || {
            let mut request = tonic::Request::new(CheckpointAddressRequest { rank: rank });
            // This notifies the server about the timeout but doesn't affect the
            // endpoint timeout which we set on client creation.
            request.set_timeout(self.timeout);

            let response = self
                .runtime
                .block_on(self.client.checkpoint_address(request))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let resp = response.into_inner();
            Ok(resp.checkpoint_server_address)
        })
    }

    fn should_commit(
        &mut self,
        py: Python<'_>,
        rank: i64,
        step: i64,
        should_commit: bool,
    ) -> PyResult<bool> {
        py.allow_threads(move || {
            let mut request = tonic::Request::new(ShouldCommitRequest {
                rank: rank,
                step: step,
                should_commit: should_commit,
            });
            // This notifies the server about the timeout but doesn't affect the
            // endpoint timeout which we set on client creation.
            request.set_timeout(self.timeout);

            let response = self
                .runtime
                .block_on(self.client.should_commit(request))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let resp = response.into_inner();
            Ok(resp.should_commit)
        })
    }
}

fn reset_python_signals(py: Python<'_>) -> PyResult<()> {
    // clear python signal handlers
    // signal.signal(signal.SIGINT, signal.SIG_DFL)
    let signal = py.import_bound("signal")?;
    let set_signal = signal.getattr("signal")?;
    let args = (signal.getattr("SIGINT")?, signal.getattr("SIG_DFL")?);
    set_signal.call1(args)?;

    Ok(())
}

#[pyfunction]
fn lighthouse_main(py: Python<'_>) {
    reset_python_signals(py).unwrap();

    let mut args = env::args();
    args.next(); // discard binary arg
    let opt = lighthouse::LighthouseOpt::from_iter(args);
    let rt = Runtime::new().unwrap();
    rt.block_on(lighthouse_main_async(opt)).unwrap();
}

async fn lighthouse_main_async(opt: lighthouse::LighthouseOpt) -> Result<()> {
    let lighthouse = lighthouse::Lighthouse::new(opt).await?;

    lighthouse.run().await?;

    Ok(())
}

#[pyclass]
struct Lighthouse {
    lighthouse: Arc<lighthouse::Lighthouse>,
    handle: JoinHandle<Result<()>>,
    _runtime: Runtime,
}

#[pymethods]
impl Lighthouse {
    #[new]
    fn new(py: Python<'_>, bind: String, min_replicas: u64) -> PyResult<Self> {
        py.allow_threads(move || {
            let rt = Runtime::new().unwrap();

            let lighthouse = rt
                .block_on(lighthouse::Lighthouse::new(lighthouse::LighthouseOpt {
                    bind: bind,
                    min_replicas: min_replicas,
                    join_timeout_ms: 100,
                    quorum_tick_ms: 100,
                }))
                .unwrap();

            Ok(Self {
                handle: rt.spawn(lighthouse.clone().run()),
                lighthouse: lighthouse,
                _runtime: rt,
            })
        })
    }

    fn address(&self) -> PyResult<String> {
        Ok(self.lighthouse.address().to_string())
    }

    fn shutdown(&self, py: Python<'_>) {
        py.allow_threads(move || {
            self.handle.abort();
        })
    }
}

#[pymodule]
fn torchft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // setup logging on import
    stderrlog::new()
        .verbosity(2)
        .show_module_names(true)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .init()
        .unwrap();

    m.add_class::<Manager>()?;
    m.add_class::<ManagerClient>()?;
    m.add_class::<Lighthouse>()?;
    m.add_function(wrap_pyfunction!(lighthouse_main, m)?)?;

    Ok(())
}
