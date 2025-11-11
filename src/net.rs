// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use std::time::Duration;

use anyhow::Result;
use tonic::transport::Channel;
use tonic::transport::Endpoint;

use crate::retry::ExponentialBackoff;
use crate::retry::retry_backoff;

pub async fn connect_once(addr: String, connect_timeout: Duration) -> Result<Channel> {
    let conn = Endpoint::new(addr)?
        .connect_timeout(connect_timeout)
        // Enable HTTP2 keep alives
        .http2_keep_alive_interval(Duration::from_secs(60))
        // Time taken for server to respond. 20s is default for GRPC.
        .keep_alive_timeout(Duration::from_secs(20))
        // Enable alive for idle connections.
        .keep_alive_while_idle(true)
        .connect()
        .await?;
    Ok(conn)
}

pub async fn connect(addr: String, connect_timeout: Duration) -> Result<Channel> {
    retry_backoff(
        ExponentialBackoff {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            timeout: connect_timeout,
            factor: 1.5,
            max_jitter: Duration::from_millis(100),
        },
        || Box::pin(connect_once(addr.clone(), connect_timeout)),
    )
    .await
}
