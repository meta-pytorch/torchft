// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use tonic::transport::Channel;
use tonic::transport::Endpoint;

use crate::retry::ExponentialBackoff;
use crate::retry::retry_backoff;

/// Formats an error with its full source chain since the Display
/// implementation for many errors (e.g. tonic's "transport error") omits the
/// underlying cause.
fn format_error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(err) = source {
        out.push_str(&format!(": {}", err));
        source = err.source();
    }
    out
}

pub async fn connect_once(addr: String, connect_timeout: Duration) -> Result<Channel> {
    let conn = Endpoint::new(addr.clone())?
        .connect_timeout(connect_timeout)
        // Enable HTTP2 keep alives
        .http2_keep_alive_interval(Duration::from_secs(60)) // 1 minute
        // Time taken for server to respond. 20s is default for GRPC.
        .keep_alive_timeout(Duration::from_secs(20))
        // Enable alive for idle connections.
        .keep_alive_while_idle(true)
        .connect()
        .await
        .map_err(|e| anyhow::anyhow!("{}", format_error_chain(&e)))
        .with_context(|| format!("failed to connect to {}", addr))?;
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
    .with_context(|| {
        format!(
            "failed to connect to {} within timeout of {:?}; verify the address is correct and the server is running and reachable, or increase connect_timeout",
            addr, connect_timeout
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connect_error_message() {
        let addr = "http://127.0.0.1:1".to_string();
        let err = connect(addr.clone(), Duration::from_millis(100))
            .await
            .expect_err("connect should fail");
        // Use the alternate format to include the full error chain, matching
        // how the error is surfaced to Python in lib.rs.
        let msg = format!("{:#}", err);
        assert!(msg.contains(&addr), "missing addr in error: {}", msg);
        assert!(msg.contains("connect_timeout"), "missing hint: {}", msg);
        assert!(
            msg.contains("Connection refused"),
            "missing underlying cause in error: {}",
            msg
        );
    }
}
