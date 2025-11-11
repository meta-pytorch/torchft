// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;

pub struct ExponentialBackoff {
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub timeout: Duration,
    pub factor: f64,
    pub max_jitter: Duration,
}

pub async fn retry_backoff<F, R>(policy: ExponentialBackoff, f: F) -> Result<R>
where
    F: Fn() -> Pin<Box<dyn Future<Output = Result<R>> + Send>>,
    R: Send,
{
    assert!(policy.initial_backoff > Duration::from_millis(0));
    assert!(policy.factor > 1.0);
    let mut backoff = policy.initial_backoff;

    let deadline = Instant::now() + policy.timeout;

    loop {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                if Instant::now() > deadline {
                    return Err(e);
                }
                let jitter = policy.max_jitter.mul_f64(rand::random::<f64>());
                tokio::time::sleep(backoff + jitter).await;
                backoff = backoff.mul_f64(policy.factor);
                if backoff > policy.max_backoff {
                    backoff = policy.max_backoff;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use super::*;

    #[tokio::test]
    async fn test_retry_backoff() -> Result<()> {
        let count = Arc::new(Mutex::new(0));
        let result = retry_backoff(
            ExponentialBackoff {
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(100),
                timeout: Duration::from_secs(1000),
                factor: 2.0,
                max_jitter: Duration::from_millis(1),
            },
            || {
                let current_count = {
                    let mut count = count.lock().unwrap();
                    *count += 1;
                    *count
                };

                Box::pin(async move {
                    if current_count < 3 {
                        Err(anyhow::anyhow!("test"))
                    } else {
                        Ok(1234)
                    }
                })
            },
        )
        .await?;
        assert!(result == 1234);
        let count = *count.lock().unwrap();
        assert!(count == 3, "count: {}", count);
        Ok(())
    }

    #[tokio::test]
    async fn test_retry_backoff_timeout() -> Result<()> {
        let result: Result<()> = retry_backoff(
            ExponentialBackoff {
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(100),
                timeout: Duration::from_millis(1),
                factor: 2.0,
                max_jitter: Duration::from_millis(1),
            },
            || Box::pin(async { Err(anyhow::anyhow!("test")) }),
        )
        .await;

        assert!(result.is_err());
        Ok(())
    }
}
