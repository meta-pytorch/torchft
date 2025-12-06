// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use std::time::Duration;

use anyhow::Result;
use tonic::metadata::Ascii;
use tonic::metadata::MetadataMap;
use tonic::metadata::MetadataValue;

const GRPC_TIMEOUT_HEADER: &str = "grpc-timeout";
const SECONDS_IN_HOUR: u64 = 60 * 60;
const SECONDS_IN_MINUTE: u64 = 60;

/// Tries to parse the `grpc-timeout` header if it is present. If we fail to parse, returns
/// the value we attempted to parse.
///
/// Follows the [gRPC over HTTP2 spec](https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md).
///
/// From https://github.com/hyperium/tonic/blob/79a06cc8067818ec53bae76ab717063683bb0acb/tonic/src/transport/service/grpc_timeout.rs#L106
/// Copyright (c) 2020 Lucio Franco MIT License
/// https://github.com/hyperium/tonic/blob/master/LICENSE
pub fn try_parse_grpc_timeout(
    headers: &MetadataMap,
) -> Result<Option<Duration>, &MetadataValue<Ascii>> {
    let Some(val) = headers.get(GRPC_TIMEOUT_HEADER) else {
        return Ok(None);
    };

    let (timeout_value, timeout_unit) = val
        .to_str()
        .map_err(|_| val)
        .and_then(|s| if s.is_empty() { Err(val) } else { Ok(s) })?
        // `MetadataValue::to_str` only returns `Ok` if the header contains ASCII so this
        // `split_at` will never panic from trying to split in the middle of a character.
        // See https://docs.rs/http/0.2.4/http/header/struct.MetadataValue.html#method.to_str
        //
        // `len - 1` also wont panic since we just checked `s.is_empty`.
        .split_at(val.len() - 1);

    // gRPC spec specifies `TimeoutValue` will be at most 8 digits
    // Caping this at 8 digits also prevents integer overflow from ever occurring
    if timeout_value.len() > 8 {
        return Err(val);
    }

    let timeout_value: u64 = timeout_value.parse().map_err(|_| val)?;

    let duration = match timeout_unit {
        // Hours
        "H" => Duration::from_secs(timeout_value * SECONDS_IN_HOUR),
        // Minutes
        "M" => Duration::from_secs(timeout_value * SECONDS_IN_MINUTE),
        // Seconds
        "S" => Duration::from_secs(timeout_value),
        // Milliseconds
        "m" => Duration::from_millis(timeout_value),
        // Microseconds
        "u" => Duration::from_micros(timeout_value),
        // Nanoseconds
        "n" => Duration::from_nanos(timeout_value),
        _ => return Err(val),
    };

    Ok(Some(duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parsing() {
        fn map(val: &str) -> MetadataMap {
            let mut map = MetadataMap::new();
            map.insert(GRPC_TIMEOUT_HEADER, val.parse().unwrap());
            map
        }

        assert!(try_parse_grpc_timeout(&map("3H")).unwrap() == Some(Duration::from_hours(3)));
        assert!(try_parse_grpc_timeout(&map("3M")).unwrap() == Some(Duration::from_mins(3)));
        assert!(try_parse_grpc_timeout(&map("3S")).unwrap() == Some(Duration::from_secs(3)));
        assert!(try_parse_grpc_timeout(&map("3m")).unwrap() == Some(Duration::from_millis(3)));
        assert!(try_parse_grpc_timeout(&map("3u")).unwrap() == Some(Duration::from_micros(3)));
        assert!(try_parse_grpc_timeout(&map("3n")).unwrap() == Some(Duration::from_nanos(3)));

        assert!(
            try_parse_grpc_timeout(&MetadataMap::new())
                .unwrap()
                .is_none()
        );
        assert!(try_parse_grpc_timeout(&map("")).is_err());
    }
}
