//! SSE framing for protocol responses.
//!
//! This module only serializes one already-converted protocol response. It
//! does not own an HTTP connection or a Runtime event stream.

use crate::Error;
use crate::components::protocol::a2a::StreamResponse;

/// Encode one A2A response as a standard SSE `data` frame.
pub fn encode(response: &StreamResponse) -> Result<String, Error> {
    let data = serde_json::to_string(response)
        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
    Ok(format!("data: {data}\n\n"))
}

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_server_sse_tests.rs"]
mod tests;
