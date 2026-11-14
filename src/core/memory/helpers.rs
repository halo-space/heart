//! Pure validation and scoring helpers for the in-memory Memory backend.
//!
//! The backend owns storage and async CRUD; these helpers remain stateless so
//! filtering and vector scoring can be tested without a second storage type.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::components::memory::{Error, Item, Request};

pub(super) fn pagination(
    options: &serde_json::Map<String, Value>,
) -> Result<(usize, usize), Error> {
    let invalid = || Error::new("INVALID_ARGUMENTS", "invalid memory pagination");
    let pagination = match options.get("pagination") {
        None => return Ok((0, 20)),
        Some(Value::Object(pagination)) => pagination,
        Some(_) => return Err(invalid()),
    };
    let read = |field, default| match pagination.get(field) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(invalid),
    };
    let offset = read("offset", 0)?;
    let limit = read("limit", 20)?;
    if limit == 0 {
        return Err(Error::new("INVALID_ARGUMENTS", "invalid memory pagination"));
    }
    Ok((offset, limit))
}

/// Find the existing record an update replaces. Content and vectors are not
/// part of this identity comparison.
pub(super) fn identity_matches(left: &Item, right: &Item) -> bool {
    left.user_id == right.user_id
        && left.agent_id == right.agent_id
        && left.session_id == right.session_id
        && left.r#type == right.r#type
}

pub(super) fn request_as_item(
    request: &Request,
    version: u64,
    created_time: i64,
    updated_time: i64,
) -> Item {
    Item {
        id: request.id,
        r#type: request.r#type.clone(),
        user_id: request.user_id,
        agent_id: request.agent_id,
        session_id: request.session_id,
        from: request.from,
        to: request.to,
        content: request.content.clone(),
        version,
        created_time,
        updated_time,
        metadata: request.metadata.clone(),
    }
}

pub(super) fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(-1.0, 1.0)
    }
}

pub(super) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
