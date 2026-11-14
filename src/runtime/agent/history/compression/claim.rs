//! Single-flight guard and usage accounting for background compression.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::model::token::Usage;

type Key = (usize, i64);
pub(crate) static ACTIVE: OnceLock<Mutex<HashSet<Key>>> = OnceLock::new();

pub(crate) struct Claim(Key);

impl Claim {
    pub(crate) fn take(key: Key) -> Option<Self> {
        let mut active = ACTIVE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        active.insert(key).then(|| Self(key))
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        ACTIVE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.0);
    }
}

/// Use provider Usage only; never tokenize historical payloads a second time.
pub(crate) fn tokens(usage: Option<&Usage>) -> u64 {
    usage.map_or(0, |usage| {
        usage.total.unwrap_or_else(|| {
            usage
                .input
                .unwrap_or(0)
                .saturating_add(usage.output.unwrap_or(0))
        })
    })
}
