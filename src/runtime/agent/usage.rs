//! Settled model-call accounting for one Agent run. Harness uses the total
//! when writing the Chat projection; storage operations stay outside Runtime.
use std::sync::{Arc, Mutex};

use crate::model::token::{self, Usage};

#[derive(Clone, Default)]
pub(crate) struct Ledger(Arc<Mutex<Vec<Option<Usage>>>>);

impl Ledger {
    pub(crate) fn settle(&self, usage: Option<Usage>) {
        self.0.lock().expect("run usage lock poisoned").push(usage);
    }

    pub(crate) fn refine_last(&self, usage: Option<Usage>) {
        if let Some(last) = self.0.lock().expect("run usage lock poisoned").last_mut() {
            *last = usage;
        }
    }

    pub(crate) fn total(&self) -> Option<Usage> {
        self.0
            .lock()
            .expect("run usage lock poisoned")
            .iter()
            .cloned()
            .fold(None, token::merge)
    }
}
