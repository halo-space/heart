//! Settled model-call accounting for one Agent run. The history lifecycle uses
//! the total when writing the Chat projection; persistence stays outside Runtime.
use std::sync::{Arc, Mutex};

use crate::model::token::{self, Usage};

#[derive(Clone, Default)]
pub(crate) struct Ledger {
    calls: Arc<Mutex<Vec<Option<Usage>>>>,
    children: Arc<Mutex<Vec<Option<Usage>>>>,
}

impl Ledger {
    pub(crate) fn settle(&self, usage: Option<Usage>) {
        self.calls
            .lock()
            .expect("run usage lock poisoned")
            .push(usage);
    }

    pub(crate) fn settle_child(&self, usage: Option<Usage>) {
        if usage.is_some() {
            self.children
                .lock()
                .expect("child usage lock poisoned")
                .push(usage);
        }
    }

    pub(crate) fn refine_last(&self, usage: Option<Usage>) {
        if let Some(last) = self
            .calls
            .lock()
            .expect("run usage lock poisoned")
            .last_mut()
        {
            *last = usage;
        }
    }

    pub(crate) fn total(&self) -> Option<Usage> {
        let calls = self
            .calls
            .lock()
            .expect("run usage lock poisoned")
            .iter()
            .cloned()
            .fold(None, token::merge);
        self.children
            .lock()
            .expect("child usage lock poisoned")
            .iter()
            .cloned()
            .fold(calls, token::merge)
    }
}
