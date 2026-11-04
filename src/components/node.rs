use std::collections::BTreeMap;
use std::future::Future;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Cancellation;

pub(crate) mod input;
pub(crate) mod schema;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wait {
    pub wait_id: String,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Signal<O> {
    Complete(O),
    Wait(Wait),
}

/// Read-only facts for the node that is currently being invoked.
///
/// A runtime gives a cloned snapshot to a Node or Agent. The snapshot is
/// deliberately made of public data values instead of runtime state types:
/// components stay independent from Workflow/Agent implementations, and a
/// node cannot mutate the owning runtime through this view. `attempts`,
/// `waiting` and `children` contain only the local facts visible at the point
/// of invocation; they are not a second checkpoint or event stream.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct View {
    pub node_id: String,
    pub exec_id: Option<i64>,
    pub attempt_idx: Option<u32>,
    pub status: String,
    pub input: Value,
    #[serde(default)]
    pub sources: BTreeMap<String, Value>,
    pub result: Option<Value>,
    #[serde(default)]
    pub attempts: Vec<Value>,
    pub error: Option<Value>,
    #[serde(default)]
    pub waiting: Vec<Value>,
    pub usage: Option<Value>,
    #[serde(default)]
    pub children: Vec<Value>,
}

impl View {
    /// Build the minimal view used for a standalone Agent invocation.
    pub fn agent(node_id: impl Into<String>, input: Value, state: Option<Value>) -> Self {
        let mut view = Self {
            node_id: node_id.into(),
            status: "running".into(),
            input,
            ..Self::default()
        };
        if let Some(state) = state {
            view.result = state.get("final_message").cloned();
            view.error = state.get("error").cloned();
            view.waiting = state
                .get("waits")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            view.usage = state.get("usage").cloned();
            view.children = state
                .get("nodes")
                .and_then(Value::as_object)
                .map(|nodes| {
                    nodes
                        .iter()
                        .map(|(node_id, state)| {
                            serde_json::json!({
                                "node_id": node_id,
                                "status": state.get("status"),
                                "messages": state.get("messages"),
                                "waits": state.get("waits"),
                                "usage": state.get("usage"),
                                "error": state.get("error"),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            view.attempts = state
                .get("execution")
                .and_then(|execution| execution.get("attempts"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            view.status = state
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("running")
                .into();
        }
        view
    }
}

pub trait Node<I, O, E>: Send + Sync + 'static {
    /// Execute this node as part of one run's cancellation scope.
    fn call(
        &self,
        input: I,
        cancellation: &Cancellation,
    ) -> impl Future<Output = Result<Signal<O>, E>> + Send;

    /// Execute with a read-only local runtime view.
    ///
    /// Existing Nodes only need `call`; the default keeps that contract
    /// unchanged. A Node that needs diagnostics can override this method and
    /// inspect `view` without receiving or mutating the global runtime state.
    fn call_with_view(
        &self,
        input: I,
        _view: &View,
        cancellation: &Cancellation,
    ) -> impl Future<Output = Result<Signal<O>, E>> + Send {
        self.call(input, cancellation)
    }
}

/// A normal async function or closure is a Node without an adapter object.
/// Its successful value is promoted to `Signal::Complete` by the blanket
/// implementation; functions that need `Wait` can implement `Node` directly.
impl<I, O, E, F, Fut> Node<I, O, E> for F
where
    I: Send + 'static,
    F: Fn(I, Cancellation) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, E>> + Send + 'static,
{
    async fn call(&self, input: I, cancellation: &Cancellation) -> Result<Signal<O>, E> {
        (self)(input, cancellation.clone())
            .await
            .map(Signal::Complete)
    }
}
