//! Nested Workflow execution and child-state reconstruction.

use super::*;

pub(super) struct SubgraphFailure {
    pub(super) error: Error,
    pub(super) state: Option<graph::State>,
}

// Derive the next outer Attempt's child scene from existing graph facts. The
// failed Attempt keeps its full unmodified scene; only complete, unaffected
// successful child executions are inherited into this new scene.
pub(super) fn retry_graph_state(
    workflow: &CompiledWorkflow,
    previous: &graph::State,
) -> graph::State {
    let definition = workflow.definition();
    let mut affected = previous
        .nodes
        .iter()
        .filter(|(_, state)| {
            state
                .executions
                .values()
                .any(|execution| execution.status == node::status::execution::Status::Failed)
        })
        .map(|(node_id, _)| node_id.clone())
        .collect::<BTreeSet<_>>();
    let mut pending = affected.iter().cloned().collect::<Vec<_>>();
    while let Some(node_id) = pending.pop() {
        let mut downstream = outgoing(definition, &node_id);
        downstream.extend(
            definition
                .graph
                .conditional_edges
                .iter()
                .filter(|relation| relation.from == node_id)
                .flat_map(|relation| relation.to.iter().cloned()),
        );
        downstream.extend(
            definition
                .graph
                .loop_definitions
                .iter()
                .filter(|relation| relation.from == node_id)
                .map(|relation| relation.to.clone()),
        );
        downstream.extend(
            definition
                .graph
                .failure_relations
                .iter()
                .filter(|relation| relation.from.contains(&node_id))
                .flat_map(|relation| relation.to.iter().cloned()),
        );
        for to in downstream {
            if definition.nodes.contains_key(&to) && affected.insert(to.clone()) {
                pending.push(to);
            }
        }
    }
    let mut state = graph::State::new(previous.chat_id, definition);
    state.initial = previous.initial.clone();
    state.step = previous
        .nodes
        .iter()
        .filter(|(node_id, _)| affected.contains(*node_id))
        .flat_map(|(_, node)| node.executions.values().map(|execution| execution.idx))
        .min()
        .unwrap_or(previous.step);
    for (node_id, node) in &previous.nodes {
        if affected.contains(node_id) {
            continue;
        }
        let executions = node
            .executions
            .iter()
            .filter(|(_, execution)| execution.status == node::status::execution::Status::Succeeded)
            .map(|(exec_id, execution)| (*exec_id, execution.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        if !executions.is_empty() {
            state.nodes.insert(
                node_id.clone(),
                node::State {
                    status: node::status::Status::Succeeded,
                    executions,
                },
            );
        }
    }
    state.recompute_usage();
    state
}

type SubgraphFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<(Value, graph::State), SubgraphFailure>>
            + Send
            + 'static,
    >,
>;

pub(super) struct SubgraphScenes {
    pub(super) snapshot: Option<Box<graph::State>>,
    pub(super) history: Option<Box<graph::State>>,
    pub(super) reply: Option<RecoveryReply>,
}

pub(super) fn run_subgraph(
    workflow: Arc<CompiledWorkflow>,
    input: Value,
    cancellation: crate::Cancellation,
    next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
    chat_id: i64,
    scenes: SubgraphScenes,
) -> SubgraphFuture {
    Box::pin(async move {
        let SubgraphScenes {
            snapshot,
            history,
            reply,
        } = scenes;
        let messages: Messages =
            serde_json::from_value(input).map_err(|error| SubgraphFailure {
                error: Error::new("INVALID_ARGUMENTS", error.to_string()),
                state: None,
            })?;
        // A nested Graph Node can have a new execution while still reusing
        // its unchanged-input internal successes from the parent's history.
        let stable_input = snapshot
            .as_ref()
            .and_then(|scene| scene.initial.as_ref())
            .unwrap_or(&messages);
        let history = history.filter(|previous| {
            previous.chat_id == chat_id
                && previous.definition_id == workflow.definition().definition_id
                && previous.version == workflow.definition().version
                && previous.initial.as_ref() == Some(stable_input)
        });
        let snapshot = snapshot.or_else(|| {
            history
                .as_ref()
                .map(|previous| Box::new(retry_graph_state(&workflow, previous)))
        });
        let mut runtime = match snapshot {
            Some(snapshot) => Engine::from_snapshot(
                chat_id,
                (*workflow).clone(),
                serde_json::to_value(*snapshot).map_err(|error| SubgraphFailure {
                    error: Error::new("SERIALIZATION", error.to_string()),
                    state: None,
                })?,
            )
            .map_err(|error| SubgraphFailure { error, state: None })?,
            None => Engine::new(chat_id, (*workflow).clone())
                .map_err(|error| SubgraphFailure { error, state: None })?,
        };
        runtime.retry_source = history;
        runtime.recovery_reply = reply;
        let run_input = runtime.state.initial.clone().unwrap_or(messages);
        let output = runtime
            .execute(run_input, next_exec_id, cancellation)
            .await
            .map_err(|error| SubgraphFailure {
                error,
                state: Some(runtime.state.clone()),
            })?;
        let state = runtime.state.clone();
        let value = serde_json::to_value(output).map_err(|error| SubgraphFailure {
            error: Error::new("SERIALIZATION", error.to_string()),
            state: Some(state.clone()),
        })?;
        Ok((value, state))
    })
}

pub(super) fn outgoing(definition: &Definition, node_id: &str) -> Vec<String> {
    let mut result: Vec<String> = definition
        .graph
        .ordinary_edges
        .iter()
        .filter(|edge| edge.from.node() == Some(node_id))
        .map(|edge| match &edge.to {
            Endpoint::Node(id) => id.clone(),
            Endpoint::End => graph::END.into(),
            Endpoint::Start => graph::START.into(),
        })
        .collect();
    result.extend(
        definition
            .graph
            .fan_out_definitions
            .iter()
            .filter(|relation| relation.from == node_id)
            .map(|relation| relation.to.clone()),
    );
    result.extend(
        definition
            .graph
            .join_edges
            .iter()
            .filter(|relation| relation.from.iter().any(|source| source == node_id))
            .map(|relation| relation.to.clone()),
    );
    result.sort();
    result.dedup();
    result
}
