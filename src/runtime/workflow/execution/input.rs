//! Workflow input binding and route projection.
//!
//! These helpers read the current graph State and construct the local input
//! or diagnostic view for one node. They do not execute nodes or persist a
//! second state representation.

use super::*;

impl Engine {
    pub(super) fn reached_end(&self) -> bool {
        self.state.nodes.values().any(|state| {
            state.executions.values().any(|execution| {
                execution.status == node::status::execution::Status::Succeeded
                    && execution.to.iter().any(|target| target == graph::END)
            })
        })
    }

    pub(super) fn end_messages(&self) -> Messages {
        let mut messages = Messages::default();
        for node_id in self.definition().nodes.keys() {
            if let Some(state) = self.state.nodes.get(node_id) {
                for execution in state.executions.values() {
                    if execution.status == node::status::execution::Status::Succeeded
                        && execution.to.iter().any(|target| target == graph::END)
                    {
                        messages.0.extend(execution.messages.0.clone());
                    }
                }
            }
        }
        messages
    }

    pub(super) fn outgoing(
        &self,
        node_id: &str,
        messages: &Messages,
    ) -> Result<Vec<String>, Error> {
        let mut targets = outgoing(self.definition(), node_id);
        let allowed: BTreeSet<String> = self
            .definition()
            .graph
            .conditional_edges
            .iter()
            .filter(|relation| relation.from == node_id)
            .flat_map(|relation| relation.to.iter().cloned())
            .collect();
        if !allowed.is_empty() {
            let selected = self.workflow.route(node_id, messages)?;
            if selected.is_empty() {
                return Err(Error::new(
                    "INVALID_ROUTE",
                    "conditional route selected no target",
                ));
            }
            if selected.iter().any(|target| !allowed.contains(target)) {
                return Err(Error::new(
                    "INVALID_ROUTE",
                    "conditional route selected an undeclared target",
                ));
            }
            targets.retain(|target| !allowed.contains(target));
            targets.extend(selected);
        }
        let loop_targets: Vec<String> = self
            .definition()
            .graph
            .loop_definitions
            .iter()
            .filter(|relation| relation.from == node_id)
            .filter(|_| self.workflow.loop_enabled(node_id, messages))
            .map(|relation| relation.to.clone())
            .collect();
        targets.extend(loop_targets);
        targets.sort();
        targets.dedup();
        Ok(targets)
    }

    pub(super) fn first_failure(&self) -> Option<Error> {
        self.state
            .nodes
            .values()
            .flat_map(|state| state.executions.values())
            .find_map(|execution| execution.error.clone())
    }

    pub(super) fn cancel_active(&mut self) {
        let active: Vec<(String, i64)> = self
            .state
            .nodes
            .iter()
            .flat_map(|(node_id, state)| {
                state
                    .executions
                    .values()
                    .filter(|execution| {
                        matches!(
                            execution.status,
                            node::status::execution::Status::Pending
                                | node::status::execution::Status::Running
                                | node::status::execution::Status::Waiting
                        )
                    })
                    .map(|execution| (node_id.clone(), execution.exec_id))
            })
            .collect();
        for (node_id, exec_id) in active {
            let _ = self.state.cancel_execution(&node_id, exec_id);
        }
    }

    pub(super) fn source_executions(&self, node_id: &str) -> Result<Vec<ExecutionRef>, Error> {
        self.state
            .activation_sources(self.definition(), node_id)
            .ok_or_else(|| Error::new("INVALID_STATE", "node has no committed activation"))
    }

    pub(super) fn input_for(&self, node_id: &str, from: &[ExecutionRef]) -> Result<Value, Error> {
        if self.is_failure_activation(node_id, from) {
            if self.definition().nodes[node_id].agent {
                return serde_json::to_value(Input::default())
                    .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
            }
            if self.workflow.input_is_messages(node_id) {
                return serde_json::to_value(Messages::default())
                    .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
            }
            return Ok(Value::Object(serde_json::Map::new()));
        }
        if self.definition().nodes[node_id].agent {
            return serde_json::to_value(self.agent_input_for(node_id, from)?)
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
        }
        if !self.definition().nodes[node_id].attrs.input.is_empty() {
            return self.bound_values(node_id, from).map(Value::Object);
        }
        if from.is_empty() {
            if self.workflow.input_is_messages(node_id) {
                return serde_json::to_value(self.state.initial.as_ref().ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                })?)
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
            }
            let values = self
                .state
                .initial
                .as_ref()
                .ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                })?
                .as_slice()
                .last()
                .map(|message| message.values.clone())
                .unwrap_or_default();
            return Ok(Value::Object(values));
        }
        if self.workflow.input_is_messages(node_id) {
            let mut messages = Messages::default();
            for source in from {
                let execution = self.state.nodes[&source.node_id]
                    .executions
                    .get(&source.exec_id)
                    .ok_or_else(|| Error::new("NOT_FOUND", "source execution missing"))?;
                messages.0.extend(execution.messages.0.clone());
            }
            return serde_json::to_value(messages)
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
        }
        self.bound_values(node_id, from).map(Value::Object)
    }

    /// Snapshot only the current node's local execution facts for an
    /// invocation. This deliberately does not expose the owning `State` or
    /// graph mutators; the Node/Agent receives a clone through `View`.
    pub(super) fn view_for(
        &self,
        node_id: &str,
        exec_id: i64,
        input: Value,
        attempt_idx: Option<u32>,
    ) -> Result<View, Error> {
        let execution = self
            .state
            .nodes
            .get(node_id)
            .and_then(|state| state.executions.get(&exec_id));
        let mut view = View {
            node_id: node_id.to_owned(),
            exec_id: Some(exec_id),
            attempt_idx,
            status: execution
                .map(|execution| match execution.status {
                    node::status::execution::Status::Pending => {
                        crate::components::node::Status::Pending
                    }
                    node::status::execution::Status::Running => {
                        crate::components::node::Status::Running
                    }
                    node::status::execution::Status::Waiting => {
                        crate::components::node::Status::Waiting
                    }
                    node::status::execution::Status::Succeeded => {
                        crate::components::node::Status::Succeeded
                    }
                    node::status::execution::Status::Failed => {
                        crate::components::node::Status::Failed
                    }
                    node::status::execution::Status::Cancelled => {
                        crate::components::node::Status::Cancelled
                    }
                })
                .unwrap_or(crate::components::node::Status::Pending),
            input,
            ..View::default()
        };
        if let Some(execution) = execution {
            view.result = (!execution.messages.is_empty())
                .then(|| serde_json::to_value(&execution.messages))
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
            view.attempts = execution
                .attempts
                .iter()
                .map(|attempt| {
                    serde_json::to_value(attempt)
                        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))
                })
                .collect::<Result<_, _>>()?;
            view.error = execution
                .error
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
            view.waiting = execution
                .waits
                .values()
                .map(|wait| {
                    serde_json::to_value(wait)
                        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))
                })
                .collect::<Result<_, _>>()?;
            view.usage = execution
                .usage
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
            if let Some(attempt) = execution.attempts.last() {
                if let Some(scene) = &attempt.agent {
                    view.children.extend(
                        scene
                            .get("nodes")
                            .and_then(Value::as_object)
                            .into_iter()
                            .flat_map(|nodes| {
                                nodes.iter().map(|(child_id, state)| {
                                    serde_json::json!({
                                        "node_id": child_id,
                                        "status": state.get("status"),
                                        "messages": state.get("messages"),
                                        "waits": state.get("waits"),
                                        "usage": state.get("usage"),
                                        "error": state.get("error"),
                                    })
                                })
                            }),
                    );
                }
                if let Some(graph) = &attempt.graph {
                    let nodes = graph
                        .nodes
                        .iter()
                        .map(|(child_id, state)| {
                            serde_json::json!({
                                "node_id": child_id,
                                "status": state.status,
                                "executions": state.executions.len(),
                            })
                        })
                        .collect::<Vec<_>>();
                    view.children.push(serde_json::json!({
                        "type": "workflow",
                        "status": graph.status,
                        "step": graph.step,
                        "nodes": nodes,
                    }));
                }
            }
            for source in &execution.from {
                if let Some(source_execution) = self
                    .state
                    .nodes
                    .get(&source.node_id)
                    .and_then(|state| state.executions.get(&source.exec_id))
                {
                    let key = format!("{}#{}", source.node_id, source.exec_id);
                    view.sources.insert(
                        key,
                        serde_json::json!({
                            "node_id": source.node_id,
                            "exec_id": source.exec_id,
                            "status": source_execution.status,
                            "messages": source_execution.messages,
                        }),
                    );
                }
            }
            let mut bindings = serde_json::Map::new();
            for (target, binding) in &self.definition().nodes[node_id].attrs.input {
                let source = execution
                    .from
                    .iter()
                    .find(|source| source.node_id == binding.from.node_id);
                let value = source
                    .and_then(|source| {
                        self.state
                            .nodes
                            .get(&source.node_id)
                            .and_then(|state| state.executions.get(&source.exec_id))
                    })
                    .and_then(|source| {
                        source
                            .messages
                            .as_slice()
                            .iter()
                            .rev()
                            .find_map(|message| message.values.get(&binding.from.field))
                    })
                    .or_else(|| {
                        (binding.from.node_id == graph::START)
                            .then(|| {
                                self.state
                                    .initial
                                    .as_ref()?
                                    .as_slice()
                                    .iter()
                                    .rev()
                                    .find_map(|message| message.values.get(&binding.from.field))
                            })
                            .flatten()
                    })
                    .cloned()
                    .unwrap_or(Value::Null);
                bindings.insert(
                    target.clone(),
                    serde_json::json!({
                        "node_id": binding.from.node_id,
                        "field": binding.from.field,
                        "value": value,
                    }),
                );
            }
            if !bindings.is_empty() {
                view.sources
                    .insert("attrs.input".into(), Value::Object(bindings));
            }
        }
        Ok(view)
    }

    pub(super) fn agent_input_for(
        &self,
        node_id: &str,
        from: &[ExecutionRef],
    ) -> Result<Input, Error> {
        let attrs = &self.definition().nodes[node_id].attrs.input;
        if !attrs.is_empty() {
            let values = self.bound_values(node_id, from)?;
            return serde_json::from_value(Value::Object(values))
                .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()));
        }
        let messages = if from.is_empty() {
            self.state.initial.clone().unwrap_or_default()
        } else {
            let mut messages = Messages::default();
            for source in from {
                if let Some(execution) = self.state.nodes[&source.node_id]
                    .executions
                    .get(&source.exec_id)
                {
                    messages.0.extend(execution.messages.0.clone());
                }
            }
            messages
        };
        Ok(messages_to_agent_input(&messages))
    }

    // Only field adaptation is shared with Agent execution. Graph activation,
    // execution selection, state types and Pregel behavior remain separate.
    pub(super) fn bound_values(
        &self,
        node_id: &str,
        from: &[ExecutionRef],
    ) -> Result<serde_json::Map<String, Value>, Error> {
        let bindings =
            self.definition().nodes[node_id]
                .attrs
                .input
                .iter()
                .map(|(field, binding)| {
                    (
                        field.as_str(),
                        binding.from.node_id.as_str(),
                        binding.from.field.as_str(),
                    )
                });
        crate::components::node::input::fields(bindings, |source_id| {
            if source_id == graph::START {
                return self.state.initial.as_ref().ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                });
            }
            let source = from
                .iter()
                .find(|source| source.node_id == source_id)
                .ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", format!("missing source {source_id}"))
                })?;
            let execution = self
                .state
                .nodes
                .get(source_id)
                .and_then(|state| state.executions.get(&source.exec_id))
                .ok_or_else(|| Error::new("NOT_FOUND", "source execution missing"))?;
            if execution.status != node::status::execution::Status::Succeeded {
                return Err(Error::new(
                    "INVALID_STATE",
                    "input source execution has not succeeded",
                ));
            }
            Ok(&execution.messages)
        })
    }

    pub(super) fn is_failure_activation(&self, node_id: &str, from: &[ExecutionRef]) -> bool {
        self.definition()
            .graph
            .failure_relations
            .iter()
            .any(|relation| {
                relation.to.iter().any(|target| target == node_id)
                    && relation.from.len() == from.len()
                    && relation
                        .from
                        .iter()
                        .all(|source| from.iter().any(|item| &item.node_id == source))
                    && from.iter().any(|source| {
                        self.state
                            .nodes
                            .get(&source.node_id)
                            .and_then(|state| state.executions.get(&source.exec_id))
                            .is_some_and(|execution| {
                                execution.status == node::status::execution::Status::Failed
                                    && (relation.codes.is_empty()
                                        || execution.error.as_ref().is_some_and(|error| {
                                            relation.codes.iter().any(|code| code == &error.code)
                                        }))
                            })
                    })
                    && (relation.on == "node_failed" || relation.on == "join_failed")
            })
    }

    pub(super) fn output_messages(&self, node_id: &str, value: Value) -> Result<Messages, Error> {
        if self.workflow.output_is_messages(node_id) {
            serde_json::from_value(value)
                .map_err(|error| Error::new("INVALID_OUTPUT", error.to_string()))
        } else {
            Ok(Messages::new([crate::Message::function(value)]))
        }
    }
}
