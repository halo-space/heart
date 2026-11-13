//! Parser and validator for the serializable Heart Native Workflow DSL.
//!
//! This module never creates Runtime state, executes user code, or looks up a
//! function by name. It validates the static graph document. Core component
//! binding and Runtime executor binding happen after parsing.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;

pub const CURRENT_VERSION: u32 = 2;
pub const START: &str = "START";
pub const END: &str = "END";

/// Parse and validate one Heart Native DSL JSON value.
pub fn parse(value: Value) -> Result<Document, Error> {
    Document::parse(value)
}

/// A serializable, executable-agnostic Native DSL document.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub version: u32,
    pub definition_id: String,
    pub nodes: BTreeMap<String, Node>,
    pub edges: Vec<Edge>,
    #[serde(default)]
    /// Join-edge declarations. Each entry describes a relation between
    /// existing nodes, never a node.
    pub join_edges: Vec<JoinEdge>,
    #[serde(default)]
    pub conditions: Vec<Condition>,
    #[serde(default)]
    pub fan_outs: Vec<FanOut>,
    #[serde(default)]
    pub loops: Vec<Loop>,
    #[serde(default)]
    pub failures: Vec<Failure>,
}

impl Document {
    /// Parse and validate a JSON value without creating a Runtime or binding
    /// executable functions.
    pub fn parse(value: Value) -> Result<Self, Error> {
        let document: Self = serde_json::from_value(value)
            .map_err(|error| invalid(format!("invalid workflow DSL: {error}")))?;
        document.validate()?;
        Ok(document)
    }

    /// Validate the complete static graph and every node's local contract.
    pub fn validate(&self) -> Result<(), Error> {
        if self.version != CURRENT_VERSION {
            return Err(invalid(format!(
                "unsupported workflow DSL version: {}",
                self.version
            )));
        }
        if self.definition_id.trim().is_empty() {
            return Err(invalid("definition_id must not be empty"));
        }
        if self.nodes.is_empty() {
            return Err(invalid("workflow DSL must contain at least one node"));
        }

        for (node_id, node) in &self.nodes {
            validate_node_id(node_id)?;
            node.validate(node_id)?;
            for binding in node.attrs.input.values() {
                if binding.from.node_id != START && !self.has_node(&binding.from.node_id) {
                    return Err(invalid(format!(
                        "node {node_id} input references unknown workflow node: {}",
                        binding.from.node_id
                    )));
                }
            }
        }
        self.validate_edges()?;
        self.validate_join_edges()?;
        self.validate_conditions()?;
        self.validate_fan_outs()?;
        self.validate_loops()?;
        self.validate_failures()?;
        self.validate_reachability()?;
        Ok(())
    }

    fn has_node(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }

    fn validate_endpoint(
        &self,
        endpoint: &str,
        allow_start: bool,
        allow_end: bool,
    ) -> Result<(), Error> {
        if endpoint == START {
            if allow_start {
                return Ok(());
            }
            return Err(invalid("START is not allowed at this endpoint"));
        }
        if endpoint == END {
            if allow_end {
                return Ok(());
            }
            return Err(invalid("END is not allowed at this endpoint"));
        }
        if !self.has_node(endpoint) {
            return Err(invalid(format!("unknown workflow node: {endpoint}")));
        }
        Ok(())
    }

    fn validate_edges(&self) -> Result<(), Error> {
        let mut seen = BTreeSet::new();
        for edge in &self.edges {
            if edge.from == END || edge.to == START || (edge.from == START && edge.to == END) {
                return Err(invalid("invalid START/END edge"));
            }
            self.validate_endpoint(&edge.from, true, false)?;
            self.validate_endpoint(&edge.to, false, true)?;
            if edge.from == edge.to {
                return Err(invalid("workflow edge cannot be a self-loop"));
            }
            if !seen.insert((&edge.from, &edge.to)) {
                return Err(invalid("workflow edge is duplicated"));
            }
        }
        if self.edges.iter().all(|edge| edge.from != START) {
            return Err(invalid("workflow requires a START edge"));
        }
        if self.edges.iter().all(|edge| edge.to != END) {
            return Err(invalid("workflow requires an END edge"));
        }
        Ok(())
    }

    fn validate_join_edges(&self) -> Result<(), Error> {
        let mut targets = BTreeSet::new();
        for join in &self.join_edges {
            if join.from.len() < 2
                || join.from.iter().collect::<BTreeSet<_>>().len() != join.from.len()
            {
                return Err(invalid("Join edge requires at least two distinct sources"));
            }
            for source in &join.from {
                self.validate_endpoint(source, false, false)?;
            }
            self.validate_endpoint(&join.to, false, false)?;
            if !targets.insert(&join.to) {
                return Err(invalid("a workflow node cannot have multiple Join edges"));
            }
            if self.edges.iter().any(|edge| edge.to == join.to) {
                return Err(invalid(
                    "Join edge target cannot also have an ordinary incoming edge",
                ));
            }
        }
        Ok(())
    }

    fn validate_conditions(&self) -> Result<(), Error> {
        let mut seen = BTreeSet::new();
        for condition in &self.conditions {
            self.validate_endpoint(&condition.from, false, false)?;
            if condition.to.is_empty()
                || condition.to.iter().collect::<BTreeSet<_>>().len() != condition.to.len()
            {
                return Err(invalid("condition requires distinct targets"));
            }
            for target in &condition.to {
                self.validate_endpoint(target, false, false)?;
            }
            if !seen.insert((&condition.from, &condition.to)) {
                return Err(invalid("condition is duplicated"));
            }
        }
        Ok(())
    }

    fn validate_fan_outs(&self) -> Result<(), Error> {
        let mut seen = BTreeSet::new();
        for fan_out in &self.fan_outs {
            self.validate_endpoint(&fan_out.from, false, false)?;
            self.validate_endpoint(&fan_out.to, false, false)?;
            if fan_out.mode != "multi" {
                return Err(invalid("fan-out mode must be multi"));
            }
            if fan_out.from == fan_out.to {
                return Err(invalid("fan-out cannot be a self-loop"));
            }
            if !seen.insert((&fan_out.from, &fan_out.to)) {
                return Err(invalid("fan-out is duplicated"));
            }
        }
        Ok(())
    }

    fn validate_loops(&self) -> Result<(), Error> {
        let mut seen = BTreeSet::new();
        for loop_edge in &self.loops {
            self.validate_endpoint(&loop_edge.from, false, false)?;
            self.validate_endpoint(&loop_edge.to, false, false)?;
            if loop_edge.from == loop_edge.to {
                return Err(invalid("loop cannot be a self-loop"));
            }
            if !seen.insert((&loop_edge.from, &loop_edge.to)) {
                return Err(invalid("loop is duplicated"));
            }
        }
        Ok(())
    }

    fn validate_failures(&self) -> Result<(), Error> {
        let mut seen = BTreeSet::new();
        for failure in &self.failures {
            if failure.from.is_empty() || failure.to.is_empty() {
                return Err(invalid("failure relation requires from and to"));
            }
            if failure.on != "node_failed" && failure.on != "join_failed" {
                return Err(invalid(
                    "failure relation on must be node_failed or join_failed",
                ));
            }
            if failure.on == "node_failed" && failure.from.len() != 1 {
                return Err(invalid("node_failed relation requires exactly one source"));
            }
            if failure.from.iter().collect::<BTreeSet<_>>().len() != failure.from.len()
                || failure.to.iter().collect::<BTreeSet<_>>().len() != failure.to.len()
            {
                return Err(invalid("failure relation contains duplicate nodes"));
            }
            for source in &failure.from {
                self.validate_endpoint(source, false, false)?;
            }
            for target in &failure.to {
                self.validate_endpoint(target, false, false)?;
            }
            if !seen.insert((&failure.from, &failure.to, &failure.on, &failure.codes)) {
                return Err(invalid("failure relation is duplicated"));
            }
        }
        Ok(())
    }

    fn validate_reachability(&self) -> Result<(), Error> {
        let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for edge in &self.edges {
            if edge.from != START && edge.to != END {
                adjacency
                    .entry(edge.from.as_str())
                    .or_default()
                    .push(edge.to.as_str());
            }
        }
        for join in &self.join_edges {
            for source in &join.from {
                adjacency.entry(source.as_str()).or_default().push(&join.to);
            }
        }
        for condition in &self.conditions {
            for target in &condition.to {
                adjacency
                    .entry(condition.from.as_str())
                    .or_default()
                    .push(target.as_str());
            }
        }
        for fan_out in &self.fan_outs {
            adjacency
                .entry(fan_out.from.as_str())
                .or_default()
                .push(fan_out.to.as_str());
        }
        for failure in &self.failures {
            for source in &failure.from {
                for target in &failure.to {
                    adjacency
                        .entry(source.as_str())
                        .or_default()
                        .push(target.as_str());
                }
            }
        }

        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        for node_id in self.nodes.keys().map(String::as_str) {
            if has_cycle(node_id, &adjacency, &mut visiting, &mut visited) {
                return Err(invalid(
                    "workflow relations cannot contain an implicit cycle; use loops for explicit cycles",
                ));
            }
        }

        let mut reachable = BTreeSet::new();
        let mut queue: VecDeque<&str> = self
            .edges
            .iter()
            .filter(|edge| edge.from == START)
            .map(|edge| edge.to.as_str())
            .collect();
        while let Some(node_id) = queue.pop_front() {
            if reachable.insert(node_id) {
                queue.extend(adjacency.get(node_id).into_iter().flatten().copied());
            }
        }
        if self
            .nodes
            .keys()
            .any(|node_id| !reachable.contains(node_id.as_str()))
        {
            return Err(invalid("every workflow node must be reachable from START"));
        }

        // A graph that is reachable from START but has a branch with no path
        // to END would only fail later inside Runtime with an empty ready set.
        // Validate the reverse relation closure here so a page-authored DSL
        // cannot encode a permanently stuck execution.
        let mut reverse: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        let mut exits = VecDeque::new();
        for edge in &self.edges {
            match (edge.from.as_str(), edge.to.as_str()) {
                (source, END) if source != START => exits.push_back(source),
                (source, target) if source != START && target != END => {
                    reverse.entry(target).or_default().push(source);
                }
                _ => {}
            }
        }
        for join in &self.join_edges {
            for source in &join.from {
                reverse.entry(join.to.as_str()).or_default().push(source);
            }
        }
        for condition in &self.conditions {
            for target in &condition.to {
                reverse
                    .entry(target.as_str())
                    .or_default()
                    .push(condition.from.as_str());
            }
        }
        for fan_out in &self.fan_outs {
            reverse
                .entry(fan_out.to.as_str())
                .or_default()
                .push(fan_out.from.as_str());
        }
        for loop_edge in &self.loops {
            reverse
                .entry(loop_edge.to.as_str())
                .or_default()
                .push(loop_edge.from.as_str());
        }
        for failure in &self.failures {
            for target in &failure.to {
                for source in &failure.from {
                    reverse.entry(target.as_str()).or_default().push(source);
                }
            }
        }
        let mut can_reach_end = BTreeSet::new();
        while let Some(node_id) = exits.pop_front() {
            if can_reach_end.insert(node_id) {
                exits.extend(reverse.get(node_id).into_iter().flatten().copied());
            }
        }
        if self
            .nodes
            .keys()
            .any(|node_id| !can_reach_end.contains(node_id.as_str()))
        {
            return Err(invalid("every workflow node must be able to reach END"));
        }
        Ok(())
    }
}

fn has_cycle<'a>(
    node_id: &'a str,
    adjacency: &BTreeMap<&'a str, Vec<&'a str>>,
    visiting: &mut BTreeSet<&'a str>,
    visited: &mut BTreeSet<&'a str>,
) -> bool {
    if visiting.contains(node_id) {
        return true;
    }
    if !visited.insert(node_id) {
        return false;
    }
    visiting.insert(node_id);
    let cycle = adjacency
        .get(node_id)
        .into_iter()
        .flatten()
        .any(|next| has_cycle(next, adjacency, visiting, visited));
    visiting.remove(node_id);
    cycle
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    #[serde(rename = "type")]
    pub node_type: String,
    pub name: String,
    #[serde(default)]
    pub config: Map<String, Value>,
    #[serde(default)]
    pub attrs: Attrs,
    pub schema: Schema,
    #[serde(default)]
    pub retry: Retry,
}

impl Node {
    fn validate(&self, node_id: &str) -> Result<(), Error> {
        if self.node_type.trim().is_empty() || self.name.trim().is_empty() {
            return Err(invalid(format!("node {node_id} requires type and name")));
        }
        if !matches!(
            self.node_type.as_str(),
            "function" | "agent" | "graph" | "tool"
        ) {
            return Err(invalid(format!(
                "unsupported workflow node type: {}",
                self.node_type
            )));
        }
        reject_sensitive_config(&Value::Object(self.config.clone()))?;
        self.schema.validate(node_id, &self.node_type)?;
        self.attrs.validate(node_id)?;
        self.retry.validate()?;
        validate_code_config(self)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Schema {
    pub input: Value,
    pub output: Value,
}

impl Schema {
    fn validate(&self, node_id: &str, node_type: &str) -> Result<(), Error> {
        let messages_node = matches!(node_type, "tool" | "graph");
        let valid_root = |value: &Value| value.is_object() || (messages_node && value.is_array());
        if !valid_root(&self.input) || !valid_root(&self.output) {
            return Err(invalid(format!(
                "node {node_id} schema input/output must be JSON objects; tool/graph nodes may use Messages arrays"
            )));
        }
        if messages_node && (!self.input.is_array() || !self.output.is_array()) {
            return Err(invalid(format!(
                "node {node_id} tool/graph schema must use Messages array roots"
            )));
        }
        for (direction, schema) in [("input", &self.input), ("output", &self.output)] {
            crate::components::node::schema::compile(schema).map_err(|error| {
                invalid(format!(
                    "node {node_id} schema {direction} is invalid: {}",
                    error.message
                ))
            })?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Attrs {
    #[serde(default)]
    pub input: BTreeMap<String, InputBinding>,
}

impl Attrs {
    fn validate(&self, node_id: &str) -> Result<(), Error> {
        for (field, binding) in &self.input {
            if field.trim().is_empty() {
                return Err(invalid(format!("node {node_id} has an empty input field")));
            }
            binding.validate(node_id)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputBinding {
    pub from: FieldSource,
}

impl InputBinding {
    fn validate(&self, node_id: &str) -> Result<(), Error> {
        if self.from.node_id != START && self.from.node_id.trim().is_empty() {
            return Err(invalid(format!(
                "node {node_id} input source node is empty"
            )));
        }
        if self.from.node_id == END || self.from.field.trim().is_empty() {
            return Err(invalid(format!(
                "node {node_id} has an invalid input source"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FieldSource {
    pub node_id: String,
    pub field: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Retry {
    #[serde(default)]
    pub max_retries: u32,
    #[serde(default = "default_min_delay")]
    pub min_delay_ms: u64,
    #[serde(default = "default_max_delay")]
    pub max_delay_ms: u64,
}

impl Default for Retry {
    fn default() -> Self {
        Self {
            max_retries: 0,
            min_delay_ms: default_min_delay(),
            max_delay_ms: default_max_delay(),
        }
    }
}

impl Retry {
    fn validate(&self) -> Result<(), Error> {
        if self.min_delay_ms > self.max_delay_ms {
            return Err(invalid("retry delay range is invalid"));
        }
        Ok(())
    }
}

fn default_min_delay() -> u64 {
    500
}

fn default_max_delay() -> u64 {
    30_000
}

fn validate_code_config(node: &Node) -> Result<(), Error> {
    let has_language = node.config.contains_key("language");
    let has_entry = node.config.contains_key("entry");
    let has_source = node.config.contains_key("source");
    if node.node_type == "function" && (has_language || has_entry || has_source) {
        for key in ["language", "entry", "source"] {
            let value = node
                .config
                .get(key)
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty());
            if value.is_none() {
                return Err(invalid(format!(
                    "code function config.{key} must be a non-empty string"
                )));
            }
        }
    }
    Ok(())
}

fn reject_sensitive_config(value: &Value) -> Result<(), Error> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let normalized = key.to_ascii_lowercase();
                if matches!(
                    normalized.as_str(),
                    "path"
                        | "workspace"
                        | "workspace_path"
                        | "api_key"
                        | "secret"
                        | "token"
                        | "credential_id"
                ) {
                    return Err(invalid(format!(
                        "workflow DSL config cannot contain credential or local path field: {key}"
                    )));
                }
                reject_sensitive_config(child)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                reject_sensitive_config(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Edge {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JoinEdge {
    pub from: Vec<String>,
    pub to: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    pub from: String,
    pub to: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FanOut {
    pub from: String,
    pub to: String,
    pub mode: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Loop {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub from: Vec<String>,
    pub to: Vec<String>,
    pub on: String,
    #[serde(default)]
    pub codes: Vec<String>,
}

fn validate_node_id(node_id: &str) -> Result<(), Error> {
    if node_id.trim().is_empty() || node_id == START || node_id == END {
        return Err(invalid("node ID is empty or reserved"));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new("INVALID_ARGUMENTS", message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_document() -> Value {
        json!({
            "version": 2,
            "definition_id": "weather",
            "nodes": {
                "search": {
                    "type": "function",
                    "name": "search",
                    "schema": {
                        "input": {"type": "object"},
                        "output": {"type": "object"}
                    },
                    "attrs": {
                        "input": {
                            "query": {"from": {"node_id": "START", "field": "query"}}
                        }
                    }
                }
            },
            "edges": [
                {"from": "START", "to": "search"},
                {"from": "search", "to": "END"}
            ]
        })
    }

    #[test]
    fn parses_native_document_and_defaults_relations_and_retry() {
        let document = Document::parse(valid_document()).expect("valid Native DSL");
        assert_eq!(document.version, CURRENT_VERSION);
        assert_eq!(document.nodes["search"].retry.max_retries, 0);
        assert!(document.join_edges.is_empty());
    }

    #[test]
    fn accepts_page_code_function_without_local_lookup() {
        let mut value = valid_document();
        value["nodes"]["search"]["config"] = json!({
            "language": "python",
            "entry": "reload",
            "source": "def reload(input):\n    return {\"x\": 1}"
        });
        let document = Document::parse(value).expect("code node is static DSL data");
        assert_eq!(document.nodes["search"].config["language"], "python");
    }

    #[test]
    fn rejects_unknown_top_level_field() {
        let mut value = valid_document();
        value["unexpected"] = json!(true);
        let error = Document::parse(value).unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
    }

    #[test]
    fn rejects_missing_code_config_fields() {
        let mut value = valid_document();
        value["nodes"]["search"]["config"] = json!({"language": "python"});
        let error = Document::parse(value).unwrap_err();
        assert!(error.message.contains("config.entry"));
    }

    #[test]
    fn rejects_unknown_source_node() {
        let mut value = valid_document();
        value["nodes"]["search"]["attrs"]["input"]["query"]["from"]["node_id"] = json!("missing");
        let error = Document::parse(value).unwrap_err();
        assert!(error.message.contains("unknown workflow node"));
    }

    #[test]
    fn rejects_duplicate_edges_and_self_edges() {
        let mut value = valid_document();
        value["edges"] = json!([
            {"from": "START", "to": "search"},
            {"from": "START", "to": "search"},
            {"from": "search", "to": "END"}
        ]);
        assert_eq!(
            Document::parse(value).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );

        let mut value = valid_document();
        value["edges"] = json!([
            {"from": "START", "to": "search"},
            {"from": "search", "to": "search"},
            {"from": "search", "to": "END"}
        ]);
        assert_eq!(
            Document::parse(value).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }

    #[test]
    fn rejects_join_mixed_with_ordinary_incoming_edge() {
        let mut value = valid_document();
        value["nodes"]["other"] = json!({
            "type": "function",
            "name": "other",
            "schema": {
                "input": {"type": "object"},
                "output": {"type": "object"}
            }
        });
        value["edges"] = json!([
            {"from": "START", "to": "search"},
            {"from": "search", "to": "other"},
            {"from": "other", "to": "END"}
        ]);
        value["join_edges"] = json!([{"from": ["search", "other"], "to": "other"}]);
        assert_eq!(
            Document::parse(value).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }

    #[test]
    fn rejects_unreachable_node() {
        let mut value = valid_document();
        value["nodes"]["orphan"] = json!({
            "type": "function",
            "name": "orphan",
            "schema": {
                "input": {"type": "object"},
                "output": {"type": "object"}
            }
        });
        let error = Document::parse(value).unwrap_err();
        assert!(error.message.contains("reachable"));
    }
}
