//! Pure Agent planning contracts and logical DAG operations.
//! Runtime execution state and persistence do not belong to Plan or Planner.

use crate::{Cancellation, Messages, agent::Input};
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub type Error = crate::Error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlanNode {
    pub node_id: String,
    pub name: String,
    pub objective: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Edge {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub version: u64,
    #[serde(
        serialize_with = "serialize_nodes",
        deserialize_with = "deserialize_nodes"
    )]
    pub nodes: BTreeMap<String, PlanNode>,
    pub edges: Vec<Edge>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NodeValue {
    name: String,
    objective: String,
}

fn serialize_nodes<S>(nodes: &BTreeMap<String, PlanNode>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut map = serializer.serialize_map(Some(nodes.len()))?;
    for (id, node) in nodes {
        map.serialize_entry(
            id,
            &NodeValue {
                name: node.name.clone(),
                objective: node.objective.clone(),
            },
        )?;
    }
    map.end()
}

fn deserialize_nodes<'de, D>(deserializer: D) -> Result<BTreeMap<String, PlanNode>, D::Error>
where
    D: Deserializer<'de>,
{
    struct Nodes;
    impl<'de> serde::de::Visitor<'de> for Nodes {
        type Value = BTreeMap<String, PlanNode>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a map of unique logical node IDs")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut nodes = BTreeMap::new();
            while let Some((node_id, value)) = map.next_entry::<String, NodeValue>()? {
                if nodes.contains_key(&node_id) {
                    return Err(serde::de::Error::custom("plan node ID is duplicated"));
                }
                nodes.insert(
                    node_id.clone(),
                    PlanNode {
                        node_id,
                        name: value.name,
                        objective: value.objective,
                    },
                );
            }
            Ok(nodes)
        }
    }
    deserializer.deserialize_map(Nodes)
}

impl Plan {
    pub fn new(version: u64) -> Result<Self, Error> {
        if version == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "plan version must be positive",
            ));
        }
        Ok(Self {
            version,
            nodes: BTreeMap::new(),
            edges: Vec::new(),
        })
    }

    pub fn validate(&self) -> Result<(), Error> {
        self.validate_structure()
    }

    /// Validate task fields and DAG structure only. Task names are not Toolkit
    /// dispatch instructions; tool selection belongs to Agent execution.
    pub(crate) fn validate_structure(&self) -> Result<(), Error> {
        if self.version == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "plan version must be positive",
            ));
        }
        for (key, node) in &self.nodes {
            if key != &node.node_id || key.trim().is_empty() {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "plan node map key must match node_id",
                ));
            }
            if node.name.trim().is_empty() || node.objective.trim().is_empty() {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "plan node name and objective must be non-empty",
                ));
            }
        }

        let mut seen = BTreeSet::new();
        let mut outgoing: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.from) || !self.nodes.contains_key(&edge.to) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "plan edge references an unknown node",
                ));
            }
            if edge.from == edge.to {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "plan self edge is not allowed",
                ));
            }
            if !seen.insert((edge.from.as_str(), edge.to.as_str())) {
                return Err(Error::new("INVALID_ARGUMENTS", "plan edge is duplicated"));
            }
            outgoing
                .entry(edge.from.as_str())
                .or_default()
                .push(edge.to.as_str());
        }
        if has_cycle(&self.nodes, &outgoing) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "plan graph must be acyclic",
            ));
        }
        Ok(())
    }

    pub fn add(&mut self, node: PlanNode, from: Vec<String>, to: Vec<String>) -> Result<(), Error> {
        let mut candidate = self.clone();
        if candidate.nodes.contains_key(&node.node_id) {
            return Err(Error::new("CONFLICT", "plan node already exists"));
        }
        let node_id = node.node_id.clone();
        candidate.nodes.insert(node_id.clone(), node);
        for source in from {
            candidate.edges.push(Edge {
                from: source,
                to: node_id.clone(),
            });
        }
        for target in to {
            candidate.edges.push(Edge {
                from: node_id.clone(),
                to: target,
            });
        }
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }

    pub fn insert(&mut self, from_id: &str, node: PlanNode, to_id: &str) -> Result<(), Error> {
        let mut candidate = self.clone();
        let edge = Edge {
            from: from_id.to_owned(),
            to: to_id.to_owned(),
        };
        if !candidate.edges.contains(&edge) {
            return Err(Error::new(
                "NOT_FOUND",
                "plan edge to insert into was not found",
            ));
        }
        if candidate.nodes.contains_key(&node.node_id) {
            return Err(Error::new("CONFLICT", "plan node already exists"));
        }
        candidate.edges.retain(|item| item != &edge);
        let node_id = node.node_id.clone();
        candidate.nodes.insert(node_id.clone(), node);
        candidate.edges.push(Edge {
            from: from_id.to_owned(),
            to: node_id.clone(),
        });
        candidate.edges.push(Edge {
            from: node_id,
            to: to_id.to_owned(),
        });
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }

    pub fn update(
        &mut self,
        node_id: &str,
        node: PlanNode,
        from: Vec<String>,
        to: Vec<String>,
    ) -> Result<(), Error> {
        let mut candidate = self.clone();
        if node.node_id != node_id || !candidate.nodes.contains_key(node_id) {
            return Err(Error::new("NOT_FOUND", "plan node to update was not found"));
        }
        candidate.nodes.insert(node_id.to_owned(), node);
        candidate
            .edges
            .retain(|edge| edge.from != node_id && edge.to != node_id);
        candidate.edges.extend(from.into_iter().map(|source| Edge {
            from: source,
            to: node_id.to_owned(),
        }));
        candidate.edges.extend(to.into_iter().map(|target| Edge {
            from: node_id.to_owned(),
            to: target,
        }));
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }

    pub fn delete(&mut self, node_id: &str) -> Result<(), Error> {
        if !self.nodes.contains_key(node_id) {
            return Err(Error::new("NOT_FOUND", "plan node to delete was not found"));
        }
        let mut candidate = self.clone();
        let mut descendants = BTreeSet::from([node_id.to_owned()]);
        let mut queue = VecDeque::from([node_id.to_owned()]);
        while let Some(current) = queue.pop_front() {
            for edge in &candidate.edges {
                if edge.from == current && descendants.insert(edge.to.clone()) {
                    queue.push_back(edge.to.clone());
                }
            }
        }
        candidate.nodes.retain(|key, _| !descendants.contains(key));
        candidate
            .edges
            .retain(|edge| !descendants.contains(&edge.from) && !descendants.contains(&edge.to));
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }
}

#[allow(async_fn_in_trait)]
/// Return one complete logical DAG from the original Agent input, optional
/// read-only plan snapshot and feedback. Implementations do not own Runtime,
/// execution history or persistence; callers control retries.
pub trait Planner: Send + Sync {
    async fn plan(
        &self,
        query: &Input,
        current_plan: Option<&Plan>,
        feedback: Option<&Messages>,
        cancellation: &Cancellation,
    ) -> Result<Plan, Error>;
}

fn has_cycle<'a>(
    nodes: &'a BTreeMap<String, PlanNode>,
    outgoing: &BTreeMap<&'a str, Vec<&'a str>>,
) -> bool {
    fn visit<'a>(
        node: &'a str,
        outgoing: &BTreeMap<&'a str, Vec<&'a str>>,
        visiting: &mut BTreeSet<&'a str>,
        visited: &mut BTreeSet<&'a str>,
    ) -> bool {
        if visiting.contains(node) {
            return true;
        }
        if !visited.insert(node) {
            return false;
        }
        visiting.insert(node);
        if outgoing
            .get(node)
            .into_iter()
            .flatten()
            .any(|next| visit(next, outgoing, visiting, visited))
        {
            return true;
        }
        visiting.remove(node);
        false
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    nodes
        .keys()
        .map(String::as_str)
        .any(|node| visit(node, outgoing, &mut visiting, &mut visited))
}
