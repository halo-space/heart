//! Private field adapter shared by execution paths. This module does not
//! define an input protocol, own state, infer bindings or merge Messages.
use serde_json::{Map, Value};

use crate::{Error, Messages};

/// Each tuple is (target field, actual source node ID, literal source field).
/// Callers select successful executions before handing their Messages here.
pub(crate) fn fields<'a>(
    bindings: impl IntoIterator<Item = (&'a str, &'a str, &'a str)>,
    mut source: impl FnMut(&str) -> Result<&'a Messages, Error>,
) -> Result<Map<String, Value>, Error> {
    let mut values = Map::new();
    for (target, node_id, field) in bindings {
        if target.is_empty() || node_id.is_empty() || field.is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "input binding contains an empty field or node ID",
            ));
        }
        if values.contains_key(target) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                format!("duplicate input field {target}"),
            ));
        }
        let value = source(node_id)?
            .as_slice()
            .iter()
            .rev()
            .find_map(|message| message.values.get(field))
            .cloned()
            .ok_or_else(|| {
                Error::new(
                    "INVALID_ARGUMENTS",
                    format!("missing field {field} from node {node_id}"),
                )
            })?;
        values.insert(target.to_owned(), value);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Plan;
    use crate::runtime::agent::{graph::Graph, node::Status};
    use serde::Deserialize;
    use serde_json::json;

    fn sources() -> Graph {
        let plan: Plan = serde_json::from_value(json!({
            "version":1,"nodes":{
                "a":{"name":"search","objective":"北京"},
                "b":{"name":"search","objective":"上海"},
                "c":{"name":"compare","objective":"比较"}
            },"edges":[{"from":"a","to":"c"},{"from":"b","to":"c"}]
        }))
        .unwrap();
        let mut graph = Graph::new(plan).unwrap();
        for (node_id, exec_id, city) in [("a", 101, "北京"), ("b", 102, "上海")] {
            graph.start(node_id, exec_id).unwrap();
            graph
                .succeed(
                    node_id,
                    Messages::new([crate::Message::function(
                        json!({"city":city, "weather":{"temperature":22}}),
                    )]),
                    None,
                )
                .unwrap();
        }
        graph
    }

    #[test]
    fn agent_sources_bind_to_typed_input_without_merging_or_mutation() {
        #[derive(Deserialize)]
        struct Input {
            first: String,
            second: String,
        }
        let graph = sources();
        let before = graph.clone();
        let sources = graph.source_messages("c").unwrap();
        let values = fields([("first", "a", "city"), ("second", "b", "city")], |id| {
            sources
                .get(id)
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "missing source"))
        })
        .unwrap();
        let input: Input = serde_json::from_value(Value::Object(values)).unwrap();
        assert_eq!(input.first, "北京");
        assert_eq!(input.second, "上海");
        assert_eq!(graph, before);
        assert_eq!(graph.state("c").unwrap().status, Status::Ready);
    }

    #[test]
    fn missing_sources_and_fields_are_not_inferred_from_another_node() {
        let graph = sources();
        let sources = graph.source_messages("c").unwrap();
        for binding in [("city", "missing", "city"), ("city", "a", "missing")] {
            assert_eq!(
                fields([binding], |id| sources.get(id).ok_or_else(|| Error::new(
                    "INVALID_ARGUMENTS",
                    "missing source"
                )))
                .unwrap_err()
                .code,
                "INVALID_ARGUMENTS"
            );
        }
    }

    #[test]
    fn field_is_literal_and_metadata_content_do_not_supply_business_input() {
        let mut message = crate::Message::function(json!({"a.b":7,"a":{"b":8}}));
        message.metadata.insert("secret".into(), json!(9));
        let messages = Messages::new([message]);
        assert_eq!(
            fields([("value", "a", "a.b")], |_| Ok(&messages)).unwrap()["value"],
            7
        );
        assert!(fields([("value", "a", "secret")], |_| Ok(&messages)).is_err());
    }

    #[test]
    fn latest_declared_value_preserves_null_arrays_and_objects() {
        let messages = Messages::new([
            crate::Message::function(json!({"x":1})),
            crate::Message::function(json!({"x":null,"items":[1,2],"nested":{"x":3}})),
        ]);
        let values = fields(
            [
                ("x", "a", "x"),
                ("items", "a", "items"),
                ("nested", "a", "nested"),
            ],
            |_| Ok(&messages),
        )
        .unwrap();
        assert!(values["x"].is_null());
        assert_eq!(values["items"], json!([1, 2]));
        assert_eq!(values["nested"], json!({"x":3}));
    }

    #[test]
    fn duplicate_targets_and_empty_bindings_are_explicit() {
        let messages = Messages::new([crate::Message::function(json!({"x":1,"y":2}))]);
        assert!(
            fields([("value", "a", "x"), ("value", "a", "y")], |_| Ok(
                &messages
            ))
            .is_err()
        );
        for binding in [("", "a", "x"), ("value", "", "x"), ("value", "a", "")] {
            assert!(fields([binding], |_| Ok(&messages)).is_err());
        }
        assert!(
            fields([], |_| panic!("empty bindings do not read sources"))
                .unwrap()
                .is_empty()
        );
    }
}
