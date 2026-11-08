//! Graph entity/relation extraction over model-complete document content.

use super::{Document, Error, complete_request};
use crate::Cancellation;
pub use crate::components::document::graph_types::{Evidence, Method};
use crate::components::message::{Message, Messages};
use crate::components::model::chat::Chat;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

pub type Options = Map<String, Value>;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Entity {
    pub id: String,
    pub r#type: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub attrs: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<Evidence>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Relation {
    pub from: String,
    pub to: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub attrs: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<Evidence>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Output {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
}

#[derive(Clone, Debug)]
pub(super) struct EntityType {
    pub(super) name: String,
    pub(super) description: String,
}
#[derive(Clone, Debug)]
pub(super) struct RelationType {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) from: Vec<String>,
    pub(super) to: Vec<String>,
}

pub struct Graph<C> {
    chat: C,
    entity_types: Vec<EntityType>,
    relation_types: Vec<RelationType>,
}

impl<C> Graph<C>
where
    C: Chat,
{
    pub fn new(options: Options, chat: C) -> Result<Self, Error> {
        let entity_values = options
            .get("entity_types")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "entity_types is required"))?;
        let mut entity_types = Vec::new();
        let mut names = BTreeSet::new();
        for value in entity_values {
            let object = value
                .as_object()
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "entity type must be object"))?;
            let name = object.get("name").and_then(Value::as_str).unwrap_or("");
            let description = object
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if name.trim().is_empty() || description.is_empty() || !names.insert(name.to_owned()) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "entity type name and description must be unique and non-empty",
                ));
            }
            entity_types.push(EntityType {
                name: name.into(),
                description: description.into(),
            });
        }
        let relation_values = options
            .get("relation_types")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "relation_types is required"))?;
        let mut relation_types = Vec::new();
        let mut relation_names = BTreeSet::new();
        for value in relation_values {
            let object = value
                .as_object()
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "relation type must be object"))?;
            let name = object.get("name").and_then(Value::as_str).unwrap_or("");
            let description = object
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let from = strings(object.get("from"))?;
            let to = strings(object.get("to"))?;
            if name.trim().is_empty()
                || description.is_empty()
                || !relation_names.insert(name.to_owned())
                || from.iter().any(|item| !names.contains(item))
                || to.iter().any(|item| !names.contains(item))
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "relation type is invalid or references unknown entity type",
                ));
            }
            relation_types.push(RelationType {
                name: name.into(),
                description: description.into(),
                from,
                to,
            });
        }
        Ok(Self {
            chat,
            entity_types,
            relation_types,
        })
    }

    pub async fn run(
        &self,
        documents: &[Document],
        cancellation: &Cancellation,
    ) -> Result<Output, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "graph extraction cancelled"));
        }
        let payload = serde_json::json!({
            "documents": documents,
            "entity_types": self.entity_types.iter().map(|item| serde_json::json!({"name": item.name, "description": item.description})).collect::<Vec<_>>(),
            "relation_types": self.relation_types.iter().map(|item| serde_json::json!({"name": item.name, "description": item.description, "from": item.from, "to": item.to})).collect::<Vec<_>>(),
        });
        let mut input = Message::new(crate::components::message::Role::User);
        input.content.push(crate::components::model::content::part::Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(serde_json::json!({"value": format!("Extract the graph from these documents using the declared entity and relation types. Return only a JSON object with entities and relations arrays. Entity shape: {{\"id\": string, \"type\": string, \"name\": string, \"attrs\": object, \"evidence\": [{{\"content\": string, \"method\": \"text\"}}]}}. Relation shape: {{\"from\": entity id, \"to\": entity id, \"kind\": relation type, \"attrs\": object, \"evidence\": [{{\"content\": string, \"method\": \"text\"}}]}}. Evidence must always be an array of objects, never a string; use an empty array if no exact source quotation is available. Do not invent evidence or identifiers absent from the source. Input: {payload}")})),
            });
        let message = complete_request(
            &self.chat,
            crate::model::chat::Request {
                messages: Messages::new([input]),
                stream: true,
                response_format: Some(crate::model::chat::ResponseFormat::JsonObject),
                ..crate::model::chat::Request::default()
            },
            cancellation,
        )
        .await?;
        decode_output(&message, &self.entity_types, &self.relation_types)
    }
}

pub(super) fn decode_output(
    message: &Message,
    entity_types: &[EntityType],
    relation_types: &[RelationType],
) -> Result<Output, Error> {
    let values =
        if message.values.contains_key("entities") || message.values.contains_key("relations") {
            message.values.clone()
        } else {
            // Some adapters expose the complete JSON as text. Decode only
            // that complete response, never Delta or reasoning content.
            let text = message
                .content
                .iter()
                .filter(|part| part.r#type == "text")
                .filter_map(|part| part.data.get("value").and_then(Value::as_str))
                .collect::<String>();
            serde_json::from_str::<Map<String, Value>>(&text).map_err(|error| {
                Error::new("INVALID_RESPONSE", "graph response must be a JSON object")
                    .with_details(error.to_string())
            })?
        };
    let entities: Vec<Entity> = serde_json::from_value(
        values
            .get("entities")
            .cloned()
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "graph entities are missing"))?,
    )
    .map_err(|error| Error::new("INVALID_RESPONSE", error.to_string()))?;
    let relations: Vec<Relation> = serde_json::from_value(
        values
            .get("relations")
            .cloned()
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "graph relations are missing"))?,
    )
    .map_err(|error| Error::new("INVALID_RESPONSE", error.to_string()))?;
    validate_output(&entities, &relations, entity_types, relation_types)
}

fn strings(value: Option<&Value>) -> Result<Vec<String>, Error> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "relation from/to must be arrays"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "relation endpoint must be string"))
        })
        .collect()
}

fn validate_output(
    entities: &[Entity],
    relations: &[Relation],
    entity_types: &[EntityType],
    relation_types: &[RelationType],
) -> Result<Output, Error> {
    let allowed: BTreeSet<_> = entity_types.iter().map(|item| item.name.as_str()).collect();
    let mut by_id = BTreeMap::new();
    for entity in entities {
        if entity.id.trim().is_empty()
            || entity.name.trim().is_empty()
            || !allowed.contains(entity.r#type.as_str())
            || by_id.insert(entity.id.as_str(), entity).is_some()
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "graph entity is invalid or duplicated",
            ));
        }
        for evidence in &entity.evidence {
            evidence.validate()?;
        }
    }
    for relation in relations {
        let from = by_id
            .get(relation.from.as_str())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "relation from entity is missing"))?;
        let to = by_id
            .get(relation.to.as_str())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "relation to entity is missing"))?;
        let definition = relation_types
            .iter()
            .find(|item| item.name == relation.kind)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "unknown relation kind"))?;
        if (!definition.from.is_empty() && !definition.from.iter().any(|item| item == &from.r#type))
            || (!definition.to.is_empty() && !definition.to.iter().any(|item| item == &to.r#type))
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "relation endpoint type is not allowed",
            ));
        }
        for evidence in &relation.evidence {
            evidence.validate()?;
        }
    }
    Ok(Output {
        entities: entities.to_vec(),
        relations: relations.to_vec(),
    })
}
