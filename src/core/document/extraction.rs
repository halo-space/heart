use std::future::Future;

use futures_core::Stream;
use serde_json::Value;

use crate::Cancellation;
#[cfg(test)]
use crate::components::document::location;
use crate::components::document::{Document, Error, Loader, Relation, Source, graph_types};
use crate::components::message::{Message, Messages};
use crate::components::model::{self, chat::Event as ChatEvent};

/// Built-in loader for the framework's default local-file capability.
#[derive(Clone, Debug, Default)]
pub struct LocalLoader;

impl Loader for LocalLoader {
    async fn load(&self, source: &Source, cancellation: &Cancellation) -> Result<Vec<u8>, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "document load cancelled"));
        }
        let data = std::fs::read(&source.uri)
            .map_err(|error| Error::new("IO_ERROR", error.to_string()))?;
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "document load cancelled"));
        }
        Ok(data)
    }
}

async fn complete_message<C: model::chat::Chat>(
    chat: &C,
    messages: Messages,
    cancellation: &Cancellation,
) -> Result<Message, Error> {
    complete_request(
        chat,
        model::chat::Request {
            messages,
            stream: true,
            ..model::chat::Request::default()
        },
        cancellation,
    )
    .await
}

async fn await_model<F: Future<Output = Result<T, model::Error>>, T>(
    future: F,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|context| {
        cancellation.register(context.waker());
        if cancellation.is_cancelled() {
            return std::task::Poll::Ready(Err(Error::new(
                "CANCELLED",
                "document model call cancelled",
            )));
        }
        let result = future.as_mut().poll(context);
        // Implementations may cancel synchronously in the same poll that
        // produces a result. Never commit that result as a successful batch.
        if cancellation.is_cancelled() {
            return std::task::Poll::Ready(Err(Error::new(
                "CANCELLED",
                "document model call cancelled",
            )));
        }
        result.map(|result| {
            result.map_err(|error| {
                Error::new("MODEL_ERROR", error.message)
                    .with_details(error.details.unwrap_or(error.code))
            })
        })
    })
    .await
}

async fn complete_request<C: model::chat::Chat>(
    chat: &C,
    request: model::chat::Request,
    cancellation: &Cancellation,
) -> Result<Message, Error> {
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "document model call cancelled"));
    }
    let mut stream_future = Box::pin(chat.stream(request));
    let stream = std::future::poll_fn(|context| {
        cancellation.register(context.waker());
        if cancellation.is_cancelled() {
            return std::task::Poll::Ready(None);
        }
        stream_future.as_mut().poll(context).map(Some)
    })
    .await;
    let stream = match stream {
        None => return Err(Error::new("CANCELLED", "document model call cancelled")),
        Some(result) => result.map_err(|error| {
            Error::new("MODEL_ERROR", error.message)
                .with_details(error.details.unwrap_or(error.code))
        })?,
    };
    let mut stream = Box::pin(stream);
    loop {
        let item = std::future::poll_fn(|context| {
            cancellation.register(context.waker());
            if cancellation.is_cancelled() {
                return std::task::Poll::Ready(None);
            }
            stream.as_mut().poll_next(context)
        })
        .await;
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "document model call cancelled"));
        }
        let item =
            item.ok_or_else(|| Error::new("INVALID_RESPONSE", "model did not return Complete"))?;
        match item.map_err(|error| {
            Error::new("MODEL_ERROR", error.message)
                .with_details(error.details.unwrap_or(error.code))
        })? {
            ChatEvent::Delta { .. } => {}
            ChatEvent::Complete {
                mut message, usage, ..
            } => {
                if let Some(usage) = usage {
                    message.usage = Some(usage);
                }
                return Ok(message);
            }
        }
    }
}

fn message_text(message: &Message, key: &str) -> Option<String> {
    message
        .values
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            let parts: Vec<_> = message
                .content
                .iter()
                .filter(|part| part.r#type == "text")
                .filter_map(|part| part.data.get("value").and_then(Value::as_str))
                .collect();
            (!parts.is_empty()).then(|| parts.concat())
        })
}

pub mod raptor {
    use super::{Document, Error, Relation, await_model, complete_message, message_text};
    use crate::components::message::{Message, Messages};
    use crate::components::model::{chat::Chat, embed::Embed};
    use crate::runtime::cancellation::Cancellation;
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};
    use std::collections::HashSet;

    pub type Options = Map<String, Value>;

    #[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
    pub struct Output {
        pub documents: Vec<Document>,
        pub relations: Vec<Relation>,
    }

    pub struct Raptor<C, E> {
        options: Options,
        chat: C,
        embed: E,
    }

    impl<C, E> Raptor<C, E>
    where
        C: Chat,
        E: Embed,
    {
        pub fn new(options: Options, chat: C, embed: E) -> Result<Self, Error> {
            let cluster_size = positive_option(&options, "cluster_size", 4)?;
            let levels = positive_option(&options, "levels", 1)?;
            let threshold = match options.get("threshold") {
                None => 0.72,
                Some(value) => value.as_f64().ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "raptor threshold must be a number")
                })?,
            };
            if cluster_size == 0
                || levels == 0
                || !threshold.is_finite()
                || !(0.0..=1.0).contains(&threshold)
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "raptor cluster_size and levels must be positive and threshold must be in [0, 1]",
                ));
            }
            Ok(Self {
                options,
                chat,
                embed,
            })
        }

        pub async fn run(
            &self,
            documents: &[Document],
            cancellation: &Cancellation,
            next_id: &mut dyn FnMut() -> i64,
        ) -> Result<Output, Error> {
            if cancellation.is_cancelled() {
                return Err(Error::new("CANCELLED", "raptor cancelled"));
            }
            let cluster_size = self
                .options
                .get("cluster_size")
                .and_then(Value::as_u64)
                .unwrap_or(4) as usize;
            let levels = self
                .options
                .get("levels")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize;
            let threshold = self
                .options
                .get("threshold")
                .and_then(Value::as_f64)
                .unwrap_or(0.72);
            if documents.is_empty() {
                return Ok(Output::default());
            }
            if documents.iter().any(|document| document.id.is_none()) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "raptor input documents require id",
                ));
            }
            let mut ids = HashSet::new();
            if documents
                .iter()
                .any(|document| !ids.insert(document.id.expect("validated above")))
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "raptor input document IDs must be unique",
                ));
            }
            let mut current = documents.to_vec();
            let mut output = Output::default();
            let mut dimension = None;
            for level in 1..=levels {
                let mut next = Vec::new();
                let mut embedded = Vec::with_capacity(current.len());
                for document in &current {
                    if cancellation.is_cancelled() {
                        return Err(Error::new("CANCELLED", "raptor cancelled"));
                    }
                    let message = await_model(
                        self.embed.embed(crate::components::model::embed::Request {
                            input: document.content.clone(),
                            dimensions: None,
                            options: Map::new(),
                        }),
                        cancellation,
                    )
                    .await?;
                    let vector = message
                        .values
                        .get("vector")
                        .and_then(Value::as_array)
                        .ok_or_else(|| {
                            Error::new("INVALID_RESPONSE", "embed response does not contain vector")
                        })?
                        .iter()
                        .map(|value| {
                            value
                                .as_f64()
                                .and_then(|value| value.is_finite().then_some(value as f32))
                        })
                        .collect::<Option<Vec<_>>>()
                        .filter(|vector| {
                            !vector.is_empty() && vector.iter().all(|value| value.is_finite())
                        })
                        .ok_or_else(|| {
                            Error::new("INVALID_RESPONSE", "embed response vector is invalid")
                        })?;
                    if dimension.is_some_and(|dimension| dimension != vector.len()) {
                        return Err(Error::new(
                            "INVALID_RESPONSE",
                            "embed response dimensions changed",
                        ));
                    }
                    dimension = Some(vector.len());
                    embedded.push((document.clone(), vector));
                }
                for group in cluster_documents(&embedded, cluster_size, threshold) {
                    if cancellation.is_cancelled() {
                        return Err(Error::new("CANCELLED", "raptor cancelled"));
                    }
                    let prompt = serde_json::to_string(
                        &group
                            .iter()
                            .map(|(document, _)| document.content.as_str())
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()))?;
                    let mut input = Message::new(crate::components::message::Role::User);
                    input
                        .content
                        .push(crate::components::model::content::Part {
                            r#type: "text".into(),
                            data: serde_json::json!({"value": format!("Summarize these documents: {prompt}")}),
                        });
                    let response =
                        complete_message(&self.chat, Messages::new([input]), cancellation).await?;
                    let summary = message_text(&response, "summary")
                        .filter(|text| !text.trim().is_empty())
                        .ok_or_else(|| Error::new("INVALID_RESPONSE", "summary is missing"))?;
                    let id = next_id();
                    if id <= 0 || !ids.insert(id) {
                        return Err(Error::new(
                            "INVALID_ARGUMENTS",
                            "raptor next_id must return a fresh positive ID",
                        ));
                    }
                    if cancellation.is_cancelled() {
                        return Err(Error::new("CANCELLED", "raptor cancelled"));
                    }
                    let summary_document = Document {
                        id: Some(id),
                        doc_id: None,
                        content: summary,
                        metadata: Map::from_iter([(
                            "raptor".into(),
                            serde_json::json!({ "level": level, "method": "raptor" }),
                        )]),
                    };
                    for (source, _) in &group {
                        output.relations.push(Relation {
                            from: source.id.expect("validated above"),
                            to: id,
                            kind: "summary".into(),
                        });
                    }
                    next.push(summary_document.clone());
                    output.documents.push(summary_document);
                }
                current = next;
            }
            if cancellation.is_cancelled() {
                return Err(Error::new("CANCELLED", "raptor cancelled"));
            }
            Ok(output)
        }
    }

    fn positive_option(options: &Options, name: &str, default: u64) -> Result<u64, Error> {
        match options.get(name) {
            None => Ok(default),
            Some(value) => value
                .as_u64()
                .filter(|value| *value > 0 && usize::try_from(*value).is_ok())
                .ok_or_else(|| {
                    Error::new(
                        "INVALID_ARGUMENTS",
                        "raptor option must be a positive integer",
                    )
                    .with_details(name)
                }),
        }
    }

    fn cluster_documents(
        documents: &[(Document, Vec<f32>)],
        cluster_size: usize,
        threshold: f64,
    ) -> Vec<Vec<(Document, Vec<f32>)>> {
        let mut used = vec![false; documents.len()];
        let mut clusters = Vec::new();
        for seed in 0..documents.len() {
            if used[seed] {
                continue;
            }
            used[seed] = true;
            let mut cluster = vec![documents[seed].clone()];
            for candidate in (seed + 1)..documents.len() {
                if used[candidate] || cluster.len() >= cluster_size {
                    continue;
                }
                if cosine(&documents[seed].1, &documents[candidate].1) >= threshold {
                    used[candidate] = true;
                    cluster.push(documents[candidate].clone());
                }
            }
            clusters.push(cluster);
        }
        clusters
    }

    fn cosine(left: &[f32], right: &[f32]) -> f64 {
        if left.len() != right.len() || left.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0_f64;
        let mut left_norm = 0.0_f64;
        let mut right_norm = 0.0_f64;
        for (left, right) in left.iter().zip(right) {
            let left = f64::from(*left);
            let right = f64::from(*right);
            dot += left * right;
            left_norm += left * left;
            right_norm += right * right;
        }
        if left_norm == 0.0 || right_norm == 0.0 {
            0.0
        } else {
            dot / (left_norm.sqrt() * right_norm.sqrt())
        }
    }
}

pub mod graph_impl {
    use super::graph_types::Evidence;
    use super::{Document, Error, complete_request};
    use crate::components::message::{Message, Messages};
    use crate::components::model::chat::Chat;
    use crate::runtime::cancellation::Cancellation;
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
    struct EntityType {
        name: String,
        description: String,
    }
    #[derive(Clone, Debug)]
    struct RelationType {
        name: String,
        description: String,
        from: Vec<String>,
        to: Vec<String>,
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
                if name.trim().is_empty()
                    || description.is_empty()
                    || !names.insert(name.to_owned())
                {
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
                let object = value.as_object().ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "relation type must be object")
                })?;
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
            input.content.push(crate::components::model::content::Part {
                r#type: "text".into(),
                data: serde_json::json!({"value": format!("Extract the graph from these documents using the declared entity and relation types. Return a JSON object with entities and relations arrays. Entity fields are id, type, name, optional attrs and evidence. Relation fields are from, to, kind, optional attrs and evidence. Do not invent evidence or identifiers absent from the source. Input: {payload}")}),
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
            let values = if message.values.contains_key("entities")
                || message.values.contains_key("relations")
            {
                message.values
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
            let relations: Vec<Relation> =
                serde_json::from_value(values.get("relations").cloned().ok_or_else(|| {
                    Error::new("INVALID_RESPONSE", "graph relations are missing")
                })?)
                .map_err(|error| Error::new("INVALID_RESPONSE", error.to_string()))?;
            validate_output(
                &entities,
                &relations,
                &self.entity_types,
                &self.relation_types,
            )
        }
    }

    fn strings(value: Option<&Value>) -> Result<Vec<String>, Error> {
        value
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "relation from/to must be arrays"))?
            .iter()
            .map(|item| {
                item.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "relation endpoint must be string")
                })
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
            let from = by_id.get(relation.from.as_str()).ok_or_else(|| {
                Error::new("INVALID_ARGUMENTS", "relation from entity is missing")
            })?;
            let to = by_id
                .get(relation.to.as_str())
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "relation to entity is missing"))?;
            let definition = relation_types
                .iter()
                .find(|item| item.name == relation.kind)
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "unknown relation kind"))?;
            if (!definition.from.is_empty()
                && !definition.from.iter().any(|item| item == &from.r#type))
                || (!definition.to.is_empty()
                    && !definition.to.iter().any(|item| item == &to.r#type))
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
}

/// Graph extraction types and evidence share one public module path.
pub mod graph {
    pub use super::graph_impl::{Entity, Graph, Options, Output, Relation};
    pub use super::graph_types::{Evidence, Method};
}

#[cfg(test)]
mod tests {
    use futures_core::Stream;
    use futures_executor::block_on;
    use serde_json::json;
    use std::future::Future;
    use std::{
        collections::VecDeque,
        pin::Pin,
        task::{Context, Poll},
    };

    use super::{
        Document, Loader, LocalLoader, Source,
        graph::{Evidence, Graph, Method},
        location::{Location, Range},
        raptor::Raptor,
    };
    use crate::components::{
        event::Event,
        message::{Message, Role},
        model::{self, chat::Chat, embed::Embed},
    };
    use crate::runtime::cancellation::Cancellation;

    struct FakeStream(VecDeque<Result<Event, model::Error>>);
    impl Stream for FakeStream {
        type Item = Result<Event, model::Error>;
        fn poll_next(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    #[derive(Clone)]
    struct FakeChat {
        response: Message,
    }
    impl Chat for FakeChat {
        type Stream = FakeStream;
        async fn stream(
            &self,
            _request: model::chat::Request,
        ) -> Result<Self::Stream, model::Error> {
            Ok(FakeStream(VecDeque::from([Ok(Event::Complete {
                message: self.response.clone(),
                usage: None,
                finish_reason: None,
            })])))
        }
    }

    struct CompleteThenPending(Option<Message>);
    impl Stream for CompleteThenPending {
        type Item = Result<Event, model::Error>;
        fn poll_next(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            match self.0.take() {
                Some(message) => Poll::Ready(Some(Ok(Event::Complete {
                    message,
                    usage: None,
                    finish_reason: None,
                }))),
                None => Poll::Pending,
            }
        }
    }

    struct NonEndingChat(Option<Message>);
    impl Chat for NonEndingChat {
        type Stream = CompleteThenPending;
        async fn stream(
            &self,
            _request: model::chat::Request,
        ) -> Result<Self::Stream, model::Error> {
            Ok(CompleteThenPending(self.0.clone()))
        }
    }

    struct PendingCreateChat;
    impl Chat for PendingCreateChat {
        type Stream = CompleteThenPending;
        async fn stream(
            &self,
            _request: model::chat::Request,
        ) -> Result<Self::Stream, model::Error> {
            std::future::pending().await
        }
    }

    struct FakeEmbed;
    impl Embed for FakeEmbed {
        async fn embed(&self, _request: model::embed::Request) -> Result<Message, model::Error> {
            Ok(model::embed::message(vec![0.1, 0.2], "fake", "embed", Some(2)).unwrap())
        }
    }

    fn summary_chat() -> FakeChat {
        let mut response = Message::new(Role::Assistant);
        response.values.insert("summary".into(), json!("摘要"));
        FakeChat { response }
    }

    fn graph_options() -> super::graph::Options {
        serde_json::from_value(json!({
            "entity_types": [{"name": "person", "description": "人物"}],
            "relation_types": []
        }))
        .unwrap()
    }

    #[test]
    fn raptor_rejects_malformed_options_instead_of_using_defaults() {
        for options in [
            json!({"threshold": "wrong"}),
            json!({"threshold": null}),
            json!({"threshold": -0.1}),
            json!({"threshold": 1.1}),
            json!({"cluster_size": -1}),
            json!({"cluster_size": 0}),
            json!({"levels": -1}),
            json!({"levels": 1.5}),
            json!({"levels": "2"}),
        ] {
            let result = Raptor::new(
                serde_json::from_value(options).unwrap(),
                summary_chat(),
                FakeEmbed,
            );
            assert_eq!(result.err().unwrap().code, "INVALID_ARGUMENTS");
        }
        assert!(Raptor::new(Default::default(), summary_chat(), FakeEmbed).is_ok());
    }

    struct PendingEmbed;
    impl Embed for PendingEmbed {
        async fn embed(&self, _request: model::embed::Request) -> Result<Message, model::Error> {
            std::future::pending().await
        }
    }

    #[test]
    fn cancellation_unblocks_raptor_while_embed_is_pending() {
        let cancellation = Cancellation::new();
        let raptor = Raptor::new(Default::default(), summary_chat(), PendingEmbed).unwrap();
        let documents = [Document {
            id: Some(1),
            content: "text".into(),
            ..Document::default()
        }];
        let mut generated = 0;
        let mut next_id = || {
            generated += 1;
            generated
        };
        let mut future = Box::pin(raptor.run(&documents, &cancellation, &mut next_id));
        let mut context = Context::from_waker(std::task::Waker::noop());
        assert!(future.as_mut().poll(&mut context).is_pending());
        cancellation.cancel();
        let Poll::Ready(result) = future.as_mut().poll(&mut context) else {
            panic!("Embed wait must stop on cancellation");
        };
        assert_eq!(result.unwrap_err().code, "CANCELLED");
        drop(future);
        assert_eq!(generated, 0);
        assert_eq!(
            block_on(raptor.run(&[], &cancellation, &mut || 100))
                .unwrap_err()
                .code,
            "CANCELLED"
        );
    }

    struct ChangingDimensions;
    impl Embed for ChangingDimensions {
        async fn embed(&self, request: model::embed::Request) -> Result<Message, model::Error> {
            let vector = if request.input == "first" {
                vec![1.0]
            } else {
                vec![1.0, 0.0]
            };
            Ok(model::embed::message(vector, "fake", "embed", None).unwrap())
        }
    }

    #[test]
    fn raptor_rejects_dimension_changes_before_summary_generation() {
        let raptor = Raptor::new(Default::default(), summary_chat(), ChangingDimensions).unwrap();
        let documents = [
            Document {
                id: Some(1),
                content: "first".into(),
                ..Document::default()
            },
            Document {
                id: Some(2),
                content: "second".into(),
                ..Document::default()
            },
        ];
        let mut generated = 0;
        let result = block_on(raptor.run(&documents, &Cancellation::new(), &mut || {
            generated += 1;
            generated
        }));
        assert_eq!(result.unwrap_err().code, "INVALID_RESPONSE");
        assert_eq!(generated, 0);
    }

    struct JsonChat {
        text: String,
    }
    impl Chat for JsonChat {
        type Stream = FakeStream;
        async fn stream(
            &self,
            request: model::chat::Request,
        ) -> Result<Self::Stream, model::Error> {
            assert_eq!(
                request.response_format,
                Some(model::chat::ResponseFormat::JsonObject)
            );
            request.validate()?;
            let mut response = Message::new(Role::Assistant);
            response.content.push(model::content::Part {
                r#type: "think".into(),
                data: json!({"value": "not JSON"}),
            });
            // A complete text response can contain more than one text part.
            response.content.push(model::content::Part {
                r#type: "text".into(),
                data: json!({"value": &self.text[..1]}),
            });
            response.content.push(model::content::Part {
                r#type: "text".into(),
                data: json!({"value": &self.text[1..]}),
            });
            Ok(FakeStream(VecDeque::from([Ok(Event::Complete {
                message: response,
                usage: None,
                finish_reason: None,
            })])))
        }
    }

    #[test]
    fn graph_requests_json_and_decodes_only_complete_text() {
        let graph = Graph::new(graph_options(), JsonChat {
            text: json!({"entities": [{"id": "p1", "type": "person", "name": "张三"}], "relations": []}).to_string(),
        }).unwrap();
        let output = block_on(graph.run(&[], &Cancellation::new())).unwrap();
        assert_eq!(output.entities[0].name, "张三");
        for text in [
            "not JSON",
            "[]",
            "{\"entities\": []}",
            "{\"entities\": [], \"relations\": 1}",
        ] {
            let graph = Graph::new(graph_options(), JsonChat { text: text.into() }).unwrap();
            assert_eq!(
                block_on(graph.run(&[], &Cancellation::new()))
                    .unwrap_err()
                    .code,
                "INVALID_RESPONSE"
            );
        }
    }

    #[test]
    fn graph_preserves_declared_type_names_without_trimming() {
        let mut response = Message::new(Role::Assistant);
        response.values.insert(
            "entities".into(),
            json!([{"id": "p1", "type": " person ", "name": "张三"}]),
        );
        response.values.insert("relations".into(), json!([]));
        let graph = Graph::new(
            serde_json::from_value(json!({
                "entity_types": [{"name": " person ", "description": "人物"}], "relation_types": []
            }))
            .unwrap(),
            FakeChat { response },
        )
        .unwrap();
        assert_eq!(
            block_on(graph.run(&[], &Cancellation::new()))
                .unwrap()
                .entities[0]
                .r#type,
            " person "
        );
    }

    #[test]
    fn location_validates_grouped_ranges_and_coordinates() {
        let location = Location {
            page: Some(Range { from: 2, to: 3 }),
            paragraph: Some(Range { from: 8, to: 2 }),
            r#char: Some(Range { from: 12, to: 18 }),
            x: Some(Range { from: 0.1, to: 0.8 }),
            y: Some(Range { from: 0.2, to: 0.9 }),
        };
        assert!(location.validate().is_ok());
    }

    #[test]
    fn evidence_requires_content_for_method_and_valid_location() {
        let evidence = Evidence {
            content: Some("张三在甲公司工作".into()),
            method: Some(Method::Text),
            location: Some(Location {
                page: Some(Range { from: 3, to: 3 }),
                paragraph: Some(Range { from: 2, to: 2 }),
                ..Location::default()
            }),
            ..Evidence::default()
        };
        assert!(evidence.validate().is_ok());
        let invalid = Evidence {
            method: Some(Method::Ocr),
            ..Evidence::default()
        };
        assert_eq!(invalid.validate().unwrap_err().code, "INVALID_ARGUMENTS");
    }

    #[test]
    fn evidence_reports_the_invalid_coordinate_endpoint() {
        let evidence = Evidence {
            content: Some("依据".into()),
            location: Some(Location {
                x: Some(Range { from: 0.1, to: 1.2 }),
                y: Some(Range { from: 0.2, to: 0.8 }),
                ..Location::default()
            }),
            ..Evidence::default()
        };
        assert_eq!(
            evidence.validate().unwrap_err().details.as_deref(),
            Some("location.x.to")
        );
    }

    #[test]
    fn document_keeps_optional_ids_and_metadata() {
        let document = Document {
            content: "hello".into(),
            metadata: serde_json::Map::from_iter([("layout".into(), json!({"role": "paragraph"}))]),
            ..Document::default()
        };
        let value = serde_json::to_value(document).unwrap();
        assert!(value.get("id").is_none());
        assert_eq!(value["metadata"]["layout"]["role"], "paragraph");
    }

    #[test]
    fn local_loader_reads_one_complete_file_and_honors_cancellation() {
        let cancellation = Cancellation::new();
        cancellation.cancel();
        let result = block_on(LocalLoader.load(
            &Source {
                uri: "/does/not/exist".into(),
            },
            &cancellation,
        ));
        assert_eq!(result.unwrap_err().code, "CANCELLED");
    }

    #[test]
    fn document_model_call_stops_on_complete_without_waiting_for_stream_end() {
        let response = Message::new(Role::Assistant);
        let result = block_on(super::complete_message(
            &NonEndingChat(Some(response.clone())),
            crate::Messages::default(),
            &Cancellation::new(),
        ))
        .unwrap();
        assert_eq!(result, response);
    }

    struct UsageChat;
    impl Chat for UsageChat {
        type Stream = FakeStream;
        async fn stream(
            &self,
            _request: model::chat::Request,
        ) -> Result<Self::Stream, model::Error> {
            Ok(FakeStream(VecDeque::from([Ok(Event::Complete {
                message: Message::new(Role::Assistant),
                usage: Some(model::token::Usage {
                    input: Some(7),
                    output: Some(3),
                    total: Some(10),
                    ..Default::default()
                }),
                finish_reason: None,
            })])))
        }
    }

    #[test]
    fn document_complete_preserves_event_usage_without_estimating_or_aggregating() {
        let message = block_on(super::complete_message(
            &UsageChat,
            crate::Messages::default(),
            &Cancellation::new(),
        ))
        .unwrap();
        assert_eq!(message.usage.unwrap().total, Some(10));
        // No second usage or Runtime state is introduced by this conversion.
    }

    #[test]
    fn cancellation_unblocks_a_pending_document_model_stream() {
        let cancellation = Cancellation::new();
        let chat = NonEndingChat(None);
        let mut future = Box::pin(super::complete_message(
            &chat,
            crate::Messages::default(),
            &cancellation,
        ));
        let waker = std::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        cancellation.cancel();
        let Poll::Ready(result) = future.as_mut().poll(&mut context) else {
            panic!("cancelled document stream should be ready");
        };
        assert_eq!(result.unwrap_err().code, "CANCELLED");
    }

    #[test]
    fn cancellation_unblocks_document_model_stream_creation() {
        let cancellation = Cancellation::new();
        let chat = PendingCreateChat;
        let mut future = Box::pin(super::complete_message(
            &chat,
            crate::Messages::default(),
            &cancellation,
        ));
        let waker = std::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        cancellation.cancel();
        let Poll::Ready(result) = future.as_mut().poll(&mut context) else {
            panic!("cancelled model creation should be ready");
        };
        assert_eq!(result.unwrap_err().code, "CANCELLED");
    }

    #[test]
    fn graph_validates_config_and_returns_complete_entities_and_relations() {
        let mut response = Message::new(Role::Assistant);
        response.values.insert(
            "entities".into(),
            json!([
                {"id": "person-1", "type": "person", "name": "张三"},
                {"id": "company-1", "type": "company", "name": "甲公司"}
            ]),
        );
        response.values.insert(
            "relations".into(),
            json!([
                {"from": "person-1", "to": "company-1", "kind": "works_for"}
            ]),
        );
        let graph = Graph::new(serde_json::from_value(json!({
            "entity_types": [
                {"name": "person", "description": "人物"},
                {"name": "company", "description": "组织"}
            ],
            "relation_types": [
                {"name": "works_for", "description": "任职", "from": ["person"], "to": ["company"]}
            ]
        })).unwrap(), FakeChat { response }).unwrap();
        let output = block_on(graph.run(
            &[Document {
                id: Some(1),
                content: "张三在甲公司工作".into(),
                ..Document::default()
            }],
            &Cancellation::new(),
        ))
        .unwrap();
        assert_eq!(output.entities.len(), 2);
        assert_eq!(output.relations[0].kind, "works_for");
    }

    #[test]
    fn raptor_returns_new_summary_documents_and_source_relations() {
        let mut response = Message::new(Role::Assistant);
        response
            .values
            .insert("summary".into(), json!("两段内容的摘要"));
        let raptor = Raptor::new(
            serde_json::from_value(json!({"cluster_size": 2, "levels": 1})).unwrap(),
            FakeChat { response },
            FakeEmbed,
        )
        .unwrap();
        let documents = vec![
            Document {
                id: Some(1),
                content: "第一段".into(),
                ..Document::default()
            },
            Document {
                id: Some(2),
                content: "第二段".into(),
                ..Document::default()
            },
        ];
        let mut next = 100;
        let output = block_on(raptor.run(&documents, &Cancellation::new(), &mut || {
            next += 1;
            next
        }))
        .unwrap();
        assert_eq!(output.documents.len(), 1);
        assert_eq!(output.documents[0].id, Some(101));
        assert_eq!(output.relations.len(), 2);
        assert!(
            output
                .relations
                .iter()
                .all(|relation| relation.kind == "summary")
        );
    }

    struct DistinctEmbed;
    impl Embed for DistinctEmbed {
        async fn embed(&self, request: model::embed::Request) -> Result<Message, model::Error> {
            let vector = if request.input.contains('一') {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            };
            Ok(model::embed::message(vector, "fake", "embed", Some(2)).unwrap())
        }
    }

    #[test]
    fn raptor_uses_embedding_similarity_instead_of_only_input_order() {
        let mut response = Message::new(Role::Assistant);
        response.values.insert("summary".into(), json!("摘要"));
        let raptor = Raptor::new(
            serde_json::from_value(json!({
                "cluster_size": 2,
                "levels": 1,
                "threshold": 0.9
            }))
            .unwrap(),
            FakeChat { response },
            DistinctEmbed,
        )
        .unwrap();
        let documents = vec![
            Document {
                id: Some(1),
                content: "第一段".into(),
                ..Document::default()
            },
            Document {
                id: Some(2),
                content: "第二段".into(),
                ..Document::default()
            },
        ];
        let mut next = 100;
        let output = block_on(raptor.run(&documents, &Cancellation::new(), &mut || {
            next += 1;
            next
        }))
        .unwrap();
        assert_eq!(output.documents.len(), 2);
        assert_eq!(output.relations.len(), 2);
        assert!(output.relations.iter().all(|relation| relation.to > 100));
    }
}
