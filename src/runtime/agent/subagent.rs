//! SubAgent assembly plus private child execution and usage settlement.
use crate::runtime::agent::node::status::Status;
use crate::runtime::agent::node::{State, Step};
use crate::{
    Cancellation, Error, Message,
    components::agent::{Agent, Input},
    components::node::View,
    event::Event,
    message::Role,
    model::token::{InputDetails, OutputDetails, Usage},
    tool::Call,
};
use futures_util::task::AtomicWaker;
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

type ErasedStream =
    Pin<Box<dyn futures_core::Stream<Item = Result<Event, Error>> + Send + 'static>>;
type RunFuture = Pin<Box<dyn Future<Output = Result<ErasedStream, Error>> + Send + 'static>>;

struct SerializedStream {
    inner: ErasedStream,
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

impl futures_core::Stream for SerializedStream {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

trait ErasedAgent: Send + Sync {
    fn run(&self, input: Input, cancellation: Cancellation, view: View) -> RunFuture;
}

mod adapter {
    use std::sync::Arc;

    pub(super) struct Agent<A>(pub(super) Arc<A>);
}

impl<A> ErasedAgent for adapter::Agent<A>
where
    A: Agent + 'static,
{
    fn run(&self, input: Input, cancellation: Cancellation, view: View) -> RunFuture {
        let agent = Arc::clone(&self.0);
        Box::pin(async move {
            // Parent run owns outward mode filtering for delegated events.
            let stream = agent
                .run_with_view(input, true, cancellation, &view)
                .await?;
            Ok(Box::pin(stream) as ErasedStream)
        })
    }
}

/// One explicitly registered Agent that another Agent may delegate to.
///
/// SubAgents are distinct from the internal Agent tool dispatcher: the
/// dispatcher contains Tool/MCP/Skill entries, while this retains a complete Agent
/// run boundary and its event stream.
#[derive(Clone)]
pub struct SubAgent {
    name: String,
    description: String,
    runner: Arc<dyn ErasedAgent>,
    serial: Arc<tokio::sync::Mutex<()>>,
}

impl SubAgent {
    pub fn new<A>(name: impl Into<String>, agent: A) -> Result<Self, Error>
    where
        A: Agent + 'static,
    {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "subagent name must not be empty",
            ));
        }
        Ok(Self {
            name,
            description: String::new(),
            runner: Arc::new(adapter::Agent(Arc::new(agent))),
            serial: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description_text(&self) -> &str {
        &self.description
    }

    pub(crate) fn definition(subagents: &[SubAgent]) -> crate::components::tool::ToolDefinition {
        Self::definition_named(subagents, "agent_spawn", "Delegate a child task")
    }

    pub(crate) fn definition_named(
        subagents: &[SubAgent],
        tool_name: &str,
        operation: &str,
    ) -> crate::components::tool::ToolDefinition {
        let member_names = subagents
            .iter()
            .map(|subagent| subagent.name.clone())
            .collect::<Vec<_>>();
        Self::definition_for_names(subagents, tool_name, operation, &member_names)
    }

    pub(crate) fn definition_for_names(
        subagents: &[SubAgent],
        tool_name: &str,
        operation: &str,
        names: &[String],
    ) -> crate::components::tool::ToolDefinition {
        let members = subagents
            .iter()
            .filter(|subagent| names.iter().any(|name| name == subagent.name()))
            .map(|subagent| {
                if subagent.description.is_empty() {
                    format!("- {}", subagent.name)
                } else {
                    format!("- {}: {}", subagent.name, subagent.description)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let input_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "metadata": {"type": "object", "additionalProperties": true},
                "payload": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["text", "audio", "image", "file", "json"]},
                            "data": {
                                "type": "object",
                                "properties": {"value": {}},
                                "required": ["value"],
                                "additionalProperties": true
                            },
                            "mime_type": {"type": ["string", "null"]}
                        },
                        "required": ["type", "data"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["payload"],
            "additionalProperties": false
        });
        crate::components::tool::ToolDefinition {
            name: tool_name.into(),
            description: format!(
                "{operation} to one of the registered members. The member input must be an agent::Input object. Registered members:\n{members}"
            ),
            parameters: serde_json::from_value(serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": names},
                    "input": input_schema
                },
                "required": ["name", "input"],
                "additionalProperties": false
            }))
            .expect("static agent_spawn schema is an object"),
        }
    }

    pub(crate) fn has_name(&self, name: &str) -> bool {
        self.name == name
    }

    pub(crate) fn run_with_view(
        &self,
        input: Input,
        cancellation: Cancellation,
        view: View,
    ) -> RunFuture {
        let runner = Arc::clone(&self.runner);
        let serial = Arc::clone(&self.serial);
        Box::pin(async move {
            let guard = serial.lock_owned().await;
            let stream = runner.run(input, cancellation, view).await?;
            Ok(Box::pin(SerializedStream {
                inner: stream,
                _guard: guard,
            }) as ErasedStream)
        })
    }
}

impl std::fmt::Debug for SubAgent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubAgent")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

#[doc(hidden)]
pub mod relay;
#[doc(hidden)]
pub mod root;
mod run;

#[doc(hidden)]
pub use relay::{Observation, Observer, Relay, cancel_ready};
#[doc(hidden)]
pub use root::{Progress, Root};
pub(crate) use root::{
    cancel_pending, complete, message, observe, prepare, resume_calls, source, sync_waits,
};
#[cfg(test)]
pub(crate) use run::result_message;
pub(crate) use run::{Failure, inherit_identity, run};

#[cfg(test)]
mod result_message_tests {
    use super::result_message;
    use crate::{Message, message::Role, tool::Call};
    use serde_json::json;

    #[test]
    fn child_private_think_is_not_forwarded_as_a_function_result() {
        let call: Call = serde_json::from_value(json!({
            "call_id": "call_child",
            "name": "agent_spawn",
            "arguments": {}
        }))
        .unwrap();
        let mut message = Message::new(Role::Assistant);
        message.content.extend([
            crate::model::content::part::Part {
                r#type: "think".into(),
                data: json!({"value":"private reasoning"})
                    .as_object()
                    .unwrap()
                    .clone(),
            },
            crate::model::content::part::Part {
                r#type: "text".into(),
                data: json!({"value":"child result"}).as_object().unwrap().clone(),
            },
        ]);

        let result = result_message(message, &call);

        assert_eq!(result.role, Role::Function);
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].r#type, "text");
        assert_eq!(result.content[0].data["value"], "child result");
        assert_eq!(result.metadata["call_id"], "call_child");
    }
}

#[cfg(test)]
mod observation_tests {
    use super::*;
    use crate::runtime::agent::node::status::Status;
    use crate::runtime::agent::node::{Attempt, Execution, State};
    use serde_json::json;
    use std::collections::BTreeMap;

    fn running_node(exec_id: i64) -> State {
        State {
            status: Status::Running,
            execution: Some(Execution {
                exec_id,
                attempts: vec![Attempt {
                    idx: 0,
                    status: Status::Running,
                    created_time: Some(1),
                    finished_time: None,
                    usage: None,
                    finish_reason: None,
                    steps: Vec::new(),
                    messages: crate::Messages::default(),
                    completion: None,
                    nodes: BTreeMap::new(),
                    agent: None,
                    error: None,
                }],
                messages: crate::Messages::default(),
                usage: None,
                error: None,
            }),
            ..Default::default()
        }
    }

    fn observation(
        event: Option<Event>,
        scene: Option<Value>,
        message: Option<Message>,
    ) -> Observation {
        Observation {
            event,
            feedback: false,
            source: ("reviewer".into(), 77),
            scene: scene.map(Arc::new),
            usage: None,
            message,
        }
    }

    #[test]
    fn intermediate_child_complete_does_not_complete_its_parent_node() {
        let mut nodes = BTreeMap::from([("reviewer".into(), running_node(77))]);
        let mut event_message = Message::new(Role::Assistant);
        event_message
            .values
            .insert("progress".into(), json!("partial"));
        let mut result = Message::function(json!({"progress":"partial"}));
        result
            .metadata
            .insert("call_id".into(), json!("child_call"));
        let observation = observation(
            Some(Event::Complete {
                message: event_message,
                usage: None,
                finish_reason: Some("tool".into()),
            }),
            None,
            Some(result),
        );

        observe(&mut nodes, &observation).unwrap();

        let child = &nodes["reviewer"];
        assert_eq!(child.status, Status::Running);
        assert!(child.messages.is_none());
        assert_eq!(
            child.execution.as_ref().unwrap().attempts[0].status,
            Status::Running
        );
        assert!(child.execution.as_ref().unwrap().messages.is_empty());
    }

    #[test]
    fn only_a_terminal_child_scene_commits_its_result_message() {
        let mut nodes = BTreeMap::from([("reviewer".into(), running_node(77))]);
        let message = Message::function(json!({"unix_seconds":1234}));
        let observation = observation(
            None,
            Some(json!({
                "type":"agent",
                "status":"succeeded",
                "usage":{"input":4,"output":2,"total":6,"details":{}}
            })),
            Some(message.clone()),
        );

        observe(&mut nodes, &observation).unwrap();

        let child = &nodes["reviewer"];
        assert_eq!(child.status, Status::Succeeded);
        assert_eq!(
            child.messages.as_ref().unwrap().as_slice(),
            std::slice::from_ref(&message)
        );
        let execution = child.execution.as_ref().unwrap();
        assert_eq!(execution.messages.as_slice(), &[message]);
        assert_eq!(
            execution.attempts[0].agent.as_ref().unwrap()["status"],
            "succeeded"
        );
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(6));
        assert!(execution.attempts[0].finished_time.is_some());
    }

    #[test]
    fn nested_child_wait_locators_are_prefixed_once_in_the_saved_order() {
        let nodes = BTreeMap::from([("reviewer".into(), running_node(77))]);
        let scene = json!({
            "type":"agent",
            "status":"waiting",
            "waits":[
                {"nodes":[],"step":"first"},
                {"nodes":[{"node_id":"inner","exec_id":88,"idx":0}],"step":"second"}
            ]
        });
        let observation = observation(None, Some(scene), None);
        let mut waits = Vec::new();

        sync_waits(&nodes, &mut waits, &observation).unwrap();
        sync_waits(&nodes, &mut waits, &observation).unwrap();

        assert_eq!(waits.len(), 2);
        assert_eq!(waits[0].step, "first");
        assert_eq!(waits[0].nodes[0].node_id, "reviewer");
        assert_eq!(waits[0].nodes[0].exec_id, 77);
        assert_eq!(waits[0].nodes[0].idx, 0);
        assert_eq!(waits[1].step, "second");
        assert_eq!(waits[1].nodes[0].node_id, "reviewer");
        assert_eq!(waits[1].nodes[1].node_id, "inner");
    }
}

#[cfg(test)]
mod relay_tests {
    use super::*;
    use crate::{Message, message::Role, runtime::agent::wait::Node};
    use futures_util::future::poll_fn;
    use serde_json::json;

    #[tokio::test]
    async fn child_event_waits_for_parent_ack_and_keeps_its_execution_source() {
        let relay = Relay::default();
        let observer = relay.observer(
            Some(501),
            Node {
                node_id: "reviewer".into(),
                exec_id: 77,
                idx: 0,
            },
        );
        let sender = tokio::spawn(async move {
            let mut message = Message::new(Role::Assistant);
            message.values.insert("phase".into(), json!("child"));
            observer
                .emit(Some(Event::Delta { message }), None, None, None, false)
                .await;
        });

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), async {
                let _ = sender.is_finished();
                while !sender.is_finished() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_err(),
            "child producer stays backpressured until its event is consumed"
        );
        let observation = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            poll_fn(|context| relay.poll_event(context)),
        )
        .await
        .expect("parent must receive the queued child event");
        let Some(Event::Delta { message }) = observation.event else {
            panic!("expected the original child Delta")
        };
        assert_eq!(message.metadata["chat_id"], 501);
        assert_eq!(message.metadata["node_id"], "reviewer");
        assert_eq!(message.metadata["exec_id"], 77);
        assert_eq!(message.values["phase"], "child");
        tokio::time::timeout(std::time::Duration::from_secs(1), sender)
            .await
            .expect("consumption acknowledges the event")
            .expect("child emitter task succeeds");
    }
}

#[cfg(test)]
mod definition_tests {
    use super::*;
    use futures_util::stream::{self, Empty};

    struct EmptyAgent;

    impl Agent for EmptyAgent {
        type Stream = Empty<Result<Event, Error>>;

        async fn run(
            &self,
            _input: Input,
            _stream: bool,
            _cancellation: Cancellation,
        ) -> Result<Self::Stream, Error> {
            Ok(stream::empty())
        }

        async fn reply(&self, _input: Input, _cancellation: Cancellation) -> Result<(), Error> {
            Ok(())
        }
    }

    #[test]
    fn delegation_definition_limits_model_targets_to_registered_members() {
        let subagents = vec![
            SubAgent::new("researcher", EmptyAgent)
                .unwrap()
                .description("researches facts"),
            SubAgent::new("reviewer", EmptyAgent)
                .unwrap()
                .description("reviews results"),
            SubAgent::new("coder", EmptyAgent)
                .unwrap()
                .description("writes code"),
        ];
        let definition = SubAgent::definition_named(&subagents, "agent_spawn", "Delegate");
        let names = &definition.parameters["properties"]["name"]["enum"];
        assert_eq!(
            names,
            &serde_json::json!(["researcher", "reviewer", "coder"])
        );
        definition.validate().unwrap();
    }
}
