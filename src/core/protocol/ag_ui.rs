use std::collections::{HashSet, VecDeque};
use std::pin::Pin;

use futures_core::Stream;
use futures_util::{StreamExt, stream};
use serde_json::Value;

use crate::Error;
use crate::components::event::Event as HeartEvent;
use crate::components::message::{Message, Role};
use crate::components::model::content::part::Part;
pub use crate::components::protocol::ag_ui::{Context, Event};

type InputStream<'a> = Pin<Box<dyn Stream<Item = Result<HeartEvent, Error>> + 'a>>;

/// Convert one Heart Event stream to an AG-UI stream.
///
/// The outer HTTP or transport layer invokes this when it selects AG-UI.
/// The adapter emits RUN_STARTED before consuming Heart events and emits
/// either RUN_FINISHED or RUN_ERROR when the source ends.
pub fn adapt<'a, S>(events: S, context: Context) -> impl Stream<Item = Result<Event, Error>> + 'a
where
    S: Stream<Item = Result<HeartEvent, Error>> + 'a,
{
    let state = State {
        source: Box::pin(events),
        context,
        queue: VecDeque::new(),
        started: false,
        terminal: false,
        sequence: 0,
        waiting_id: None,
        active_text_id: None,
        active_reasoning_id: None,
        text_started: HashSet::new(),
        reasoning_started: HashSet::new(),
        tools_started: HashSet::new(),
    };

    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.queue.pop_front() {
                return Some((Ok(event), state));
            }

            if !state.started {
                state.started = true;
                return Some((
                    Ok(Event::RunStarted {
                        thread_id: state.context.thread_id.clone(),
                        run_id: state.context.run_id.clone(),
                        parent_run_id: state.context.parent_run_id.clone(),
                    }),
                    state,
                ));
            }

            if state.terminal {
                return None;
            }

            match state.source.next().await {
                Some(Ok(event)) => {
                    state.sequence += 1;
                    let converted = state.convert(event);
                    state.queue.extend(converted);
                }
                Some(Err(error)) => {
                    state.terminal = true;
                    if error.code == "WAITING" {
                        let outcome = state.interrupt_outcome();
                        state.queue.push_back(Event::RunFinished {
                            thread_id: state.context.thread_id.clone(),
                            run_id: state.context.run_id.clone(),
                            outcome: Some(outcome),
                        });
                    } else {
                        state.queue.push_back(Event::RunError {
                            message: error.message,
                            code: Some(error.code),
                        });
                    }
                }
                None => {
                    state.terminal = true;
                    state.queue.push_back(Event::RunFinished {
                        thread_id: state.context.thread_id.clone(),
                        run_id: state.context.run_id.clone(),
                        outcome: state
                            .waiting_id
                            .take()
                            .map(|id| serde_json::json!({"type": "interrupt", "interrupt_id": id})),
                    });
                }
            }
        }
    })
}

struct State<'a> {
    source: InputStream<'a>,
    context: Context,
    queue: VecDeque<Event>,
    started: bool,
    terminal: bool,
    sequence: u64,
    waiting_id: Option<String>,
    active_text_id: Option<String>,
    active_reasoning_id: Option<String>,
    text_started: HashSet<String>,
    reasoning_started: HashSet<String>,
    tools_started: HashSet<String>,
}

impl State<'_> {
    fn convert(&mut self, event: HeartEvent) -> Vec<Event> {
        match event {
            HeartEvent::Delta { message } => self.message(message, false),
            HeartEvent::Complete {
                message,
                usage: _,
                finish_reason: _,
            } => self.message(message, true),
        }
    }

    fn message(&mut self, message: Message, complete: bool) -> Vec<Event> {
        if let Some(reply_id) = message.metadata.get("reply_id") {
            self.waiting_id = Some(value_id(reply_id));
        }
        if matches!(message.role, Role::Tool | Role::Function) {
            return vec![self.tool_result(message)];
        }

        let mut result = Vec::new();
        for part in &message.content {
            match part.r#type.as_str() {
                "text" => {
                    if let Some(text) = text_value(part) {
                        let message_id = self.message_id(&message, "text");
                        let was_active = self.text_started.contains(&message_id);
                        if complete && was_active {
                            self.active_text_id = None;
                            self.text_started.remove(&message_id);
                            result.push(Event::TextMessageEnd { message_id });
                            continue;
                        }
                        if self.text_started.insert(message_id.clone()) {
                            result.push(Event::TextMessageStart {
                                message_id: message_id.clone(),
                                role: role_name(&message.role).to_owned(),
                            });
                        }
                        result.push(Event::TextMessageContent {
                            message_id: message_id.clone(),
                            delta: text,
                        });
                        if complete {
                            self.active_text_id = None;
                            self.text_started.remove(&message_id);
                            result.push(Event::TextMessageEnd {
                                message_id: message_id.clone(),
                            });
                        }
                    }
                }
                "think" => {
                    if let Some(text) = text_value(part) {
                        let reasoning_id = self.message_id(&message, "think");
                        let was_active = self.reasoning_started.contains(&reasoning_id);
                        if complete && was_active {
                            self.active_reasoning_id = None;
                            self.reasoning_started.remove(&reasoning_id);
                            result.push(Event::ReasoningMessageEnd {
                                message_id: reasoning_id,
                            });
                            continue;
                        }
                        if self.reasoning_started.insert(reasoning_id.clone()) {
                            result.push(Event::ReasoningMessageStart {
                                message_id: reasoning_id.clone(),
                            });
                        }
                        result.push(Event::ReasoningMessageContent {
                            message_id: reasoning_id.clone(),
                            delta: text,
                        });
                        if complete {
                            self.active_reasoning_id = None;
                            self.reasoning_started.remove(&reasoning_id);
                            result.push(Event::ReasoningMessageEnd {
                                message_id: reasoning_id,
                            });
                        }
                    }
                }
                "tool_call" => self.tool_call(part, &message, complete, &mut result),
                other => result.push(Event::Custom {
                    name: format!("heart.message.part.{other}"),
                    value: Value::Object(part.data.clone()),
                }),
            }
        }

        if result.is_empty() && !message.values.is_empty() {
            result.push(Event::Custom {
                name: format!("heart.message.{}", role_name(&message.role)),
                value: Value::Object(message.values),
            });
        }
        result
    }

    fn tool_call(
        &mut self,
        part: &Part,
        message: &Message,
        complete: bool,
        output: &mut Vec<Event>,
    ) {
        let call_id = part
            .data
            .get("call_id")
            .and_then(Value::as_str)
            .or_else(|| message.metadata.get("call_id").and_then(Value::as_str))
            .unwrap_or("unknown")
            .to_owned();
        let name = part
            .data
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("tool")
            .to_owned();
        let was_active = self.tools_started.contains(&call_id);
        if !was_active {
            output.push(Event::ToolCallStart {
                tool_call_id: call_id.clone(),
                tool_call_name: name,
            });
        }
        let arguments = part
            .data
            .get("arguments")
            .map(json_text)
            .or_else(|| text_value(part));
        if let Some(arguments) = arguments
            && !(complete && was_active)
        {
            output.push(Event::ToolCallArgs {
                tool_call_id: call_id.clone(),
                delta: arguments,
            });
        }
        if complete {
            self.tools_started.remove(&call_id);
            output.push(Event::ToolCallEnd {
                tool_call_id: call_id,
            });
        }
    }

    fn tool_result(&self, message: Message) -> Event {
        let tool_call_id = message
            .metadata
            .get("call_id")
            .and_then(Value::as_str)
            .or_else(|| message.values.get("call_id").and_then(Value::as_str))
            .unwrap_or("unknown")
            .to_owned();
        let content = if !message.values.is_empty() {
            Value::Object(message.values)
        } else {
            Value::Array(
                message
                    .content
                    .into_iter()
                    .map(|part| Value::Object(part.data))
                    .collect(),
            )
        };
        Event::ToolCallResult {
            message_id: format!("{}:result", tool_call_id),
            tool_call_id,
            content,
        }
    }

    fn message_id(&mut self, message: &Message, kind: &str) -> String {
        if let Some(id) = message.metadata.get("message_id").and_then(Value::as_str) {
            return id.to_owned();
        }
        if let (Some(node), Some(exec)) = (
            message.metadata.get("node_id").and_then(Value::as_str),
            message.metadata.get("exec_id").and_then(Value::as_i64),
        ) {
            return format!("{node}:{exec}");
        }
        let active = match kind {
            "text" => &mut self.active_text_id,
            "think" => &mut self.active_reasoning_id,
            _ => &mut self.active_text_id,
        };
        active
            .get_or_insert_with(|| format!("{}:message:{}", self.context.run_id, self.sequence))
            .clone()
    }

    fn interrupt_outcome(&mut self) -> Value {
        serde_json::json!({
            "type": "interrupt",
            "interrupt_id": self.waiting_id.take(),
        })
    }
}

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        Role::Function => "function",
    }
}

fn text_value(part: &Part) -> Option<String> {
    part.data
        .get("value")
        .or_else(|| part.data.get("text"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn json_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

/// Convert a Runtime snapshot at an explicit HTTP output boundary.
pub fn snapshot(value: Value) -> Event {
    Event::StateSnapshot { snapshot: value }
}

/// Convert a Runtime state change at an explicit HTTP output boundary.
pub fn delta(value: Value) -> Event {
    Event::StateDelta { delta: value }
}

fn value_id(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.as_i64().map(|id| id.to_string()))
        .unwrap_or_else(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;
    use serde_json::json;

    use super::adapt;
    use crate::components::event::Event as HeartEvent;
    use crate::components::message::{Message, Role};
    use crate::components::model::content::part::Part;
    use crate::components::protocol::ag_ui::{Context, Event};

    #[tokio::test]
    async fn adapts_text_reasoning_and_run_lifecycle() {
        let mut delta = Message::new(Role::Assistant);
        delta.content.push(Part {
            r#type: "think".into(),
            data: json!({"value": "先判断"}).as_object().unwrap().clone(),
        });
        delta.content.push(Part {
            r#type: "text".into(),
            data: json!({"value": "北京晴天"}).as_object().unwrap().clone(),
        });
        let events = futures_util::stream::iter(vec![Ok(HeartEvent::Delta { message: delta })]);
        let output = adapt(events, Context::new("session-1", "chat-1"))
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(output[0], Ok(Event::RunStarted { .. })));
        assert!(output
            .iter()
            .any(|event| matches!(event, Ok(Event::ReasoningMessageContent { delta, .. }) if delta == "先判断")));
        assert!(output
            .iter()
            .any(|event| matches!(event, Ok(Event::TextMessageContent { delta, .. }) if delta == "北京晴天")));
        assert!(matches!(output.last(), Some(Ok(Event::RunFinished { .. }))));
    }

    #[tokio::test]
    async fn maps_tool_call_and_result_without_internal_wrapper() {
        let mut call = Message::new(Role::Assistant);
        call.content.push(Part {
            r#type: "tool_call".into(),
            data: json!({
                "call_id": "call-1",
                "name": "weather",
                "arguments": {"city": "北京"}
            })
            .as_object()
            .unwrap()
            .clone(),
        });
        let mut result = Message::new(Role::Function);
        result.metadata.insert("call_id".into(), json!("call-1"));
        result.values.insert("data".into(), json!("晴天"));
        let events = futures_util::stream::iter(vec![
            Ok(HeartEvent::Complete {
                message: call,
                usage: None,
                finish_reason: None,
            }),
            Ok(HeartEvent::Complete {
                message: result,
                usage: None,
                finish_reason: None,
            }),
        ]);
        let output = adapt(events, Context::new("session-1", "chat-1"))
            .collect::<Vec<_>>()
            .await;
        assert!(output.iter().any(|event| matches!(
            event,
            Ok(Event::ToolCallStart { tool_call_id, tool_call_name })
                if tool_call_id == "call-1" && tool_call_name == "weather"
        )));
        assert!(output.iter().any(|event| matches!(
            event,
            Ok(Event::ToolCallResult { tool_call_id, .. }) if tool_call_id == "call-1"
        )));
    }

    #[tokio::test]
    async fn reuses_message_id_and_does_not_repeat_complete_text() {
        fn text(value: &str) -> Message {
            let mut message = Message::new(Role::Assistant);
            message.content.push(Part {
                r#type: "text".into(),
                data: json!({"value": value}).as_object().unwrap().clone(),
            });
            message
        }

        let events = futures_util::stream::iter(vec![
            Ok(HeartEvent::Delta {
                message: text("北"),
            }),
            Ok(HeartEvent::Delta {
                message: text("京"),
            }),
            Ok(HeartEvent::Complete {
                message: text("北京"),
                usage: None,
                finish_reason: Some("stop".into()),
            }),
        ]);
        let output = adapt(events, Context::new("session-1", "chat-1"))
            .collect::<Vec<_>>()
            .await;
        let ids = output
            .iter()
            .filter_map(|event| match event {
                Ok(Event::TextMessageStart { message_id, .. })
                | Ok(Event::TextMessageContent { message_id, .. })
                | Ok(Event::TextMessageEnd { message_id }) => Some(message_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!ids.is_empty());
        assert!(ids.windows(2).all(|window| window[0] == window[1]));
        assert_eq!(
            output
                .iter()
                .filter(|event| matches!(event, Ok(Event::TextMessageStart { .. })))
                .count(),
            1
        );
        assert_eq!(
            output
                .iter()
                .filter(|event| matches!(event, Ok(Event::TextMessageContent { .. })))
                .count(),
            2
        );
        assert_eq!(
            output
                .iter()
                .filter(|event| matches!(event, Ok(Event::TextMessageEnd { .. })))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn maps_source_error_to_run_error_without_finished() {
        let events = futures_util::stream::iter(vec![Err(crate::Error::new("FAILED", "network"))]);
        let output = adapt(events, Context::new("session-1", "chat-1"))
            .collect::<Vec<_>>()
            .await;
        assert!(
            output
                .iter()
                .any(|event| matches!(event, Ok(Event::RunError { .. })))
        );
        assert!(
            !output
                .iter()
                .any(|event| matches!(event, Ok(Event::RunFinished { outcome: None, .. })))
        );
    }

    #[tokio::test]
    async fn maps_waiting_to_interrupt_outcome() {
        let mut question = Message::new(Role::Assistant);
        question.metadata.insert("reply_id".into(), json!(42));
        question.content.push(Part {
            r#type: "text".into(),
            data: json!({"value": "请确认"}).as_object().unwrap().clone(),
        });
        let events = futures_util::stream::iter(vec![
            Ok(HeartEvent::Complete {
                message: question,
                usage: None,
                finish_reason: None,
            }),
            Err(crate::Error::new("WAITING", "waiting for reply")),
        ]);
        let output = adapt(events, Context::new("session-1", "chat-1"))
            .collect::<Vec<_>>()
            .await;
        assert!(output.iter().any(|event| matches!(
            event,
            Ok(Event::RunFinished { outcome: Some(outcome), .. })
                if outcome["type"] == "interrupt" && outcome["interrupt_id"] == "42"
        )));
        assert!(
            !output
                .iter()
                .any(|event| matches!(event, Ok(Event::RunError { .. })))
        );
    }

    #[test]
    fn serializes_ag_ui_field_names() {
        let event = Event::TextMessageContent {
            message_id: "message-1".into(),
            delta: "hello".into(),
        };
        let value = serde_json::to_value(event).unwrap();
        assert_eq!(value["type"], "TEXT_MESSAGE_CONTENT");
        assert_eq!(value["messageId"], "message-1");
        assert_eq!(value["delta"], "hello");
    }
}
