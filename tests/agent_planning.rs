use futures_core::Stream;
use futures_executor::block_on;
use futures_util::{StreamExt, poll};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::evaluation::Evaluation,
    agent::event::Event,
    agent::message::Role,
    agent::model::{
        chat::{Chat, Request, ResponseFormat},
        content::part::Part,
        token::Usage,
    },
    agent::tool::{Call, Tool, ToolDefinition},
    agent::{self, Agent, Input, Payload},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

fn input() -> Input {
    Input {
        metadata: serde_json::from_value(json!({"chat_id":101,"session_id":102})).unwrap(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"先规划查询天气"}),
            mime_type: None,
        }],
    }
}
fn complete(message: Message) -> Result<Event, Error> {
    Ok(Event::Complete {
        message,
        usage: Some(Usage {
            input: Some(10),
            output: Some(5),
            total: Some(15),
            ..Usage::default()
        }),
        finish_reason: Some("stop".into()),
    })
}
fn trigger(arguments: Value) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({"call_id":"p1","name":"plan","arguments":arguments})
            .as_object()
            .unwrap()
            .clone(),
    });
    message
}
fn plan(version: u64) -> Value {
    json!({"version":version,"nodes":{"a":{"name":"search","objective":"查询天气"}},"edges":[]})
}
fn parallel_plan(version: u64) -> Value {
    json!({
        "version": version,
        "nodes": {
            "a": {"name": "search_a", "objective": "查询北京天气"},
            "b": {"name": "search_b", "objective": "查询上海天气"}
        },
        "edges": []
    })
}
fn plan_message(value: Value) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.values = serde_json::from_value(value).unwrap();
    message
}
fn final_message() -> Message {
    let mut message = Message::new(Role::Assistant);
    message.values.insert("answer".into(), json!("done"));
    message
}
fn next_plan_exec_id() -> i64 {
    9001
}
fn next_parallel_exec_id() -> i64 {
    use std::sync::atomic::{AtomicI64, Ordering};
    static NEXT: AtomicI64 = AtomicI64::new(9100);
    NEXT.fetch_add(1, Ordering::Relaxed)
}
type Replies = VecDeque<Vec<Result<Event, Error>>>;

#[derive(Clone)]
struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    replies: Arc<Mutex<Replies>>,
    pending: bool,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}
impl Model {
    fn new(replies: Vec<Vec<Result<Event, Error>>>) -> Self {
        Self {
            requests: Default::default(),
            replies: Arc::new(Mutex::new(replies.into())),
            pending: false,
            dropped: Default::default(),
        }
    }
}
struct Events {
    events: VecDeque<Result<Event, Error>>,
    pending: bool,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}
impl Drop for Events {
    fn drop(&mut self) {
        self.dropped
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}
impl Stream for Events {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.events.is_empty() && self.pending {
            Poll::Pending
        } else {
            Poll::Ready(self.events.pop_front())
        }
    }
}
impl Chat for Model {
    type Stream = Events;
    async fn stream(&self, request: Request) -> Result<Events, Error> {
        let local = std::rc::Rc::new(());
        std::future::ready(()).await;
        drop(local);
        request.validate()?;
        self.requests.lock().unwrap().push(request);
        let events = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected model request")
            .into();
        Ok(Events {
            events,
            pending: self.pending,
            dropped: self.dropped.clone(),
        })
    }
}
struct Search {
    definition: ToolDefinition,
    calls: Arc<Mutex<usize>>,
}
impl Search {
    fn new(name: &str) -> Self {
        Self {
            definition: ToolDefinition {
                name: name.into(),
                description: "weather search".into(),
                parameters: serde_json::from_value(
                    json!({"type":"object","properties":{},"additionalProperties":false}),
                )
                .unwrap(),
            },
            calls: Default::default(),
        }
    }
}
impl Tool for Search {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        *self.calls.lock().unwrap() += 1;
        Ok(Message::function(json!({"weather":"sunny"})))
    }
}
fn context(request: &Request) -> Value {
    let message = request.messages.as_slice().last().unwrap();
    serde_json::from_str(message.content[0].data["value"].as_str().unwrap()).unwrap()
}

fn answer(value: &str) -> Message {
    let mut message = final_message();
    message.values.insert("answer".into(), json!(value));
    message
}

fn revised_plan() -> Value {
    let mut value = plan(2);
    value["nodes"]["a"]["objective"] = json!("重新查证天气，补充日期");
    value
}

async fn final_evaluation(
    _: Input,
    result: halo::agent::Messages,
    _: Cancellation,
) -> Result<Evaluation, Error> {
    let passed = result.as_slice()[0].values["answer"] == "accepted";
    Ok(Evaluation {
        passed,
        feedback: (!passed).then(|| {
            halo::agent::Messages::new([Message::function(json!({"feedback": "需要补充天气日期"}))])
        }),
    })
}

#[test]
fn rejected_final_replans_directly_and_reexecutes_even_the_same_logical_node() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(1)))],
            vec![complete(answer("old task result"))],
            vec![complete(answer("rejected"))],
            vec![complete(plan_message(revised_plan()))],
            vec![complete(answer("new task result"))],
            vec![complete(answer("accepted"))],
        ]);
        let requests = model.requests.clone();
        let evaluations = Arc::new(Mutex::new(Vec::new()));
        let seen = evaluations.clone();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_parallel_exec_id)
            .evaluator(
                move |query: Input, result: halo::agent::Messages, cancellation: Cancellation| {
                    seen.lock().unwrap().push((query.clone(), result.clone()));
                    final_evaluation(query, result, cancellation)
                },
            )
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events.len(), 8);
        assert!(events.iter().all(Result::is_ok));
        let nodes: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Ok(Event::Complete { message, .. }) if message.metadata.contains_key("node_id") => {
                    Some(message)
                }
                _ => None,
            })
            .collect();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].metadata["node_id"], nodes[1].metadata["node_id"]);
        assert_ne!(nodes[0].metadata["exec_id"], nodes[1].metadata["exec_id"]);
        // Only the real initial model-selected plan call receives an acknowledgement.
        assert_eq!(events.iter().filter(|event| matches!(event,
            Ok(Event::Complete {message, usage: None, ..}) if message.metadata.contains_key("call_id")
        )).count(), 1);
        let seen = evaluations.lock().unwrap();
        assert_eq!(seen.len(), 2); // Never evaluate individual task/model/tool results.
        assert!(seen.iter().all(|(query, _)| query == &input()));
        assert_eq!(seen[0].1.as_slice()[0].values["answer"], "rejected");
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 7);
        let replan = &requests[4];
        assert!(replan.tools.is_empty());
        assert_eq!(replan.response_format, Some(ResponseFormat::JsonObject));
        let mut message_metadata = input().metadata;
        message_metadata.remove("chat_id");
        assert_eq!(replan.messages.as_slice()[1].metadata, message_metadata);
        assert_eq!(context(replan)["current_plan"], plan(1));
        assert_eq!(
            context(replan)["feedback"][0]["values"]["feedback"],
            "需要补充天气日期"
        );
        assert_eq!(context(&requests[5])["sources"], json!({}));
        assert_eq!(context(&requests[5])["objective"], "重新查证天气，补充日期");
        let synthesis = context(&requests[6]);
        assert!(synthesis.to_string().contains("new task result"));
        assert!(!synthesis.to_string().contains("old task result"));
        assert!(
            !serde_json::to_string(&requests[6].messages)
                .unwrap()
                .contains("rejected")
        );
    });
}

#[test]
fn graph_rejection_without_feedback_still_replans_instead_of_succeeding() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(1)))],
            vec![complete(answer("task"))],
            vec![complete(answer("rejected"))],
            vec![complete(plan_message(revised_plan()))],
            vec![complete(answer("task2"))],
            vec![complete(answer("accepted"))],
        ]);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_parallel_exec_id)
            .evaluator(
                |_: Input, result: halo::agent::Messages, _: Cancellation| async move {
                    Ok(Evaluation {
                        passed: result.as_slice()[0].values["answer"] == "accepted",
                        feedback: None,
                    })
                },
            )
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        assert_eq!(requests.lock().unwrap().len(), 7);
        assert_eq!(
            context(&requests.lock().unwrap()[4])["feedback"],
            Value::Null
        );
    });
}

#[test]
fn invalid_or_failed_replan_never_falls_back_to_rejected_plan() {
    block_on(async {
        let mut invalid = revised_plan();
        invalid["edges"] = json!([{"from":"a","to":"a"}]);
        for last in [
            complete(plan_message(invalid)),
            Err(Error::new("TRANSPORT", "replan offline")),
        ] {
            let model = Model::new(vec![
                vec![complete(trigger(json!({})))],
                vec![complete(plan_message(plan(1)))],
                vec![complete(answer("old task"))],
                vec![complete(answer("rejected"))],
                vec![last.clone()],
            ]);
            let requests = model.requests.clone();
            let agent = agent::react()
                .model(model)
                .with_id_fn(next_parallel_exec_id)
                .evaluator(final_evaluation)
                .build()
                .unwrap();
            let events: Vec<_> = agent
                .run(input(), true, Cancellation::new())
                .await
                .unwrap()
                .collect()
                .await;
            assert!(events.last().unwrap().is_err());
            assert_eq!(
                events.last().unwrap().as_ref().unwrap_err().code,
                if last.is_err() {
                    "TRANSPORT"
                } else {
                    "INVALID_ARGUMENTS"
                }
            );
            if last.is_ok() {
                assert!(events[5].is_ok());
            } // Real invalid Complete still forwarded.
            assert_eq!(requests.lock().unwrap().len(), 5);
        }
    });
}

#[test]
fn evaluation_replan_shares_max_iters_and_cannot_reset_the_budget() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(1)))],
            vec![complete(answer("task"))],
            vec![complete(answer("rejected"))],
        ]);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_parallel_exec_id)
            .with_max_iters(4)
            .evaluator(final_evaluation)
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(
            events.last().unwrap().as_ref().unwrap_err().code,
            "MAX_ITERS"
        );
        assert_eq!(requests.lock().unwrap().len(), 4);
    });
}

#[test]
fn cancellation_wakes_evaluation_replan_without_acknowledging_it() {
    block_on(async {
        let mut model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(1)))],
            vec![complete(answer("task"))],
            vec![complete(answer("rejected"))],
            vec![],
        ]);
        model.pending = true;
        let requests = model.requests.clone();
        let dropped = model.dropped.clone();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_parallel_exec_id)
            .evaluator(final_evaluation)
            .build()
            .unwrap();
        let cancellation = Cancellation::new();
        let mut events = agent
            .run(input(), true, cancellation.clone())
            .await
            .unwrap();
        for _ in 0..5 {
            events.next().await.unwrap().unwrap();
        }
        dropped.store(false, std::sync::atomic::Ordering::SeqCst);
        let mut pending = Box::pin(events.next());
        assert!(poll!(pending.as_mut()).is_pending());
        assert_eq!(requests.lock().unwrap().len(), 5);
        cancellation.cancel();
        assert!(
            matches!(pending.await, Some(Ok(Event::Complete {finish_reason: Some(reason), ..})) if reason == "cancelled")
        );
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        assert!(events.next().await.is_none());
    });
}

#[test]
fn planning_only_rejection_corrects_the_answer_without_reexecuting_a_graph() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(1)))],
            vec![complete(answer("rejected"))],
            vec![complete(answer("accepted"))],
        ]);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .evaluator(final_evaluation)
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        assert_eq!(events.len(), 5);
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, Ok(Event::Complete { message, .. })
            if message.metadata.contains_key("exec_id")))
        );
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert!(requests[3].response_format.is_none());
        assert_eq!(
            requests[3].messages.as_slice().last().unwrap().values["feedback"],
            "需要补充天气日期"
        );
    });
}

#[test]
fn replan_cannot_reuse_an_execution_id_from_the_previous_plan() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(1)))],
            vec![complete(answer("old task"))],
            vec![complete(answer("rejected"))],
            vec![complete(plan_message(revised_plan()))],
        ]);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_plan_exec_id)
            .evaluator(final_evaluation)
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(
            events.last().unwrap().as_ref().unwrap_err().code,
            "CONFLICT"
        );
        assert_eq!(requests.lock().unwrap().len(), 5);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Ok(Event::Complete { message, .. })
            if message.metadata.contains_key("exec_id")))
                .count(),
            1
        );
    });
}

#[test]
fn built_in_plan_is_advertised_but_never_guessed_from_query_text() {
    block_on(async {
        let model = Model::new(vec![vec![complete(final_message())]]);
        let requests = model.requests.clone();
        let agent = agent::react().model(model).build().unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events.len(), 1);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].tools.len(), 1);
        assert_eq!(requests[0].tools[0].name, "plan");
        assert_eq!(
            requests[0].tools[0].parameters["additionalProperties"],
            false
        );
        assert!(requests[0].response_format.is_none());
    });
}

#[test]
fn trigger_streams_planner_events_and_returns_validated_plan_without_running_tools() {
    block_on(async {
        let mut delta = Message {
            content: vec![Part {
                r#type: "text".into(),
                data: json!({"value":"partial"}).as_object().unwrap().clone(),
            }],
            ..Message::new(Role::Assistant)
        };
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![
                Ok(Event::Delta {
                    message: delta.clone(),
                }),
                complete(plan_message(plan(7))),
            ],
            vec![complete(final_message())],
        ]);
        let requests = model.requests.clone();
        let search = Search::new("search");
        let calls = search.calls.clone();
        let agent = agent::react().model(model).tools(search).build().unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events.len(), 5);
        delta.metadata.insert("chat_id".into(), json!(101));
        assert_eq!(events[1], Ok(Event::Delta { message: delta }));
        assert!(
            matches!(&events[2],Ok(Event::Complete {message,usage:Some(usage),..}) if message.values["version"]==7 && usage.total==Some(15))
        );
        assert!(
            matches!(&events[3],Ok(Event::Complete {message,usage:None,..}) if message.values["plan"]==plan(7) && message.metadata["call_id"]=="p1")
        );
        assert_eq!(*calls.lock().unwrap(), 0); // Plan creation is not execution.
        let requests = requests.lock().unwrap();
        let request = &requests[1];
        assert_eq!(request.response_format, Some(ResponseFormat::JsonObject));
        assert!(request.tools.is_empty());
        assert!(request.tool_choice.is_none());
        assert!(request.stream);
        let mut message_metadata = input().metadata;
        message_metadata.remove("chat_id");
        assert_eq!(request.messages.as_slice()[1].metadata, message_metadata);
        assert_eq!(
            context(request),
            json!({"current_plan":null,"feedback":null})
        );
        assert!(
            requests[2]
                .messages
                .as_slice()
                .iter()
                .any(|m| m.values.get("plan") == Some(&plan(7)))
        );
    });
}

#[test]
fn configured_id_source_executes_plan_nodes_before_final_synthesis() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(8)))],
            vec![complete(final_message())],
            vec![complete(final_message())],
        ]);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_plan_exec_id)
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;

        assert_eq!(events.len(), 5);
        assert!(events.iter().all(Result::is_ok));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        // Planner request has no business tools. The planned task request
        // exposes the Toolkit but does not expose the reserved plan control.
        assert!(requests[1].tools.is_empty());
        assert!(requests[2].tools.is_empty());
        assert!(
            requests[2]
                .messages
                .as_slice()
                .iter()
                .any(|message| { message.values.get("objective") == Some(&json!("查询天气")) })
        );
        assert!(requests[3].tools.is_empty());
        assert!(events.iter().any(|event| {
            matches!(event, Ok(Event::Complete { message, .. }) if message.values.get("answer") == Some(&json!("done")))
        }));
    });
}

#[test]
fn configured_id_source_runs_ready_plan_frontier_concurrently() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(parallel_plan(10)))],
            vec![complete(final_message())],
            vec![complete(final_message())],
            vec![complete(final_message())],
        ]);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_parallel_exec_id)
            .with_max_concurrency(2)
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;

        assert_eq!(events.len(), 6);
        assert!(events.iter().all(Result::is_ok));
        let task_nodes = events
            .iter()
            .filter_map(|event| match event {
                Ok(Event::Complete { message, .. }) => message.metadata.get("node_id"),
                _ => None,
            })
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        assert_eq!(task_nodes, vec!["a", "b"]);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 5);
        assert_eq!(
            requests[2].messages.as_slice().last().unwrap().values["objective"],
            "查询北京天气"
        );
        assert_eq!(
            requests[3].messages.as_slice().last().unwrap().values["objective"],
            "查询上海天气"
        );
    });
}

#[test]
fn repeated_trigger_passes_current_plan_without_inferring_capabilities_or_changing_version() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(4)))],
            vec![complete(trigger(json!({})))],
            vec![complete(plan_message(plan(9)))],
            vec![complete(final_message())],
        ]);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .tools(Search::new("search"))
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        let requests = requests.lock().unwrap();
        assert_eq!(context(&requests[3])["current_plan"], plan(4));
        assert!(context(&requests[3]).get("capabilities").is_none());
        assert_eq!(
            requests[1].messages.as_slice()[1],
            requests[3].messages.as_slice()[1]
        );
        assert!(
            requests[4]
                .messages
                .as_slice()
                .iter()
                .any(|m| m.values.get("plan") == Some(&plan(9)))
        );
    });
}

#[test]
fn invalid_trigger_arguments_are_direct_context_without_starting_planner() {
    block_on(async {
        let model = Model::new(vec![
            vec![complete(trigger(json!({"query":"override"})))],
            vec![complete(final_message())],
        ]);
        let requests = model.requests.clone();
        let agent = agent::react().model(model).build().unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(Result::is_ok));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].response_format.is_none());
        assert!(requests[1].messages.as_slice().iter().any(|m| {
            m.values
                .get("error")
                .is_some_and(|e| e["code"] == "INVALID_ARGUMENTS")
        }));
    });
}

#[test]
fn invalid_complete_plan_is_terminal_and_never_acknowledged_as_a_plan() {
    block_on(async {
        for candidate in [
            json!({"version":1,"nodes":{"a":{"name":"search","objective":"bad"}},"edges":[{"from":"a","to":"a"}]}),
            json!({"version":1,"nodes":{"a":{"name":"search","objective":"bad","arguments":{}}},"edges":[]}),
        ] {
            let model = Model::new(vec![
                vec![complete(trigger(json!({})))],
                vec![complete(plan_message(candidate))],
            ]);
            let requests = model.requests.clone();
            let agent = agent::react()
                .model(model)
                .tools(Search::new("search"))
                .build()
                .unwrap();
            let events: Vec<_> = agent
                .run(input(), true, Cancellation::new())
                .await
                .unwrap()
                .collect()
                .await;
            assert_eq!(events.len(), 3);
            assert!(matches!(events[1], Ok(Event::Complete { .. })));
            assert!(events[2].is_err());
            assert_eq!(requests.lock().unwrap().len(), 2);
        }
    });
}

#[test]
fn planner_transport_failure_is_preserved_without_retry_or_tool_error_wrapping() {
    block_on(async {
        let error = Error::new("TRANSPORT", "planner offline").with_details("network");
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![Err(error.clone())],
        ]);
        let agent = agent::react().model(model).build().unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], Ok(Event::Complete { .. })));
        assert_eq!(events[1], Err(error));
    });
}

#[test]
fn cancellation_wakes_pending_planner_and_counts_active_input_only_once() {
    block_on(async {
        let mut model = Model::new(vec![vec![complete(trigger(json!({})))], vec![]]);
        model.pending = true;
        let dropped = model.dropped.clone();
        let agent = agent::react().model(model).build().unwrap();
        let cancellation = Cancellation::new();
        let mut events = agent
            .run(input(), true, cancellation.clone())
            .await
            .unwrap();
        events.next().await.unwrap().unwrap();
        dropped.store(false, std::sync::atomic::Ordering::SeqCst);
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        cancellation.cancel();
        assert!(
            matches!(next.await.unwrap().unwrap(),Event::Complete {usage:Some(usage),finish_reason:Some(reason),..} if reason=="cancelled" && usage.input.unwrap()>0)
        );
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        assert!(events.next().await.is_none());
    });
}

#[test]
fn builtin_name_cannot_be_shadowed_by_a_registered_tool() {
    let model = Model::new(vec![]);
    assert_eq!(
        agent::react()
            .model(model)
            .tools(Search::new("plan"))
            .build()
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn planner_model_calls_share_the_existing_max_iters_budget() {
    block_on(async {
        let model = Model::new(vec![vec![complete(trigger(json!({})))]]);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .with_max_iters(1)
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events[1].as_ref().unwrap_err().code, "MAX_ITERS");
        assert_eq!(requests.lock().unwrap().len(), 1);
    });
}

#[test]
fn mixed_batch_finishes_business_calls_before_starting_planner() {
    block_on(async {
        let mut selection = trigger(json!({}));
        selection.content.push(Part {
            r#type: "tool_call".into(),
            data: json!({"call_id":"s1","name":"search","arguments":{}})
                .as_object()
                .unwrap()
                .clone(),
        });
        let model = Model::new(vec![
            vec![complete(selection)],
            vec![complete(plan_message(plan(1)))],
            vec![complete(final_message())],
        ]);
        let search = Search::new("search");
        let calls = search.calls.clone();
        let agent = agent::react()
            .model(model)
            .tools(search)
            .with_max_concurrency(1)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        events.next().await.unwrap().unwrap();
        assert!(
            matches!(events.next().await.unwrap().unwrap(),Event::Complete {message,..} if message.values["weather"]=="sunny")
        );
        assert_eq!(*calls.lock().unwrap(), 1);
        assert!(
            matches!(events.next().await.unwrap().unwrap(),Event::Complete {message,..} if message.values["version"]==1)
        );
        let rest: Vec<_> = events.collect().await;
        assert_eq!(rest.len(), 2);
        assert!(rest.iter().all(Result::is_ok));
    });
}

#[test]
fn cancelled_planner_complete_does_not_acknowledge_or_commit_its_plan() {
    block_on(async {
        let cancelled = Ok(Event::Complete {
            message: plan_message(plan(1)),
            usage: None,
            finish_reason: Some("cancelled".into()),
        });
        let model = Model::new(vec![vec![complete(trigger(json!({})))], vec![cancelled]]);
        let agent = agent::react()
            .model(model)
            .tools(Search::new("search"))
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[1],Ok(Event::Complete {finish_reason:Some(reason),..}) if reason=="cancelled")
        );
    });
}

#[test]
fn missing_planner_complete_cannot_be_reconstructed_from_delta() {
    block_on(async {
        let mut delta = Message::new(Role::Assistant);
        delta.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":plan(1).to_string()})
                .as_object()
                .unwrap()
                .clone(),
        });
        let model = Model::new(vec![
            vec![complete(trigger(json!({})))],
            vec![Ok(Event::Delta { message: delta })],
        ]);
        let agent = agent::react()
            .model(model)
            .tools(Search::new("search"))
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(events.len(), 3);
        assert!(matches!(events[1], Ok(Event::Delta { .. })));
        assert_eq!(events[2].as_ref().unwrap_err().code, "INVALID_RESPONSE");
    });
}

#[test]
fn planner_usage_is_in_chat_total_but_control_ack_has_no_double_charge() {
    use chat::Store as _;
    use halo::{
        agent::memory::{
            self,
            short::{chat, session},
        },
        core,
    };
    use session::Store as _;
    block_on(async {
        for invalid in [false, true] {
            let memory = memory::Config::new(
                memory::short::Pair::new(
                    core::memory::short::session::InMemory::new(),
                    core::memory::short::chat::InMemory::new(),
                ),
                core::memory::InMemory::new(),
            );
            memory
                .session()
                .create(
                    session::Session {
                        id: 102,
                        tenant_id: 1,
                        user_id: 2,
                        r#type: memory::short::Type::Agent,
                        title: None,
                        summary: None,
                        usage: None,
                        status: session::Status::Active,
                        metadata: Default::default(),
                        created_time: 0,
                        updated_time: 0,
                    },
                    &Cancellation::new(),
                )
                .await
                .unwrap();
            let mut value = plan(1);
            if invalid {
                value["edges"] = json!([{"from":"a","to":"a"}]);
            }
            let mut replies = vec![
                vec![complete(trigger(json!({})))],
                vec![complete(plan_message(value))],
            ];
            if !invalid {
                replies.push(vec![complete(final_message())]);
            }
            let model = Model::new(replies);
            let agent = agent::react()
                .model(model)
                .memory(memory.clone())
                .tools(Search::new("search"))
                .build()
                .unwrap();
            let mut query = input();
            query.metadata.extend(
                serde_json::from_value::<serde_json::Map<String, Value>>(
                    json!({"tenant_id":1,"user_id":2,"agent_id":3}),
                )
                .unwrap(),
            );
            let events: Vec<_> = agent
                .run(query, true, Cancellation::new())
                .await
                .unwrap()
                .collect()
                .await;
            assert_eq!(events.last().unwrap().is_err(), invalid);
            let record = memory.chat().read(101, &Cancellation::new()).await.unwrap();
            assert_eq!(
                record.usage.unwrap().total,
                Some(if invalid { 30 } else { 45 })
            );
            assert_eq!(
                record.status,
                if invalid {
                    chat::Status::Failed
                } else {
                    chat::Status::Completed
                }
            );
            if invalid {
                assert!(record.graph.is_none());
                let state = record.state.unwrap();
                assert_eq!(state["type"], "agent");
                assert_eq!(state["status"], "failed");
            } else {
                assert_eq!(
                    record.graph.as_ref().unwrap()["nodes"]["a"]["name"],
                    "search"
                );
                let state = record.state.unwrap();
                assert_eq!(state["type"], "agent");
                assert_eq!(state["status"], "succeeded");
                assert_eq!(state["nodes"]["a"]["status"], "ready");
            }
        }
    });
}

#[test]
fn configured_plan_closure_persists_final_graph_and_agent_state() {
    use chat::Store as _;
    use halo::agent::checkpoint::Checkpoint as _;
    use halo::{
        agent::memory::{
            self,
            short::{chat, session},
        },
        core,
    };
    use session::Store as _;
    block_on(async {
        let memory = memory::Config::new(
            memory::short::Pair::new(
                core::memory::short::session::InMemory::new(),
                core::memory::short::chat::InMemory::new(),
            ),
            core::memory::InMemory::new(),
        );
        let checkpoint = std::sync::Arc::new(core::checkpoint::InMemory::new());
        memory
            .session()
            .create(
                session::Session {
                    id: 102,
                    tenant_id: 1,
                    user_id: 2,
                    r#type: memory::short::Type::Agent,
                    title: None,
                    summary: None,
                    usage: None,
                    status: session::Status::Active,
                    metadata: Default::default(),
                    created_time: 0,
                    updated_time: 0,
                },
                &Cancellation::new(),
            )
            .await
            .unwrap();
        let agent = agent::react()
            .model(Model::new(vec![
                vec![complete(trigger(json!({})))],
                vec![complete(plan_message(plan(1)))],
                vec![complete(answer("task"))],
                vec![complete(final_message())],
            ]))
            .with_id_fn(next_plan_exec_id)
            .memory(memory.clone())
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let mut query = input();
        query.metadata.extend(
            serde_json::from_value::<serde_json::Map<String, Value>>(
                json!({"tenant_id":1,"user_id":2,"agent_id":3}),
            )
            .unwrap(),
        );
        let events: Vec<_> = agent
            .run(query, true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        let chat = memory.chat().read(101, &Cancellation::new()).await.unwrap();
        assert_eq!(chat.status, chat::Status::Completed);
        assert_eq!(chat.graph.unwrap()["nodes"]["a"]["name"], "search");
        let state = chat.state.clone().unwrap();
        assert_eq!(state["type"], "agent");
        assert_eq!(state["status"], "succeeded");
        assert_eq!(state["nodes"]["a"]["status"], "succeeded");
        assert_eq!(state["nodes"]["a"]["execution"]["exec_id"], 9001);
        let checkpoint_state = checkpoint.load(101, &Cancellation::new()).await.unwrap();
        assert_eq!(checkpoint_state["type"], "agent");
        assert_eq!(checkpoint_state["status"], "succeeded");
        assert_eq!(checkpoint_state["nodes"]["a"]["status"], "succeeded");
    });
}

#[test]
fn checkpoint_backend_saves_an_ordinary_react_runtime_without_graph() {
    use halo::agent::checkpoint::Checkpoint as _;
    block_on(async {
        let checkpoint = std::sync::Arc::new(halo::core::checkpoint::InMemory::new());
        let agent = agent::react()
            .model(Model::new(vec![vec![complete(final_message())]]))
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        let snapshot = checkpoint.load(101, &Cancellation::new()).await.unwrap();
        assert_eq!(snapshot["type"], "agent");
        assert_eq!(snapshot["status"], "succeeded");
        assert!(snapshot.get("graph").is_none());
        assert!(snapshot.get("nodes").is_none());
    });
}

#[test]
fn plan_failure_and_cancellation_persist_once_only_at_terminal_closeout() {
    use chat::Store as _;
    use halo::{
        agent::memory::{
            self,
            short::{chat, session},
        },
        core,
    };
    use session::Store as _;
    block_on(async {
        for cancelled in [false, true] {
            let memory = memory::Config::new(
                memory::short::Pair::new(
                    core::memory::short::session::InMemory::new(),
                    core::memory::short::chat::InMemory::new(),
                ),
                core::memory::InMemory::new(),
            );
            memory
                .session()
                .create(
                    session::Session {
                        id: 102,
                        tenant_id: 1,
                        user_id: 2,
                        r#type: memory::short::Type::Agent,
                        title: None,
                        summary: None,
                        usage: None,
                        status: session::Status::Active,
                        metadata: Default::default(),
                        created_time: 0,
                        updated_time: 0,
                    },
                    &Cancellation::new(),
                )
                .await
                .unwrap();
            let task_result = if cancelled {
                Ok(Event::Complete {
                    message: answer("partial task"),
                    usage: Some(Usage {
                        input: Some(10),
                        output: Some(2),
                        total: Some(12),
                        ..Usage::default()
                    }),
                    finish_reason: Some("cancelled".into()),
                })
            } else {
                Err(Error::new("TRANSPORT", "task connection lost"))
            };
            let agent = agent::react()
                .model(Model::new(vec![
                    vec![complete(trigger(json!({})))],
                    vec![complete(plan_message(plan(1)))],
                    vec![task_result],
                ]))
                .with_id_fn(next_plan_exec_id)
                .memory(memory.clone())
                .build()
                .unwrap();
            let mut query = input();
            query.metadata.extend(
                serde_json::from_value::<serde_json::Map<String, Value>>(
                    json!({"tenant_id":1,"user_id":2,"agent_id":3}),
                )
                .unwrap(),
            );
            let mut events = agent.run(query, true, Cancellation::new()).await.unwrap();
            // Trigger, real planner Complete and acknowledgement are visible,
            // but neither Chat.graph nor Chat.state is written yet.
            for _ in 0..3 {
                events.next().await.unwrap().unwrap();
                let chat = memory.chat().read(101, &Cancellation::new()).await.unwrap();
                assert_eq!(chat.status, chat::Status::Running);
                assert!(chat.graph.is_none());
                assert!(chat.state.is_none());
            }
            let remaining: Vec<_> = events.by_ref().collect().await;
            assert!(remaining.last().unwrap().is_err());
            let chat = memory.chat().read(101, &Cancellation::new()).await.unwrap();
            assert_eq!(
                chat.status,
                if cancelled {
                    chat::Status::Stopped
                } else {
                    chat::Status::Failed
                }
            );
            assert_eq!(chat.graph.unwrap(), plan(1));
            let state = chat.state.clone().unwrap();
            assert_eq!(
                state["status"],
                if cancelled { "stopped" } else { "failed" }
            );
            assert_eq!(
                state["nodes"]["a"]["status"],
                if cancelled { "cancelled" } else { "failed" }
            );
            assert_eq!(state["nodes"]["a"]["execution"]["exec_id"], 9001);
            assert_eq!(
                state["error"]["code"],
                if cancelled { "CANCELLED" } else { "TRANSPORT" }
            );
            assert_eq!(state["usage"], serde_json::to_value(chat.usage).unwrap());
        }
    });
}
