use super::*;
use crate::agent::Agent as _;
use crate::checkpoint::Checkpoint as _;
use crate::components::model::token::Usage;
use crate::tool::Tool;
use futures_executor::block_on;
use std::sync::{Mutex, atomic::AtomicI64};

#[derive(Default)]
struct Probe {
    requests: Vec<(Option<String>, Request)>,
    rounds: BTreeMap<String, usize>,
    calls: BTreeMap<(String, String), usize>,
    evaluations: BTreeMap<(String, String), usize>,
    final_evaluations: usize,
    eval_error: bool,
    eval_wait: bool,
}

struct Model(Arc<Mutex<Probe>>);

impl Chat for Model {
    type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let node = request
            .messages
            .as_slice()
            .iter()
            .rev()
            .find_map(|message| {
                message
                    .values
                    .get("plan_node")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            });
        let mut probe = self.0.lock().unwrap();
        let mut message = Message::new(Role::Assistant);
        match node.as_deref() {
            Some("a" | "b") => {
                let id = node.as_ref().unwrap();
                let round = probe.rounds.entry(id.clone()).or_default();
                if *round == 0 {
                    // Same call IDs in different plan nodes must remain isolated.
                    for name in ["lookup", "clock"] {
                        message.content.push(Part {
                            r#type: "tool_call".into(),
                            data: part_data(
                                json!({"call_id":name,"name":name,"arguments":{"target":id}}),
                            ),
                        });
                    }
                } else {
                    message
                        .values
                        .insert("answer".into(), json!(format!("checked-{id}")));
                }
                *round += 1;
            }
            Some("c") => {
                message.values.insert("answer".into(), json!("merged"));
            }
            Some(other) => panic!("unexpected plan node: {other}"),
            None if request
                .messages
                .0
                .last()
                .unwrap()
                .values
                .contains_key("plan_complete") =>
            {
                message.values.insert("answer".into(), json!("final"));
            }
            None if probe.requests.is_empty() => {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: part_data(json!({"call_id":"plan_1","name":"plan","arguments":{}})),
                });
            }
            None => {
                message.values = part_data(json!({"version":1,"nodes":{
                    "a":{"name":"research","objective":"verify first result"},
                    "b":{"name":"research","objective":"verify second result"},
                    "c":{"name":"combine","objective":"combine verified results"}
                },"edges":[{"from":"a","to":"c"},{"from":"b","to":"c"}]}));
            }
        }
        probe.requests.push((node, request));
        Ok(futures_util::stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(2),
                output: Some(1),
                total: Some(3),
                ..Usage::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

struct CheckedTool {
    definition: ToolDefinition,
    probe: Arc<Mutex<Probe>>,
}

impl CheckedTool {
    fn new(name: &str, probe: Arc<Mutex<Probe>>) -> Self {
        Self {
            definition: ToolDefinition {
                name: name.into(),
                description: format!("{name} business capability"),
                parameters: part_data(
                    json!({"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}),
                ),
            },
            probe,
        }
    }
}

impl Tool for CheckedTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        let target = call.arguments["target"].as_str().unwrap().to_owned();
        *self
            .probe
            .lock()
            .unwrap()
            .calls
            .entry((call.name.clone(), target.clone()))
            .or_default() += 1;
        let mut message = Message::new(Role::Tool);
        message.values = part_data(json!({"result":format!("{}-{target}",call.name)}));
        Ok(message)
    }

    async fn after(&self, _: &Call, message: &mut Message, _: &Cancellation) -> Result<(), Error> {
        message.values.insert("after".into(), json!(true));
        Ok(())
    }

    async fn eval(
        &self,
        call: &Call,
        message: &Message,
        _: &Cancellation,
    ) -> Result<ToolEvaluation, Error> {
        assert_eq!(message.values["after"], true);
        let target = call.arguments["target"].as_str().unwrap().to_owned();
        let mut probe = self.probe.lock().unwrap();
        *probe
            .evaluations
            .entry((call.name.clone(), target.clone()))
            .or_default() += 1;
        if call.name == "lookup" && target == "a" {
            if probe.eval_wait {
                probe.eval_wait = false;
                return Err(Error::new("WAITING", "business evaluation needs approval"));
            }
            if probe.eval_error {
                return Err(Error::new("TIMEOUT", "eval service unavailable"));
            }
            let mut feedback = Message::new(Role::User);
            feedback.content.push(Part {
                r#type: "text".into(),
                data: part_data(
                    json!({"value":"Verify the date without repeating the successful lookup."}),
                ),
            });
            feedback
                .values
                .insert("business_feedback".into(), json!("verify-date"));
            return Ok(ToolEvaluation::failed(Messages::new([feedback])));
        }
        Ok(ToolEvaluation::passed())
    }
}

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(60_000);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn has_feedback(messages: &Messages) -> bool {
    messages
        .0
        .iter()
        .any(|message| message.values.contains_key("business_feedback"))
}

#[test]
fn planned_eval_wait_resumes_the_same_attempt_and_keeps_feedback_local() {
    block_on(async {
        let probe = Arc::new(Mutex::new(Probe {
            eval_wait: true,
            ..Default::default()
        }));
        let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
        let agent = crate::agent::react()
            .model(Model(probe.clone()))
            .toolkit(
                Toolkit::new()
                    .tool(CheckedTool::new("lookup", probe.clone()))
                    .tool(CheckedTool::new("clock", probe.clone())),
            )
            .with_id_fn(next_id)
            .with_max_concurrency(2)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let input = Input {
            metadata: part_data(json!({"chat_id":602})),
            payload: vec![],
        };
        let events = agent
            .run(input.clone(), true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
        let before = checkpoint.load(602, &Cancellation::new()).await.unwrap();
        let attempt = &before["nodes"]["a"]["execution"]["attempts"][0];
        let tool = attempt["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .unwrap();
        assert_eq!(tool["stage"], "eval");
        assert_eq!(before["nodes"]["b"]["status"], "succeeded");
        let b = before["nodes"]["b"].clone();
        drop(agent);
        let restored = crate::agent::react()
            .model(Model(probe.clone()))
            .toolkit(
                Toolkit::new()
                    .tool(CheckedTool::new("lookup", probe.clone()))
                    .tool(CheckedTool::new("clock", probe.clone())),
            )
            .with_id_fn(next_id)
            .with_max_concurrency(2)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let events = restored
            .run(input, true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let after = checkpoint.load(602, &Cancellation::new()).await.unwrap();
        assert_eq!(
            after["nodes"]["b"], b,
            "completed sibling remains unchanged"
        );
        assert_eq!(
            after["nodes"]["a"]["execution"]["exec_id"],
            before["nodes"]["a"]["execution"]["exec_id"]
        );
        assert_eq!(
            after["nodes"]["a"]["execution"]["attempts"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            after["nodes"]["a"]["execution"]["attempts"][0]["created_time"],
            attempt["created_time"]
        );
        let actual: Messages =
            serde_json::from_value(after["nodes"]["a"]["messages"].clone()).unwrap();
        assert!(!has_feedback(&actual));
        assert!(
            actual
                .0
                .iter()
                .all(|m| m.role != Role::System && m.role != Role::User)
        );
        let context: Messages = serde_json::from_value(
            after["nodes"]["a"]["execution"]["attempts"][0]["messages"].clone(),
        )
        .unwrap();
        assert!(has_feedback(&context));
        let probe = probe.lock().unwrap();
        assert!(probe.calls.values().all(|count| *count == 1));
        assert_eq!(probe.evaluations[&("lookup".into(), "a".into())], 2);
        assert_eq!(probe.evaluations[&("lookup".into(), "b".into())], 1);
        let request = &probe
            .requests
            .iter()
            .find(|(node, req)| node.as_deref() == Some("a") && has_feedback(&req.messages))
            .unwrap()
            .1;
        assert_eq!(
            request
                .messages
                .0
                .iter()
                .filter(|m| m.metadata.get("call_id") == Some(&json!("lookup")))
                .count(),
            1
        );
    });
}

#[test]
fn tool_business_feedback_repairs_only_its_plan_node_and_keeps_successful_calls() {
    block_on(async {
        let probe: Arc<Mutex<Probe>> = Default::default();
        let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
        let evaluator_probe = probe.clone();
        let agent = crate::agent::react()
            .model(Model(probe.clone()))
            .toolkit(
                Toolkit::new()
                    .tool(CheckedTool::new("lookup", probe.clone()))
                    .tool(CheckedTool::new("clock", probe.clone())),
            )
            .with_id_fn(next_id)
            .with_max_concurrency(2)
            .checkpoint(checkpoint.clone())
            .evaluator(move |_: Input, result: Messages, _: Cancellation| {
                let probe = evaluator_probe.clone();
                async move {
                    assert_eq!(result.0[0].values["answer"], "final");
                    probe.lock().unwrap().final_evaluations += 1;
                    Ok(Evaluation {
                        passed: true,
                        feedback: None,
                    })
                }
            })
            .build()
            .unwrap();
        let input = Input {
            metadata: part_data(json!({"chat_id":600})),
            payload: vec![],
        };
        let events = agent
            .run(input, true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let snapshot = checkpoint.load(600, &Cancellation::new()).await.unwrap();
        let probe = probe.lock().unwrap();
        assert_eq!(probe.final_evaluations, 1);
        assert_eq!(probe.calls.len(), 4);
        assert!(probe.calls.values().all(|count| *count == 1));
        assert!(probe.evaluations.values().all(|count| *count == 1));
        assert_eq!(probe.rounds["a"], 2);
        assert_eq!(probe.rounds["b"], 2);
        let a_requests = probe
            .requests
            .iter()
            .filter(|(id, _)| id.as_deref() == Some("a"))
            .collect::<Vec<_>>();
        assert!(has_feedback(&a_requests[1].1.messages));
        for (node, request) in &probe.requests {
            if node.as_deref() != Some("a") {
                assert!(!has_feedback(&request.messages));
            }
        }
        assert_eq!(snapshot["status"], "succeeded");
        assert_eq!(snapshot["usage"]["total"], 24);
        for id in ["a", "b", "c"] {
            assert_eq!(snapshot["nodes"][id]["status"], "succeeded");
            assert!(snapshot["nodes"][id]["execution"]["error"].is_null());
        }
        let saved: Messages = serde_json::from_value(
            snapshot["nodes"]["a"]["execution"]["attempts"][0]["messages"].clone(),
        )
        .unwrap();
        assert!(
            has_feedback(&saved),
            "the plan node must retain actual evaluation feedback"
        );
        let output: Messages =
            serde_json::from_value(snapshot["nodes"]["a"]["messages"].clone()).unwrap();
        assert!(
            !has_feedback(&output),
            "feedback is internal process data, not downstream output"
        );
    });
}

#[test]
fn tool_eval_technical_failure_keeps_call_results_and_finishes_independent_nodes() {
    block_on(async {
        let probe = Arc::new(Mutex::new(Probe {
            eval_error: true,
            ..Default::default()
        }));
        let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
        let agent = crate::agent::react()
            .model(Model(probe.clone()))
            .toolkit(
                Toolkit::new()
                    .tool(CheckedTool::new("lookup", probe.clone()))
                    .tool(CheckedTool::new("clock", probe.clone())),
            )
            .with_id_fn(next_id)
            .with_max_concurrency(2)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let events = agent
            .run(
                Input {
                    metadata: part_data(json!({"chat_id":601})),
                    payload: vec![],
                },
                true,
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let errors = events
            .iter()
            .filter_map(|event| event.as_ref().err())
            .collect::<Vec<_>>();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code, "TIMEOUT");
        assert_eq!(errors[0].message, "eval service unavailable");
        assert!(events.iter().any(|event| matches!(event,
            Ok(Event::Complete { message, .. }) if message.values.get("result") == Some(&json!("lookup-a"))
        )));
        let snapshot = checkpoint.load(601, &Cancellation::new()).await.unwrap();
        let probe = probe.lock().unwrap();
        assert_eq!(probe.calls[&("lookup".into(), "a".into())], 1);
        assert_eq!(probe.evaluations[&("lookup".into(), "a".into())], 1);
        assert_eq!(probe.rounds["a"], 1);
        assert_eq!(probe.rounds["b"], 2);
        assert!(
            !probe
                .requests
                .iter()
                .any(|(node, _)| node.as_deref() == Some("c"))
        );
        assert_eq!(snapshot["status"], "failed");
        assert_eq!(snapshot["nodes"]["a"]["status"], "failed");
        assert_eq!(
            snapshot["nodes"]["a"]["execution"]["error"]["code"],
            "TIMEOUT"
        );
        assert_eq!(snapshot["nodes"]["b"]["status"], "succeeded");
        let saved: Messages = serde_json::from_value(
            snapshot["nodes"]["a"]["execution"]["attempts"][0]["messages"].clone(),
        )
        .unwrap();
        assert!(
            saved
                .0
                .iter()
                .any(|message| message.values.get("result") == Some(&json!("lookup-a")))
        );
        assert!(!has_feedback(&saved));
        assert!(
            events
                .iter()
                .filter_map(|event| event.as_ref().ok())
                .all(|event| {
                    let message = match event {
                        Event::Delta { message } | Event::Complete { message, .. } => message,
                    };
                    !message.values.contains_key("error")
                }),
            "eval errors must not become tool-call repair messages"
        );
    });
}
