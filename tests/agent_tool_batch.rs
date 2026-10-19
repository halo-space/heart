use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use futures_executor::block_on;
use futures_util::{FutureExt, StreamExt, poll, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, toolkit::Toolkit},
    event::Event,
    message::{Messages, Role},
    model::{
        chat::{Chat, Request},
        content::part::Part,
    },
    tool::{Call, Evaluation, Tool, ToolDefinition},
};
use serde_json::json;
use tokio::sync::oneshot;

#[derive(Default)]
struct Probe {
    started: Vec<usize>,
    evaluated: Vec<usize>,
    active: usize,
    peak: usize,
    calls: BTreeMap<usize, oneshot::Receiver<()>>,
    evals: BTreeMap<usize, oneshot::Receiver<()>>,
    started_signal: Option<oneshot::Sender<()>>,
}
type Shared = Arc<Mutex<Probe>>;

struct Active(Shared);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.lock().unwrap().active -= 1;
    }
}

struct BatchTool {
    definition: ToolDefinition,
    probe: Shared,
    fail_call: Option<usize>,
    fail_eval: Option<usize>,
    reject_eval: Option<usize>,
}
impl Tool for BatchTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        let index = call.arguments["index"].as_u64().unwrap() as usize;
        let (receiver, signal) = {
            let mut probe = self.probe.lock().unwrap();
            probe.started.push(index);
            probe.active += 1;
            probe.peak = probe.peak.max(probe.active);
            let signal = if index == 1 {
                probe.started_signal.take()
            } else {
                None
            };
            (probe.calls.remove(&index), signal)
        };
        let _active = Active(self.probe.clone());
        if let Some(signal) = signal {
            let _ = signal.send(());
        }
        if let Some(receiver) = receiver {
            receiver.await.unwrap();
        }
        if self.fail_call == Some(index) {
            return Err(
                Error::new("BACKEND", "original tool error").with_details("retained details")
            );
        }
        Ok(Message::function(json!({"index":index})))
    }
    async fn eval(&self, call: &Call, _: &Message, _: &Cancellation) -> Result<Evaluation, Error> {
        let index = call.arguments["index"].as_u64().unwrap() as usize;
        let receiver = {
            let mut probe = self.probe.lock().unwrap();
            probe.evaluated.push(index);
            probe.evals.remove(&index)
        };
        if let Some(receiver) = receiver {
            receiver.await.unwrap();
        }
        if self.fail_eval == Some(index) {
            return Err(Error::new("BACKEND", "eval failed"));
        }
        if self.reject_eval == Some(index) {
            return Ok(Evaluation::failed(Messages::new([Message::function(
                json!({"feedback":"business rejection"}),
            )])));
        }
        Ok(Evaluation::passed())
    }
}

#[derive(Clone)]
struct Model {
    count: usize,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        // The Chat contract does not require a Send creation future.
        let local = std::rc::Rc::new(());
        std::future::ready(()).await;
        let _ = local;
        let first = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len() % 2 == 1
        };
        let mut message = Message::new(Role::Assistant);
        if first {
            for index in 1..=self.count {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":format!("call_{index}"),"name":"batch","arguments":{"index":index}}).as_object().unwrap().clone(),
                });
            }
        } else {
            message.values.insert("answer".into(), json!("done"));
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

fn setup(count: usize) -> (Model, BatchTool, Shared) {
    let probe: Shared = Default::default();
    let model = Model {
        count,
        requests: Default::default(),
    };
    let tool = BatchTool {
        definition: ToolDefinition {name:"batch".into(), description:"test batch".into(),
            parameters: serde_json::from_value(json!({"type":"object","properties":{"index":{"type":"integer"}},"required":["index"],"additionalProperties":false})).unwrap()},
        probe:probe.clone(), fail_call:None, fail_eval:None, reject_eval:None,
    };
    (model, tool, probe)
}

fn gates(probe: &Shared, count: usize) -> BTreeMap<usize, oneshot::Sender<()>> {
    (1..=count)
        .map(|index| {
            let (sender, receiver) = oneshot::channel();
            probe.lock().unwrap().calls.insert(index, receiver);
            (index, sender)
        })
        .collect()
}
fn message(event: Result<Event, Error>) -> Message {
    match event.unwrap() {
        Event::Complete { message, .. } => message,
        _ => panic!("expected Complete"),
    }
}
fn input() -> Input {
    Input {
        metadata: Default::default(),
        payload: vec![agent::Payload {
            r#type: "text".into(),
            data: json!({"value":"test"}),
            mime_type: None,
        }],
    }
}

#[test]
fn concurrency_configuration_is_validated_and_survives_model_binding() {
    let (model, _, _) = setup(0);
    assert_eq!(
        agent::react()
            .model(model.clone())
            .build()
            .unwrap()
            .max_concurrency(),
        100
    );
    assert_eq!(
        agent::react()
            .with_max_concurrency(7)
            .model(model.clone())
            .build()
            .unwrap()
            .max_concurrency(),
        7
    );
    assert_eq!(
        agent::react()
            .model(model)
            .with_max_concurrency(0)
            .build()
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn bounded_overlap_emits_completed_results_but_model_sees_original_order() {
    block_on(async {
        let (model, tool, probe) = setup(3);
        let requests = model.requests.clone();
        let mut releases = gates(&probe, 3);
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .with_max_concurrency(2)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        message(events.next().await.unwrap());
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().started, vec![1, 2]);
        releases.remove(&2).unwrap().send(()).unwrap();
        assert_eq!(message(next.await.unwrap()).values["index"], 2);
        assert_eq!(requests.lock().unwrap().len(), 1);
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().started, vec![1, 2, 3]);
        releases.remove(&3).unwrap().send(()).unwrap();
        assert_eq!(message(next.await.unwrap()).values["index"], 3);
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(requests.lock().unwrap().len(), 1);
        releases.remove(&1).unwrap().send(()).unwrap();
        assert_eq!(message(next.await.unwrap()).values["index"], 1);
        assert_eq!(
            message(events.next().await.unwrap()).values["answer"],
            "done"
        );
        assert!(events.next().await.is_none());
        let requests = requests.lock().unwrap();
        let results: Vec<_> = requests[1]
            .messages
            .as_slice()
            .iter()
            .filter_map(|m| m.values.get("index"))
            .cloned()
            .collect();
        assert_eq!(results, vec![json!(1), json!(2), json!(3)]);
        let probe = probe.lock().unwrap();
        assert_eq!(probe.peak, 2);
        assert_eq!(probe.active, 0);
        assert_eq!(probe.evaluated.len(), 3);
    });
}

#[test]
fn one_slot_keeps_calls_serial_without_eagerly_starting_the_queue() {
    block_on(async {
        let (model, tool, probe) = setup(2);
        let mut releases = gates(&probe, 2);
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .with_max_concurrency(1)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        message(events.next().await.unwrap());
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().started, vec![1]);
        releases.remove(&1).unwrap().send(()).unwrap();
        assert_eq!(message(next.await.unwrap()).values["index"], 1);
        assert_eq!(probe.lock().unwrap().started, vec![1]);
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().started, vec![1, 2]);
        releases.remove(&2).unwrap().send(()).unwrap();
        assert_eq!(message(next.await.unwrap()).values["index"], 2);
        message(events.next().await.unwrap());
        assert_eq!(probe.lock().unwrap().peak, 1);
    });
}

#[test]
fn pending_eval_holds_the_slot_and_blocks_next_inference() {
    block_on(async {
        let (model, tool, probe) = setup(2);
        let requests = model.requests.clone();
        let (release, receiver) = oneshot::channel();
        probe.lock().unwrap().evals.insert(1, receiver);
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .with_max_concurrency(1)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        message(events.next().await.unwrap());
        assert_eq!(message(events.next().await.unwrap()).values["index"], 1);
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().started, vec![1]);
        assert_eq!(requests.lock().unwrap().len(), 1);
        release.send(()).unwrap();
        assert_eq!(message(next.await.unwrap()).values["index"], 2);
        message(events.next().await.unwrap());
        assert_eq!(probe.lock().unwrap().evaluated, vec![1, 2]);
    });
}

#[test]
fn failed_call_is_direct_context_and_does_not_repeat_successful_siblings() {
    block_on(async {
        let (model, mut tool, probe) = setup(3);
        tool.fail_call = Some(2);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
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
        let error = requests[1]
            .messages
            .as_slice()
            .iter()
            .find(|m| m.values.get("call_id") == Some(&json!("call_2")))
            .unwrap();
        assert_eq!(error.values["error"]["code"], "BACKEND");
        assert_eq!(error.values["error"]["message"], "original tool error");
        assert_eq!(error.values["error"]["details"], "retained details");
        assert!(
            !serde_json::to_string(error)
                .unwrap()
                .contains("tool_result")
        );
        let probe = probe.lock().unwrap();
        assert_eq!(probe.started.len(), 3);
        assert_eq!(probe.evaluated.len(), 2);
    });
}

#[test]
fn cancellation_releases_admitted_futures_without_starting_queued_calls() {
    block_on(async {
        let (model, tool, probe) = setup(3);
        let _releases = gates(&probe, 3);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .with_max_concurrency(2)
            .build()
            .unwrap();
        let cancellation = Cancellation::new();
        let mut events = agent
            .run(input(), true, cancellation.clone())
            .await
            .unwrap();
        message(events.next().await.unwrap());
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().active, 2);
        cancellation.cancel();
        assert!(
            matches!(next.await.unwrap().unwrap(), Event::Complete {finish_reason:Some(reason),..} if reason == "cancelled")
        );
        {
            let probe = probe.lock().unwrap();
            assert_eq!(probe.active, 0);
            assert_eq!(probe.started, vec![1, 2]);
        }
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert!(events.next().await.is_none());
    });
}

#[test]
fn dropping_stream_releases_pending_tool_resources() {
    block_on(async {
        let (model, tool, probe) = setup(2);
        let _releases = gates(&probe, 2);
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        message(events.next().await.unwrap());
        assert!(events.next().now_or_never().is_none());
        assert_eq!(probe.lock().unwrap().active, 2);
        drop(events);
        assert_eq!(probe.lock().unwrap().active, 0);
    });
}

#[test]
fn eval_failure_keeps_success_event_and_never_repeats_call() {
    block_on(async {
        let (model, mut tool, probe) = setup(2);
        tool.fail_eval = Some(1);
        let _releases = gates(&probe, 2);
        // Only call 2 blocks; call 1 finishes and then its eval fails.
        probe.lock().unwrap().calls.remove(&1);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        message(events.next().await.unwrap());
        assert_eq!(message(events.next().await.unwrap()).values["index"], 1);
        let error = events.next().await.unwrap().unwrap_err();
        assert_eq!(error.message, "eval failed");
        assert_eq!(probe.lock().unwrap().active, 0);
        assert_eq!(
            probe
                .lock()
                .unwrap()
                .started
                .iter()
                .filter(|&&i| i == 1)
                .count(),
            1
        );
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert!(events.next().await.is_none());
    });
}

#[test]
fn business_feedback_does_not_masquerade_as_tool_error_context() {
    block_on(async {
        let (model, mut tool, probe) = setup(2);
        tool.reject_eval = Some(1);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        assert_eq!(probe.lock().unwrap().started.len(), 2);
        assert!(
            !serde_json::to_string(&requests.lock().unwrap()[1])
                .unwrap()
                .contains("business rejection")
        );
    });
}

struct Allow;
impl halo_agents::permission::Permission for Allow {
    async fn check(
        &self,
        _: &halo_agents::permission::Request,
        _: &Cancellation,
    ) -> Result<halo_agents::permission::Decision, Error> {
        Ok(halo_agents::permission::Decision {
            behavior: halo_agents::permission::Behavior::Allow,
            reason: None,
        })
    }
}
struct HookFailure(Mutex<Option<oneshot::Receiver<()>>>);
impl halo_agents::middleware::Middleware for HookFailure {
    async fn before_permission(
        &self,
        value: &mut serde_json::Value,
        _: &Cancellation,
    ) -> Result<(), Error> {
        if value["metadata"]["call"]["arguments"]["index"] == json!(2) {
            // Exercise release of an actually running sibling rather than
            // assuming how many internal handoff polls precede admission.
            let started = self.0.lock().unwrap().take();
            if let Some(started) = started {
                started.await.unwrap();
            }
            return Err(Error::new("BACKEND", "permission hook failed"));
        }
        Ok(())
    }
}

#[test]
fn permission_hook_failure_releases_siblings_and_never_enters_model_context() {
    block_on(async {
        let (model, tool, probe) = setup(3);
        let _releases = gates(&probe, 3);
        let (started, receive_started) = oneshot::channel();
        probe.lock().unwrap().started_signal = Some(started);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .permission(Allow)
            .middleware(HookFailure(Mutex::new(Some(receive_started))))
            .with_max_concurrency(2)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        message(events.next().await.unwrap());
        assert_eq!(
            events.next().await.unwrap().unwrap_err().message,
            "permission hook failed"
        );
        {
            let probe = probe.lock().unwrap();
            assert_eq!(probe.started, vec![1]);
            assert_eq!(probe.active, 0);
        }
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert!(events.next().await.is_none());
    });
}

#[test]
fn reusing_agent_does_not_reuse_prior_batch_results_or_pending_work() {
    block_on(async {
        let (model, tool, probe) = setup(3);
        let requests = model.requests.clone();
        let agent = agent::react()
            .model(model)
            .toolkit(Toolkit::new().tool(tool))
            .with_max_concurrency(2)
            .build()
            .unwrap();
        for _ in 0..2 {
            let events: Vec<_> = agent
                .run(input(), true, Cancellation::new())
                .await
                .unwrap()
                .collect()
                .await;
            assert_eq!(events.len(), 5);
            assert!(events.iter().all(Result::is_ok));
        }
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        for index in [1, 3] {
            assert_eq!(
                requests[index]
                    .messages
                    .as_slice()
                    .iter()
                    .filter(|m| m.values.contains_key("index"))
                    .count(),
                3
            );
        }
        assert_eq!(probe.lock().unwrap().started.len(), 6);
        assert_eq!(probe.lock().unwrap().active, 0);
    });
}
