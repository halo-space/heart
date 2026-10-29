//! Existing AgentGraph and component contracts exercised together. Execution
//! IDs and input bindings are supplied explicitly by the caller; this is not
//! an assertion that ReAct's automatic DAG dispatch is already integrated.
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use futures_executor::block_on;
use futures_util::{StreamExt, poll, stream::FuturesUnordered};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::function::Function,
    agent::plan::Plan,
    agent::runtime::agent::{
        graph::{Graph, Start},
        node::Status,
    },
    agent::tool::{self, Call, Tool, ToolDefinition},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Notify;

struct Weather {
    definition: ToolDefinition,
    started: Arc<AtomicUsize>,
    beijing: Arc<Notify>,
    shanghai: Arc<Notify>,
}

impl Weather {
    fn new() -> Self {
        Self {
            definition: ToolDefinition {
                name: "get_weather".into(),
                description: "Return test weather for one city".into(),
                parameters: serde_json::from_value(json!({
                    "type":"object", "properties":{"city":{"type":"string"}},
                    "required":["city"], "additionalProperties":false
                }))
                .unwrap(),
            },
            started: Default::default(),
            beijing: Default::default(),
            shanghai: Default::default(),
        }
    }
}

impl Tool for Weather {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.started.fetch_add(1, Ordering::SeqCst);
        let city = call.arguments["city"].as_str().unwrap();
        let condition = if city == "北京" {
            self.beijing.notified().await;
            "sunny"
        } else {
            self.shanghai.notified().await;
            "rainy"
        };
        Ok(Message::function(
            json!({"weather":{"city":city,"condition":condition}}),
        ))
    }
}

fn plan() -> Plan {
    serde_json::from_value(json!({
        "version":1,
        "nodes":{
            "beijing":{"name":"get_weather","objective":"查询北京天气"},
            "shanghai":{"name":"get_weather","objective":"查询上海天气"},
            "advice":{"name":"camping_advice","objective":"比较两地露营条件"}
        },
        "edges":[{"from":"beijing","to":"advice"},{"from":"shanghai","to":"advice"}]
    }))
    .unwrap()
}

fn call(city: &str) -> Call {
    Call {
        call_id: format!("weather_{city}"),
        name: "get_weather".into(),
        arguments: serde_json::from_value(json!({"city":city})).unwrap(),
    }
}

#[derive(Deserialize, Serialize)]
struct WeatherData {
    city: String,
    condition: String,
}
struct Advice {
    beijing: WeatherData,
    shanghai: WeatherData,
}

#[test]
fn concurrent_same_capability_nodes_feed_a_typed_function_without_merging_messages() {
    block_on(async {
        let original = plan();
        let mut graph = Graph::new(original.clone()).unwrap();
        graph
            .start_batch(
                2,
                &[
                    Start {
                        node_id: "beijing".into(),
                        exec_id: 101,
                    },
                    Start {
                        node_id: "shanghai".into(),
                        exec_id: 102,
                    },
                ],
            )
            .unwrap();
        let weather = Weather::new();
        let cancellation = Cancellation::new();
        let mut jobs = FuturesUnordered::new();
        for (node_id, city) in [("beijing", "北京"), ("shanghai", "上海")] {
            let weather = &weather;
            let cancellation = &cancellation;
            jobs.push(async move {
                (
                    node_id,
                    tool::execute(weather, call(city), cancellation).await,
                )
            });
        }
        assert!(poll!(jobs.next()).is_pending());
        assert_eq!(weather.started.load(Ordering::SeqCst), 2);
        weather.shanghai.notify_one();
        let (node_id, result) = jobs.next().await.unwrap();
        assert_eq!(node_id, "shanghai");
        let (message, _) = result.unwrap();
        graph
            .succeed(node_id, Messages::new([message]), None)
            .unwrap();
        assert_eq!(graph.state("advice").unwrap().status, Status::Pending);
        assert_eq!(
            graph.source_messages("advice").unwrap_err().code,
            "INVALID_STATE"
        );

        weather.beijing.notify_one();
        let (node_id, result) = jobs.next().await.unwrap();
        let (message, _) = result.unwrap();
        graph
            .succeed(node_id, Messages::new([message]), None)
            .unwrap();
        assert_eq!(graph.ready_nodes(), vec!["advice"]);
        let sources: BTreeMap<_, _> = graph.source_messages("advice").unwrap();
        // Explicit business binding: each field names its actual logical
        // source and source values field, never the shared capability name.
        let input = Advice {
            beijing: serde_json::from_value(
                sources["beijing"].as_slice()[0].values["weather"].clone(),
            )
            .unwrap(),
            shanghai: serde_json::from_value(
                sources["shanghai"].as_slice()[0].values["weather"].clone(),
            )
            .unwrap(),
        };
        let advice = |input: Advice, _: Cancellation| async move {
            Ok::<_, Error>(json!({"beijing":input.beijing,"shanghai":input.shanghai}))
        };
        graph.start("advice", 103).unwrap();
        let output = Function::call(&advice, input, &cancellation).await.unwrap();
        graph
            .succeed("advice", Messages::new([Message::function(output)]), None)
            .unwrap();
        assert!(graph.is_complete());
        let result = &graph
            .state("advice")
            .unwrap()
            .messages
            .as_ref()
            .unwrap()
            .as_slice()[0];
        assert_eq!(result.values["beijing"]["city"], "北京");
        assert_eq!(result.values["shanghai"]["city"], "上海");
        assert!(!result.values.contains_key("tool_result"));
        assert_eq!(graph.plan, original);
        assert_eq!(weather.started.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn invalid_tool_input_records_direct_error_and_preserves_independent_branch() {
    block_on(async {
        let mut graph = Graph::new(plan()).unwrap();
        graph.start("beijing", 101).unwrap();
        let weather = Weather::new();
        let mut invalid = call("北京");
        invalid.arguments.clear();
        let error = tool::execute(&weather, invalid, &Cancellation::new())
            .await
            .unwrap_err();
        graph.fail_with("beijing", Some(error.clone())).unwrap();
        assert_eq!(weather.started.load(Ordering::SeqCst), 0);
        let state = graph.state("beijing").unwrap();
        assert!(state.messages.is_none());
        assert_eq!(state.execution.as_ref().unwrap().error, Some(error));
        assert_eq!(graph.ready_nodes(), vec!["shanghai"]);
        assert_eq!(
            graph.source_messages("advice").unwrap_err().code,
            "INVALID_STATE"
        );
        graph.start("shanghai", 102).unwrap();
        weather.shanghai.notify_one();
        let (message, _) = tool::execute(&weather, call("上海"), &Cancellation::new())
            .await
            .unwrap();
        graph
            .succeed("shanghai", Messages::new([message]), None)
            .unwrap();
        assert_eq!(graph.state("advice").unwrap().status, Status::Pending);
        assert!(!graph.is_complete());
    });
}
