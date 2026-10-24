//! Offline example: no network, secrets, Runtime or persistence.

use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::evaluation::{Evaluation, Evaluator as _},
    agent::event::Event,
    agent::message::Role,
    agent::model::chat::{Chat, Request},
    agent::plan::Planner as _,
    agent::{Input, Payload},
    core,
};
use serde_json::json;

struct Model;
impl Chat for Model {
    type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, _request: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        message.values = serde_json::from_value(json!({
            "version": 1,
            "nodes": {
                "research": {"name": "search", "objective": "查询天气"},
                "answer": {"name": "summarize", "objective": "整理查询结果"}
            },
            "edges": [{"from": "research", "to": "answer"}]
        }))
        .unwrap();
        Ok(futures_util::stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

// Application business logic directly implements the shared evaluation contract.
async fn check_answer(
    _input: Input,
    result: Messages,
    _cancellation: Cancellation,
) -> Result<Evaluation, Error> {
    Ok(Evaluation {
        passed: result
            .0
            .iter()
            .any(|message| message.values.contains_key("weather")),
        feedback: None,
    })
}

fn main() -> Result<(), Error> {
    futures_executor::block_on(async {
        let cancellation = Cancellation::new();
        let input = Input {
            payload: vec![Payload {
                r#type: "text".into(),
                data: json!({"value": "北京天气怎么样？"}),
                mime_type: None,
            }],
            ..Default::default()
        };
        let planner = core::plan::Planner::new(Model, Request::default())?;
        let plan = planner.plan(&input, None, None, &cancellation).await?;
        println!(
            "{}",
            serde_json::to_string_pretty(&plan)
                .map_err(|error| Error::new("INVALID_RESPONSE", error.to_string()))?
        );

        // This example does not execute the graph or invent a Runtime result.
        // A caller supplies its own final result for independent evaluation.
        let result = Messages::new([Message::function(json!({"weather": "晴"}))]);
        let evaluation = check_answer
            .evaluate(&input, &result, &cancellation)
            .await?;
        println!("passed={}", evaluation.passed);
        Ok::<_, Error>(())
    })
}
