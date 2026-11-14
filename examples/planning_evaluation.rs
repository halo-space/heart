//! Real DeepSeek Planner + application Evaluator example.
//!
//! The Planner uses the configured remote model. The Evaluator remains an
//! application callback and is shown with a sample result after planning.

use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::core,
    agent::evaluation::{Evaluation, Evaluator as _},
    agent::model::chat::Request,
    agent::plan::Planner as _,
    agent::provider::deepseek::DeepSeek,
    agent::{Input, Payload},
};
use serde_json::json;

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

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenvy::dotenv().ok();
    let cancellation = Cancellation::new();
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value": "北京天气怎么样？"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let provider =
        DeepSeek::builder()
            .base_url(
                std::env::var("DEEPSEEK_BASE_URL")
                    .unwrap_or_else(|_| "https://api.deepseek.com".into()),
            )
            .api_key(std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
                Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required in .env")
            })?)
            .build()?;
    let chat = provider.chat(halo::agent::model::chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    let planner = core::plan::Planner::new(chat, Request::default())?;
    let plan = planner.plan(&input, None, None, &cancellation).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&plan)
            .map_err(|error| Error::new("INVALID_RESPONSE", error.to_string()))?
    );

    // The plan is real model output. The result shown here is the kind of
    // application-owned message that a later execution stage evaluates.
    let result = Messages::new([Message::function(json!({"weather": "晴"}))]);
    let evaluation = check_answer
        .evaluate(&input, &result, &cancellation)
        .await?;
    println!("passed={}", evaluation.passed);
    Ok(())
}
