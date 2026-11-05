use futures_executor::block_on;
use halo::agent::tool::Tool;
use halo::agent::{Cancellation, Error, Message};
use serde_json::json;

#[halo::tool(description = "查询指定城市天气")]
async fn get_weather(location: String, cancellation: &Cancellation) -> Result<String, Error> {
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "weather request cancelled"));
    }
    Ok(format!("{location} 今天晴，26°C。"))
}

#[halo::tool(name = "structured_weather", description = "返回结构化天气结果")]
async fn structured_weather(
    location: String,
    cancellation: &Cancellation,
) -> Result<Message, Error> {
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "weather request cancelled"));
    }
    Ok(Message::function(json!({
        "location": location,
        "temperature": 26,
    })))
}

#[test]
fn tool_macro_generates_definition_and_message_result() {
    block_on(async {
        let weather = get_weather();
        assert_eq!(weather.definition().name, "get_weather");
        assert_eq!(weather.definition().description, "查询指定城市天气");
        assert_eq!(weather.definition().parameters["type"], "object");
        assert_eq!(
            weather.definition().parameters["properties"]["location"]["type"],
            "string"
        );

        let call = halo::agent::tool::Call {
            call_id: "call_001".into(),
            name: "get_weather".into(),
            arguments: serde_json::Map::from_iter([(String::from("location"), json!("北京"))]),
        };
        let (message, evaluation) =
            halo::agent::tool::execute(&weather, call, &Cancellation::new())
                .await
                .unwrap();
        assert_eq!(message.values["value"], "北京 今天晴，26°C。");
        assert!(evaluation.passed);
    });
}

#[test]
fn tool_macro_preserves_direct_message_results_and_cancellation() {
    block_on(async {
        let tool = structured_weather();
        let cancellation = Cancellation::new();
        cancellation.cancel();
        let call = halo::agent::tool::Call {
            call_id: "call_002".into(),
            name: "structured_weather".into(),
            arguments: serde_json::Map::from_iter([(String::from("location"), json!("北京"))]),
        };
        let error = halo::agent::tool::execute(&tool, call, &cancellation)
            .await
            .unwrap_err();
        assert_eq!(error.code, "CANCELLED");
    });
}
