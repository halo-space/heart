//! A plain async Rust function adapted to Heart's existing Tool contract.

use halo::Cancellation;
use halo::agent::{Error, tool::Call};
use serde_json::json;

#[halo::tool(description = "查询指定城市天气")]
async fn get_weather(location: String, cancellation: &Cancellation) -> Result<String, Error> {
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "weather request cancelled"));
    }
    Ok(format!("{location} 今天晴，26°C。"))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let call = Call {
        call_id: "call_001".into(),
        name: "get_weather".into(),
        arguments: serde_json::Map::from_iter([(String::from("location"), json!("北京"))]),
    };
    let (message, _) =
        halo::agent::tool::execute(&get_weather(), call, &Cancellation::new()).await?;
    println!("{}", message.values["value"]);
    Ok(())
}
