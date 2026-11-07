//! 真实天气问答 Agent。
//!
//! 运行前设置 `DEEPSEEK_API_KEY`，可选设置 `DEEPSEEK_BASE_URL`、
//! `DEEPSEEK_MODEL` 和 `WEATHER_CITY`。模型调用 DeepSeek，天气工具调用
//! Open-Meteo 的真实地理编码和预报接口，不使用预置天气结果。

use std::sync::LazyLock;

use futures_util::StreamExt;
use halo::agent::tool::{Call, Tool, ToolDefinition};
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload};
use halo::agent::{event::Event, message::Message, model::chat, provider::deepseek::DeepSeek};
use reqwest::Client;
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Value, json};

struct Weather;

#[derive(Debug, Deserialize)]
struct Location {
    latitude: f64,
    longitude: f64,
    name: String,
    country: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeocodingResponse {
    results: Option<Vec<Location>>,
}

#[derive(Debug, Deserialize)]
struct Forecast {
    current: Option<CurrentWeather>,
}

#[derive(Debug, Deserialize)]
struct CurrentWeather {
    temperature_2m: Option<f64>,
    relative_humidity_2m: Option<f64>,
    wind_speed_10m: Option<f64>,
    weather_code: Option<i64>,
}

fn url(base: &str, pairs: &[(&str, &str)]) -> Result<Url, Error> {
    let mut url = Url::parse(base)
        .map_err(|error| Error::new("INVALID_ARGUMENTS", format!("invalid URL: {error}")))?;
    {
        let mut query = url.query_pairs_mut();
        for (key, value) in pairs {
            query.append_pair(key, value);
        }
    }
    Ok(url)
}

impl Tool for Weather {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "get_weather".into(),
            description: "查询指定城市当前天气。必须在需要天气信息时调用，不要猜测天气。".into(),
            parameters: serde_json::from_value(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如 Beijing"}
                },
                "required": ["city"],
                "additionalProperties": false
            }))
            .expect("valid weather tool schema"),
        });
        &DEFINITION
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "weather request cancelled"));
        }
        let city = call
            .arguments
            .get("city")
            .and_then(Value::as_str)
            .filter(|city| !city.trim().is_empty())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "city is required"))?;

        let client = Client::new();
        let location_url = url(
            "https://geocoding-api.open-meteo.com/v1/search",
            &[
                ("name", city),
                ("count", "1"),
                ("language", "zh"),
                ("format", "json"),
            ],
        )?;
        let location = client
            .get(location_url)
            .send()
            .await
            .map_err(|error| Error::new("NETWORK", format!("geocoding request failed: {error}")))?
            .error_for_status()
            .map_err(|error| {
                Error::new("NETWORK", format!("geocoding returned an error: {error}"))
            })?
            .json::<GeocodingResponse>()
            .await
            .map_err(|error| {
                Error::new(
                    "INVALID_RESPONSE",
                    format!("invalid geocoding response: {error}"),
                )
            })?
            .results
            .and_then(|mut results| results.pop())
            .ok_or_else(|| Error::new("NOT_FOUND", format!("city not found: {city}")))?;

        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "weather request cancelled"));
        }

        let latitude = location.latitude.to_string();
        let longitude = location.longitude.to_string();
        let forecast_url = url(
            "https://api.open-meteo.com/v1/forecast",
            &[
                ("latitude", &latitude),
                ("longitude", &longitude),
                (
                    "current",
                    "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                ),
                ("timezone", "auto"),
            ],
        )?;
        let forecast = client
            .get(forecast_url)
            .send()
            .await
            .map_err(|error| Error::new("NETWORK", format!("forecast request failed: {error}")))?
            .error_for_status()
            .map_err(|error| Error::new("NETWORK", format!("forecast returned an error: {error}")))?
            .json::<Forecast>()
            .await
            .map_err(|error| {
                Error::new(
                    "INVALID_RESPONSE",
                    format!("invalid forecast response: {error}"),
                )
            })?;

        let current = forecast
            .current
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "forecast has no current weather"))?;
        Ok(Message::function(json!({
            "city": location.name,
            "country": location.country,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "temperature_c": current.temperature_2m,
            "relative_humidity_percent": current.relative_humidity_2m,
            "wind_speed_kmh": current.wind_speed_10m,
            "weather_code": current.weather_code,
            "source": "Open-Meteo"
        })))
    }
}

async fn run() -> Result<(), Error> {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(
            std::env::var("DEEPSEEK_API_KEY")
                .map_err(|_| Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required"))?,
        )
        .build()?;
    let model = provider.chat(
        chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".into()),
        )
        .options(serde_json::Map::from_iter([(
            "thinking".into(),
            json!({"type": "disabled"}),
        )])),
    )?;
    let agent = agent::chat()
        .model(model)
        .system_prompt(
            "你是天气助手。遇到天气问题必须调用 get_weather 获取实时数据，回答时说明城市、温度、湿度和风速；不得猜测或编造天气。",
        )
        .tools(Weather)
        .build()?;
    let city = std::env::var("WEATHER_CITY").unwrap_or_else(|_| "北京".into());
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value": format!("请查询{city}现在的天气，并给出适合出行的建议。")}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => {
                for part in message.content {
                    print!(
                        "{}",
                        part.data
                            .get("value")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                    );
                }
            }
            Event::Complete { usage, .. } => {
                println!("\nusage={usage:?}");
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    run().await
}
