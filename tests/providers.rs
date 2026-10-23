use futures_util::StreamExt;
use halo::{
    event::Event,
    message::{Message, Messages, Role},
    model::{
        chat::{self, Chat as _},
        content::part::Part,
        embed::{self, Embed as _},
    },
    provider::{
        deepseek::{self, DeepSeek},
        openai::{self, OpenAI},
    },
};
use serde_json::{Map, Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
};

// One deterministic request per fixture. No external API or real credentials.
async fn server(
    status: u16,
    content_type: &str,
    body: String,
    fragment: bool,
) -> (String, JoinHandle<(String, Value)>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let content_type = content_type.to_owned();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut data = Vec::new();
        let mut buffer = [0; 2048];
        let (header_end, content_length) = loop {
            let count = socket.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0);
            data.extend_from_slice(&buffer[..count]);
            if let Some(end) = data.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&data[..end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                break (end + 4, length);
            }
        };
        while data.len() < header_end + content_length {
            let count = socket.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0);
            data.extend_from_slice(&buffer[..count]);
        }
        let headers = String::from_utf8(data[..header_end].to_vec()).unwrap();
        let request =
            serde_json::from_slice(&data[header_end..header_end + content_length]).unwrap();
        let response = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        if fragment {
            // Deliberately split JSON and UTF-8 across transport boundaries.
            for bytes in body.as_bytes().chunks(7) {
                socket.write_all(bytes).await.unwrap();
                tokio::task::yield_now().await;
            }
        } else {
            socket.write_all(body.as_bytes()).await.unwrap();
        }
        socket.shutdown().await.unwrap();
        (headers, request)
    });
    (url, task)
}

fn provider(url: String) -> OpenAI {
    OpenAI::new(openai::Config {
        base_url: url,
        api_key: "fixture-secret".into(),
        provider_id: 11,
        ..Default::default()
    })
    .unwrap()
}

fn config() -> chat::Config {
    chat::Config {
        model_name: "test-chat".into(),
        options: Map::new(),
    }
}
fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": value}).as_object().unwrap().clone(),
    });
    message
}
fn request(stream: bool) -> chat::Request {
    chat::Request {
        messages: Messages::new([text(Role::User, "北京天气怎么样")]),
        stream,
        ..Default::default()
    }
}
fn response(content: &str) -> Value {
    json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
}
fn sse(values: Vec<Value>, done: bool) -> String {
    let mut body = ": heartbeat\r\n\r\n".to_owned();
    for value in values {
        body.push_str(&format!("data: {value}\r\n\r\n"));
    }
    if done {
        body.push_str("data: [DONE]\r\n\r\n");
    }
    body
}
fn chunk(delta: Value, finish: Option<&str>) -> Value {
    json!({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
}

#[tokio::test]
async fn openai_nonstream_request_options_and_message_boundaries() {
    let (url, wire) = server(200, "application/json", response("晴天").to_string(), false).await;
    let mut config = config();
    config.options = serde_json::from_value(json!({"seed": 1, "model": "wrong", "api_key": "never-send", "temperature": 2, "unknown": 1})).unwrap();
    let chat = provider(url).chat(config.clone()).unwrap();
    let mut request = request(false);
    request.temperature = Some(0.5);
    request.max_tokens = Some(123);
    request.options.insert("seed".into(), json!(2));
    request.messages.0[0]
        .values
        .insert("private".into(), json!("business-data"));
    request.messages.0[0]
        .metadata
        .insert("chat_id".into(), json!(123));
    // The concrete native async implementation is usable on a multithread runtime.
    let events = tokio::spawn(async move {
        chat.stream(request)
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    })
    .await
    .unwrap();
    assert_eq!(events.len(), 1);
    let Event::Complete {
        message,
        usage,
        finish_reason,
    } = events[0].as_ref().unwrap()
    else {
        panic!("expected Complete")
    };
    assert_eq!(message.content[0].data["value"], "晴天");
    assert_eq!(message.usage.as_ref(), usage.as_ref());
    assert_eq!(usage.as_ref().unwrap().total, Some(15));
    assert_eq!(finish_reason.as_deref(), Some("stop"));
    assert_eq!(message.metadata["provider_id"], 11);
    let (headers, body) = wire.await.unwrap();
    assert!(headers.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(
        headers
            .to_lowercase()
            .contains("authorization: bearer fixture-secret")
    );
    assert_eq!(body["model"], "test-chat");
    assert_eq!(body["seed"], 2);
    assert_eq!(body["temperature"], 0.5);
    assert_eq!(body["max_completion_tokens"], 123);
    for key in ["api_key", "unknown", "max_tokens"] {
        assert!(body.get(key).is_none());
    }
    assert!(!body.to_string().contains("business-data"));
    assert!(!body["messages"][0].to_string().contains("chat_id"));
    assert_eq!(config.options["seed"], 1);
}

#[tokio::test]
async fn sse_fragmented_unicode_and_usage_after_finish_are_complete_once() {
    let body = sse(
        vec![
            chunk(json!({"role":"assistant"}), None),
            chunk(json!({"content":"北京"}), None),
            chunk(json!({"content":"晴天"}), None),
            chunk(json!({}), Some("stop")),
            json!({"choices":[], "usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}),
        ],
        true,
    );
    let (url, wire) = server(200, "text/event-stream; charset=utf-8", body, true).await;
    let events = provider(url)
        .chat(config())
        .unwrap()
        .stream(request(true))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(events.len(), 3);
    assert!(matches!(&events[0], Ok(Event::Delta { .. })));
    let Event::Complete { message, usage, .. } = events[2].as_ref().unwrap() else {
        panic!()
    };
    assert_eq!(message.content[0].data["value"], "北京晴天");
    let usage = usage.as_ref().unwrap();
    assert_eq!(usage.total, Some(14));
    assert_eq!(usage.input_details.as_ref().unwrap().cached, Some(3));
    assert_eq!(usage.output_details.as_ref().unwrap().think, Some(2));
    assert_eq!(
        wire.await.unwrap().1["stream_options"]["include_usage"],
        true
    );
}

#[tokio::test]
async fn deepseek_think_and_interleaved_tool_arguments_use_shared_events() {
    let body = sse(
        vec![
            chunk(json!({"reasoning_content":"先查天气"}), None),
            chunk(
                json!({"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"weather","arguments":"{\"city\":"}},{"index":1,"id":"call_b","type":"function","function":{"name":"search","arguments":"{\"q\":\""}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls":[{"index":1,"function":{"arguments":"露营\"}"}},{"index":0,"function":{"arguments":"\"北京\"}"}}]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
        ],
        true,
    );
    let (url, wire) = server(200, "text/event-stream", body, true).await;
    let provider = DeepSeek::new(deepseek::Config {
        base_url: url,
        api_key: "fixture-secret".into(),
        ..Default::default()
    })
    .unwrap();
    let mut request = request(true);
    request.max_tokens = Some(100);
    let events = provider
        .chat(config())
        .unwrap()
        .stream(request)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let Event::Complete {
        message,
        finish_reason,
        ..
    } = events.last().unwrap().as_ref().unwrap()
    else {
        panic!()
    };
    assert_eq!(message.content[0].r#type, "think");
    assert_eq!(message.content[0].data["value"], "先查天气");
    assert_eq!(message.content[1].data["arguments"], json!({"city":"北京"}));
    assert_eq!(message.content[2].data["arguments"], json!({"q":"露营"}));
    assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
    let body = wire.await.unwrap().1;
    assert_eq!(body["messages"][0]["content"], "北京天气怎么样");
    assert_eq!(body["max_tokens"], 100);
}

#[tokio::test]
async fn truncated_stream_has_one_terminal_error_and_no_complete_or_replay() {
    let (url, wire) = server(
        200,
        "text/event-stream",
        sse(vec![chunk(json!({"content":"部分结果"}), None)], false),
        false,
    )
    .await;
    let mut events = provider(url)
        .chat(config())
        .unwrap()
        .stream(request(true))
        .await
        .unwrap();
    assert!(matches!(events.next().await, Some(Ok(Event::Delta { .. }))));
    assert_eq!(
        events.next().await.unwrap().unwrap_err().code,
        "INVALID_RESPONSE"
    );
    assert!(events.next().await.is_none());
    assert!(events.next().await.is_none());
    wire.await.unwrap();
}

#[tokio::test]
async fn done_without_finish_and_malformed_json_are_terminal_errors() {
    for body in [
        sse(vec![chunk(json!({"content":"x"}), None)], true),
        "data: invalid-json\n\ndata: [DONE]\n\n".into(),
    ] {
        let (url, wire) = server(200, "text/event-stream", body, false).await;
        let events = provider(url)
            .chat(config())
            .unwrap()
            .stream(request(true))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert_eq!(
            events.last().unwrap().as_ref().unwrap_err().code,
            "INVALID_RESPONSE"
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Ok(Event::Complete { .. })))
        );
        wire.await.unwrap();
    }
}

#[tokio::test]
async fn malformed_complete_tool_calls_return_errors_without_panicking() {
    for calls in [
        json!([1]),
        json!([{"id":"c","type":"function","function":{"name":7,"arguments":"{}"}}]),
        json!([{"id":"c","type":"function","function":{"name":"f","arguments":{}}}]),
        json!([{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]),
    ] {
        let value = json!({"choices":[{"message":{"role":"assistant","tool_calls":calls},"finish_reason":"tool_calls"}]});
        let (url, wire) = server(200, "application/json", value.to_string(), false).await;
        assert_eq!(
            provider(url)
                .chat(config())
                .unwrap()
                .stream(request(false))
                .await
                .err()
                .unwrap()
                .code,
            "INVALID_RESPONSE"
        );
        wire.await.unwrap();
    }
}

#[tokio::test]
async fn malformed_arguments_remain_original_text() {
    let value = json!({"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{broken"}}]},"finish_reason":"tool_calls"}]});
    let (url, wire) = server(200, "application/json", value.to_string(), false).await;
    let mut events = provider(url)
        .chat(config())
        .unwrap()
        .stream(request(false))
        .await
        .unwrap();
    let Event::Complete { message, .. } = events.next().await.unwrap().unwrap() else {
        panic!()
    };
    assert_eq!(message.content[0].data["arguments"], "{broken");
    wire.await.unwrap();
}

#[tokio::test]
async fn structured_output_keeps_text_and_only_populates_valid_values() {
    for (content, valid) in [
        ("{\"answer\":\"晴天\"}", true),
        ("{\"answer\":123}", false),
        ("not json", false),
    ] {
        let (url, wire) = server(
            200,
            "application/json",
            response(content).to_string(),
            false,
        )
        .await;
        let mut request = request(false);
        request.response_format = Some(chat::ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}),
            strict: true,
        });
        let mut events = provider(url)
            .chat(config())
            .unwrap()
            .stream(request)
            .await
            .unwrap();
        let Event::Complete { message, .. } = events.next().await.unwrap().unwrap() else {
            panic!()
        };
        assert_eq!(!message.values.is_empty(), valid);
        assert_eq!(message.content[0].data["value"], content);
        assert_eq!(
            wire.await.unwrap().1["response_format"]["type"],
            "json_schema"
        );
    }
}

#[tokio::test]
async fn http_errors_are_uniform_and_do_not_echo_provider_secrets() {
    for (status, code) in [
        (400, "INVALID_ARGUMENTS"),
        (401, "AUTHENTICATION"),
        (403, "PERMISSION_DENIED"),
        (408, "TIMEOUT"),
        (429, "RATE_LIMITED"),
        (500, "UNAVAILABLE"),
    ] {
        let (url, wire) = server(
            status,
            "application/json",
            json!({"error":{"message":"fixture-secret private-prompt"}}).to_string(),
            false,
        )
        .await;
        let error = provider(url)
            .chat(config())
            .unwrap()
            .stream(request(false))
            .await
            .err()
            .unwrap();
        assert_eq!(error.code, code);
        let diagnostic = format!("{error:?}");
        assert!(!diagnostic.contains("fixture-secret"));
        assert!(!diagnostic.contains("private-prompt"));
        wire.await.unwrap();
    }
}

#[tokio::test]
async fn context_length_error_preserves_standard_code() {
    let (url, wire) = server(
        400,
        "application/json",
        json!({"error":{"code":"context_length_exceeded"}}).to_string(),
        false,
    )
    .await;
    assert_eq!(
        provider(url)
            .chat(config())
            .unwrap()
            .stream(request(false))
            .await
            .err()
            .unwrap()
            .code,
        "CONTEXT_LENGTH_EXCEEDED"
    );
    wire.await.unwrap();
}

#[tokio::test]
async fn timeout_and_connection_failure_do_not_retry() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let pending = tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.unwrap();
        std::future::pending::<()>().await;
    });
    let client = OpenAI::new(openai::Config {
        base_url: url,
        api_key: "fixture-secret".into(),
        timeout: 1,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        client
            .chat(config())
            .unwrap()
            .stream(request(false))
            .await
            .err()
            .unwrap()
            .code,
        "TIMEOUT"
    );
    pending.abort();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    assert_eq!(
        provider(url)
            .chat(config())
            .unwrap()
            .stream(request(false))
            .await
            .err()
            .unwrap()
            .code,
        "TRANSPORT"
    );
}

#[tokio::test]
async fn openai_single_embed_returns_message_with_metadata_and_usage() {
    let (url,wire) = server(200,"application/json",json!({"data":[{"index":0,"embedding":[0.1,-0.2]}],"usage":{"prompt_tokens":3,"total_tokens":3}}).to_string(),false).await;
    let embed = provider(url)
        .embed(embed::Config {
            model_name: "test-embed".into(),
            options: Map::new(),
        })
        .unwrap();
    let message = embed
        .embed(embed::Request {
            input: "北京".into(),
            dimensions: Some(2),
            options: serde_json::from_value(json!({"user":"caller","input":"cannot-override"}))
                .unwrap(),
        })
        .await
        .unwrap();
    assert_eq!(message.metadata["dimension"], 2);
    assert_eq!(message.values["vector"].as_array().unwrap().len(), 2);
    assert_eq!(message.usage.as_ref().unwrap().input, Some(3));
    assert_eq!(message.usage.as_ref().unwrap().output, None);
    let (headers, body) = wire.await.unwrap();
    assert!(headers.starts_with("POST /v1/embeddings"));
    assert_eq!(body["input"], "北京");
    assert_eq!(body["dimensions"], 2);
    assert_eq!(body["encoding_format"], "float");
}

#[tokio::test]
async fn embed_rejects_wrong_dimensions_multiple_results_and_invalid_vectors() {
    for data in [
        json!([{"index":0,"embedding":[0.1]}]),
        json!([{"index":0,"embedding":[0.1,0.2]},{"index":1,"embedding":[0.1,0.2]}]),
        json!([{"index":0,"embedding":[1e100,0]}]),
        json!([{"index":0,"embedding":[null,0]}]),
    ] {
        let (url, wire) = server(
            200,
            "application/json",
            json!({"data":data}).to_string(),
            false,
        )
        .await;
        let embed = provider(url)
            .embed(embed::Config {
                model_name: "test-embed".into(),
                options: Map::new(),
            })
            .unwrap();
        assert_eq!(
            embed
                .embed(embed::Request {
                    input: "x".into(),
                    dimensions: Some(2),
                    options: Map::new()
                })
                .await
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
        wire.await.unwrap();
    }
}

#[tokio::test]
async fn unsupported_deepseek_capabilities_fail_before_network() {
    let provider = DeepSeek::new(deepseek::Config {
        base_url: "http://127.0.0.1:1".into(),
        api_key: "fixture-secret".into(),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        provider
            .embed(embed::Config {
                model_name: "embed".into(),
                options: Map::new()
            })
            .err()
            .unwrap()
            .code,
        "UNSUPPORTED"
    );
    let mut request = request(false);
    request.response_format = Some(chat::ResponseFormat::JsonSchema {
        name: "x".into(),
        schema: json!({"type":"object"}),
        strict: true,
    });
    assert_eq!(
        provider
            .chat(config())
            .unwrap()
            .stream(request)
            .await
            .err()
            .unwrap()
            .code,
        "UNSUPPORTED"
    );
    let mut request = self::request(false);
    request.tools.push(chat::ToolDefinition {
        name: "weather".into(),
        description: "Weather".into(),
        parameters: serde_json::from_value(json!({"type":"object"})).unwrap(),
    });
    request.tool_choice = Some(chat::ToolChoice::Required);
    assert_eq!(
        provider
            .chat(config())
            .unwrap()
            .stream(request)
            .await
            .err()
            .unwrap()
            .code,
        "UNSUPPORTED"
    );
}

#[test]
fn provider_configuration_rejects_unsafe_urls_and_redacts_credentials() {
    let config = openai::Config {
        api_key: "fixture-secret".into(),
        base_url: "https://example.com/private-path".into(),
        ..Default::default()
    };
    let debug = format!("{config:?}");
    assert!(!debug.contains("fixture-secret"));
    assert!(!debug.contains("private-path"));
    for url in [
        "file:///tmp/x",
        "https://name:secret@example.com",
        "https://example.com?api_key=secret",
        "https://example.com#secret",
    ] {
        assert_eq!(
            OpenAI::new(openai::Config {
                base_url: url.into(),
                api_key: "fixture-secret".into(),
                ..Default::default()
            })
            .err()
            .unwrap()
            .code,
            "INVALID_ARGUMENTS"
        );
    }
    assert_eq!(
        OpenAI::new(openai::Config::default()).err().unwrap().code,
        "AUTHENTICATION"
    );
}
