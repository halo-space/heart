use super::chat::{Accumulator, complete_response, encode_message, encode_request};
use super::embed::encode_embed_request;
use super::transport::{Kind, parse_usage};
use crate::event::Event;
use crate::message::{Message, Messages, Role};
use crate::model::{
    self,
    chat::{Request, ResponseFormat, ToolChoice},
    content::part::{Part, object as part_data},
};
use serde_json::{Map, Value, json};
#[test]
fn direct_tool_messages_do_not_acquire_a_tool_result_envelope() {
    let mut message = Message::function(json!({"temperature": 22}));
    message.metadata.insert("call_id".into(), json!("call_1"));
    let wire = encode_message(Kind::OpenAI, &message).unwrap();
    assert_eq!(wire["role"], "tool");
    assert_eq!(wire["tool_call_id"], "call_1");
    assert_eq!(
        serde_json::from_str::<Value>(wire["content"].as_str().unwrap()).unwrap(),
        json!({"temperature": 22})
    );
    let mut error = Message::new(Role::Tool);
    error.metadata.insert("call_id".into(), json!("call_1"));
    error.content.push(Part {
        r#type: "text".into(),
        data: part_data(json!({"value": "city is missing"})),
    });
    assert_eq!(
        encode_message(Kind::DeepSeek, &error).unwrap()["content"],
        "city is missing"
    );
    assert!(!wire.to_string().contains("tool_result"));
    assert_eq!(
        encode_message(Kind::OpenAI, &Message::function(json!({"query":"x"})))
            .unwrap_err()
            .code,
        "UNSUPPORTED"
    );
}

#[test]
fn image_uses_nested_source_detail_and_does_not_invent_id_resolution() {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "image".into(),
        data: part_data(
            json!({"source":{"kind":"url","value":"https://example.com/image.png","detail":"low"}}),
        ),
    });
    let wire = encode_message(Kind::OpenAI, &message).unwrap();
    assert_eq!(wire["content"][0]["image_url"]["detail"], "low");
    assert_eq!(
        encode_message(Kind::DeepSeek, &message).unwrap_err().code,
        "UNSUPPORTED"
    );
    message.content[0].data["source"]["kind"] = json!("id");
    assert_eq!(
        encode_message(Kind::OpenAI, &message).unwrap_err().code,
        "UNSUPPORTED"
    );
    message.content[0].data["source"]["kind"] = json!("base64");
    message.content[0].data["source"]["value"] = json!("not-a-data-url");
    assert_eq!(
        encode_message(Kind::OpenAI, &message).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
    message.content[0].data["source"]["value"] = json!("data:image/png;base64,ZmFrZQ==");
    assert_eq!(
        encode_message(Kind::OpenAI, &message).unwrap()["content"][0]["image_url"]["url"],
        "data:image/png;base64,ZmFrZQ=="
    );
}

#[test]
fn deepseek_required_tool_choice_uses_explicit_thinking_options() {
    let config = model::chat::Config {
        model_name: "test".into(),
        context_limit: None,
        options: Map::new(),
    };
    let mut request = Request {
        messages: Messages::new([Message::new(Role::User)]),
        tools: vec![model::chat::ToolDefinition {
            name: "weather".into(),
            description: "weather".into(),
            parameters: serde_json::from_value(json!({"type":"object"})).unwrap(),
        }],
        tool_choice: Some(ToolChoice::Required),
        ..Default::default()
    };
    assert_eq!(
        encode_request(Kind::DeepSeek, &config, &request)
            .unwrap_err()
            .code,
        "UNSUPPORTED"
    );
    request
        .options
        .insert("thinking".into(), json!({"type":"disabled"}));
    assert_eq!(
        encode_request(Kind::DeepSeek, &config, &request).unwrap()["tool_choice"],
        "required"
    );
    request
        .options
        .insert("reasoning_effort".into(), json!("high"));
    assert_eq!(
        encode_request(Kind::DeepSeek, &config, &request)
            .unwrap_err()
            .code,
        "UNSUPPORTED"
    );
    request
        .options
        .insert("reasoning_effort".into(), json!("none"));
    assert!(encode_request(Kind::DeepSeek, &config, &request).is_ok());
}

#[test]
fn vllm_options_and_reasoning_use_the_shared_chat_contract() {
    let config = model::chat::Config {
        model_name: "Qwen3.5-122B-A10B".into(),
        context_limit: None,
        options: Map::new(),
    };
    let request = Request {
        messages: Messages::new([Message::new(Role::User)]),
        stream: true,
        max_tokens: Some(128),
        options: Map::from_iter([
            ("top_k".into(), json!(20)),
            (
                "chat_template_kwargs".into(),
                json!({"enable_thinking": true}),
            ),
        ]),
        ..Default::default()
    };
    let wire = encode_request(Kind::Vllm, &config, &request).unwrap();
    assert_eq!(wire["max_tokens"], 128);
    assert_eq!(wire["top_k"], 20);
    assert_eq!(wire["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(wire["stream_options"]["include_usage"], true);

    let value = json!({
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning": "先分析问题",
                "content": "这是答案"
            },
            "finish_reason": "stop"
        }]
    });
    let Event::Complete { message, .. } = complete_response(&value, Map::new(), None).unwrap()
    else {
        panic!()
    };
    assert_eq!(message.content[0].r#type, "think");
    assert_eq!(message.content[0].data["value"], "先分析问题");
    assert_eq!(message.content[1].r#type, "text");
    assert_eq!(message.content[1].data["value"], "这是答案");
}

#[test]
fn provider_options_are_forwarded_and_reserved_fields_are_rejected() {
    let config = model::chat::Config::new("test");
    let request = Request {
        messages: Messages::new([Message::new(Role::User)]),
        options: Map::from_iter([("provider_extension".into(), json!({"enabled": true}))]),
        ..Default::default()
    };
    let wire = encode_request(Kind::OpenAI, &config, &request).unwrap();
    assert_eq!(wire["provider_extension"]["enabled"], true);

    let request = Request {
        messages: Messages::new([Message::new(Role::User)]),
        options: Map::from_iter([("messages".into(), json!([]))]),
        ..Default::default()
    };
    let error = encode_request(Kind::OpenAI, &config, &request).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert_eq!(error.details.as_deref(), Some("field=messages"));
}

#[test]
fn embed_options_are_forwarded_and_reserved_fields_are_rejected() {
    let config = model::embed::Config {
        model_name: "embed-v1".into(),
        options: Map::new(),
    };
    let request = model::embed::Request {
        input: "北京".into(),
        dimensions: None,
        options: Map::from_iter([("provider_extension".into(), json!("fast"))]),
    };
    let wire = encode_embed_request(Kind::OpenAI, &config, &request).unwrap();
    assert_eq!(wire["provider_extension"], "fast");

    let request = model::embed::Request {
        input: "北京".into(),
        dimensions: None,
        options: Map::from_iter([("input".into(), json!("other"))]),
    };
    let error = encode_embed_request(Kind::OpenAI, &config, &request).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert_eq!(error.details.as_deref(), Some("field=input"));
}

#[test]
fn malformed_stream_tool_fields_and_nonassistant_role_are_rejected() {
    for delta in [
        json!({"role":"user"}),
        json!({"tool_calls":[{"index":0,"id":7}]}),
        json!({"tool_calls":[{"index":0,"function":[]}]}),
        json!({"tool_calls":[{"index":0,"function":{"arguments":7}}]}),
    ] {
        assert_eq!(
            Accumulator::new(Map::new())
                .delta(&json!({"choices":[{"index":0,"delta":delta}]}))
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
    }
}

#[test]
fn json_object_fallback_and_usage_missing_values_remain_explicit() {
    for (content, expected) in [
        ("{\"x\":1}", json!({"x":1})),
        ("[1]", json!({})),
        ("invalid", json!({})),
    ] {
        let value = json!({"choices":[{"message":{"content":content},"finish_reason":"stop"}]});
        let Event::Complete { message, usage, .. } =
            complete_response(&value, Map::new(), Some(ResponseFormat::JsonObject)).unwrap()
        else {
            panic!()
        };
        assert_eq!(json!(message.values), expected);
        assert_eq!(message.content[0].data["value"], content);
        assert!(usage.is_none());
    }
    let usage = parse_usage(Some(&json!({"prompt_tokens":3,"completion_tokens":2})))
        .unwrap()
        .unwrap();
    assert_eq!(usage.total, Some(5));
    assert_eq!(
        parse_usage(Some(&json!({"prompt_tokens":-1})))
            .unwrap_err()
            .code,
        "INVALID_RESPONSE"
    );
}

#[test]
fn named_tool_choice_and_deepseek_reasoning_history_are_translated() {
    let mut history = Message::new(Role::Assistant);
    history.content = vec![
        Part {
            r#type: "think".into(),
            data: part_data(json!({"value":"先查数据"})),
        },
        Part {
            r#type: "tool_call".into(),
            data: part_data(json!({"call_id":"c","name":"weather","arguments":{"city":"北京"}})),
        },
    ];
    let wire = encode_message(Kind::DeepSeek, &history).unwrap();
    assert_eq!(wire["reasoning_content"], "先查数据");
    assert_eq!(
        wire["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"北京\"}"
    );
    let config = model::chat::Config {
        model_name: "test".into(),
        context_limit: None,
        options: Map::new(),
    };
    let request = Request {
        messages: Messages::new([Message::new(Role::User)]),
        tools: vec![model::chat::ToolDefinition {
            name: "weather".into(),
            description: "weather".into(),
            parameters: serde_json::from_value(json!({"type":"object"})).unwrap(),
        }],
        tool_choice: Some(ToolChoice::Named {
            name: "weather".into(),
        }),
        ..Default::default()
    };
    assert_eq!(
        encode_request(Kind::OpenAI, &config, &request).unwrap()["tool_choice"],
        json!({"type":"function","function":{"name":"weather"}})
    );
}
