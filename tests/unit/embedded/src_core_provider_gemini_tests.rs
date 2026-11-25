use super::*;

#[test]
fn config_redacts_key_and_uses_gemini_endpoint() {
    let config = Config {
        api_key: "secret".into(),
        ..Config::default()
    };
    assert_eq!(config.provider_name, "gemini");
    assert!(format!("{config:?}").contains("[redacted]"));
    assert!(!format!("{config:?}").contains("secret"));
}

#[test]
fn media_input_preserves_nested_source_contract() {
    let part = Part {
        r#type: "image".into(),
        data: object(json!({
            "source": {
                "kind": "base64",
                "value": "data:image/png;base64,AAAA",
                "mime_type": "image/png"
            }
        })),
        mime_type: None,
    };
    let value = part_to_input(&part).unwrap();
    assert_eq!(value["type"], "image");
    assert_eq!(value["mime_type"], "image/png");
    assert_eq!(value["data"], "AAAA");
}

#[test]
fn tts_body_uses_gemini_interactions_audio_shape() {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: object(json!({"value": "你好"})),
        mime_type: None,
    });
    let request = model::tts::Request {
        messages: Messages::new([message]),
        voice: Some("Kore".into()),
        format: Some("audio/wav".into()),
        stream: false,
        ..model::tts::Request::default()
    };
    let body = tts_body(&model::tts::Config::new("gemini-3.8-flash-tts"), &request).unwrap();
    assert_eq!(body["model"], "gemini-3.8-flash-tts");
    assert_eq!(body["response_format"]["mime_type"], "audio/wav");
    assert_eq!(
        body["generation_config"]["speech_config"][0]["voice"],
        "Kore"
    );
    assert_eq!(body["input"][0]["content"][0]["text"], "你好");
}

#[test]
fn multimodal_body_preserves_image_url_input() {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: object(json!({"value": "描述图片"})),
        mime_type: None,
    });
    message.content.push(Part {
        r#type: "image".into(),
        data: object(json!({
            "source": {
                "kind": "url",
                "value": "https://example.com/image.png",
                "mime_type": "image/png"
            }
        })),
        mime_type: None,
    });
    let request = model::multimodal::Request {
        messages: Messages::new([message]),
        ..model::multimodal::Request::default()
    };
    let body = multimodal_body(
        &model::multimodal::Config::new("gemini-3.8-flash"),
        &request,
    )
    .unwrap();
    assert_eq!(body["model"], "gemini-3.8-flash");
    assert_eq!(body["input"][1]["uri"], "https://example.com/image.png");
    assert_eq!(body["input"][1]["mime_type"], "image/png");
}

#[test]
fn multimodal_body_rejects_media_outside_the_v3_input_scope() {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "audio".into(),
        data: object(json!({
            "source": {
                "kind": "base64",
                "value": "data:audio/wav;base64,AAAA",
                "mime_type": "audio/wav"
            }
        })),
        mime_type: Some("audio/wav".into()),
    });
    let request = model::multimodal::Request {
        messages: Messages::new([message]),
        ..model::multimodal::Request::default()
    };
    let error = multimodal_body(
        &model::multimodal::Config::new("gemini-3.5-flash"),
        &request,
    )
    .unwrap_err();
    assert_eq!(error.code, "UNSUPPORTED");
}

#[test]
fn multimodal_body_rejects_unsupported_output_modalities() {
    let request = model::multimodal::Request {
        messages: Messages::new([{
            let mut message = Message::new(Role::User);
            message.content.push(Part {
                r#type: "text".into(),
                data: object(json!({"value": "描述图片"})),
                mime_type: None,
            });
            message
        }]),
        modalities: vec!["image".into()],
        ..model::multimodal::Request::default()
    };
    let error = multimodal_body(
        &model::multimodal::Config::new("gemini-3.5-flash"),
        &request,
    )
    .unwrap_err();
    assert_eq!(error.code, "UNSUPPORTED");
}
