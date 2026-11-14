use super::{Config, Request, ResponseFormat, ToolChoice};
use crate::components::tool::ToolDefinition;
use serde_json::json;

#[test]
fn request_validates_modalities_and_schema() {
    let mut request = Request {
        modalities: vec!["text".into(), "image".into()],
        response_format: Some(ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: json!({"type":"object","properties":{}}),
            strict: true,
        }),
        ..Request::default()
    };
    request.validate().unwrap();
    request.modalities.push("  ".into());
    assert!(request.validate().is_err());
}

#[test]
fn config_keeps_model_defaults_separate() {
    let config = Config::new("vision-model");
    assert_eq!(config.model_name, "vision-model");
    assert!(config.options.is_empty());
}

#[test]
fn tool_choice_is_owned_by_multimodal_contract() {
    let request = Request {
        tools: vec![ToolDefinition {
            name: "describe".into(),
            description: "Describe the image".into(),
            parameters: json!({"type":"object"}).as_object().unwrap().clone(),
        }],
        tool_choice: Some(ToolChoice::Named {
            name: "describe".into(),
        }),
        ..Request::default()
    };
    request.validate().unwrap();
}
