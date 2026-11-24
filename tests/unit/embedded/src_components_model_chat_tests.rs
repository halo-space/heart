use super::*;
use serde_json::json;

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: "test tool".into(),
        parameters: serde_json::from_value(json!({"type": "object", "properties": {}})).unwrap(),
    }
}

fn messages() -> Messages {
    let mut system = Message::new(Role::System);
    system.content.push(Part {
        r#type: "text".into(),
        data: serde_json::from_value(json!({"value":"You are a test assistant."})).unwrap(),
        mime_type: None,
    });
    let mut user = Message::new(Role::User);
    user.content.push(Part {
        r#type: "text".into(),
        data: serde_json::from_value(json!({"value":"Run the test."})).unwrap(),
        mime_type: None,
    });
    Messages::new([system, user])
}

#[test]
fn request_validates_tool_choice_and_response_format() {
    let mut request = Request {
        messages: messages(),
        tools: vec![tool("weather")],
        tool_choice: Some(ToolChoice::Named {
            name: "weather".into(),
        }),
        response_format: Some(ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: json!({"type": "object", "properties": {}}),
            strict: true,
        }),
        ..Request::default()
    };
    assert!(request.validate().is_ok());
    request.tool_choice = Some(ToolChoice::Named {
        name: "missing".into(),
    });
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
}

#[test]
fn request_requires_system_and_user_messages_for_model_dispatch() {
    let mut request = Request {
        messages: Messages::new([Message::new(Role::User)]),
        ..Request::default()
    };
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");

    request.messages = Messages::new([Message::new(Role::System)]);
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");

    request.messages = messages();
    assert!(request.validate().is_ok());

    let mut late_system = messages();
    late_system.push(Message::new(Role::System));
    request.messages = late_system;
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
}

#[test]
fn config_keeps_model_defaults_separate_from_request_options() {
    let config = Config {
        model_name: "demo".into(),
        context_limit: None,
        options: Map::from_iter([("thinking".into(), json!(true))]),
    };
    let request = Request {
        options: Map::from_iter([("temperature_mode".into(), json!("fast"))]),
        ..Request::default()
    };
    assert_eq!(config.model_name, "demo");
    assert!(config.options.contains_key("thinking"));
    assert!(request.options.contains_key("temperature_mode"));
}

#[test]
fn config_constructor_keeps_provider_and_model_binding_separate() {
    let config = Config::new("demo").options(Map::from_iter([("thinking".into(), json!(true))]));
    assert_eq!(config.model_name, "demo");
    assert_eq!(config.options["thinking"], json!(true));
}
