use super::*;
use crate::components::message::Role;
use crate::components::model::content::part::Part;
use serde_json::Map;
use serde_json::json;

fn message(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(json!({"value": text})),
        }],
        values: Map::new(),
        metadata: Map::new(),
        usage: None,
    }
}

#[test]
fn renders_text_and_expands_messages_in_order() {
    let template = Template::new()
        .message(message("问题：{question}"))
        .messages("history");
    let history = Messages::new([message("历史")]);
    let result = template
        .format(
            &Variables::new()
                .value("question", json!("天气"))
                .messages("history", history),
        )
        .unwrap();
    assert_eq!(result.0.len(), 2);
    assert_eq!(result.0[0].content[0].data["value"], "问题：天气");
    assert_eq!(result.0[1].content[0].data["value"], "历史");
}

#[test]
fn rejects_missing_and_non_scalar_variables_without_mutating_template() {
    let template = Template::new().message(message("{question}"));
    assert_eq!(
        template.format(&Variables::new()).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
    let variables = Variables::new().value("question", json!({"text": "天气"}));
    assert_eq!(
        template.format(&variables).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
    let template = Template::new().message(Message {
        role: Role::User,
        content: vec![Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                json!({"value": ["not", "text"]}),
            ),
        }],
        values: Map::new(),
        metadata: Map::new(),
        usage: None,
    });
    assert_eq!(
        template.format(&Variables::new()).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn supports_escaped_braces_and_repeated_variables() {
    let template = Template::new().message(message("{{x}} {name} {name}"));
    let result = template
        .format(&Variables::new().value("name", json!("A")))
        .unwrap();
    assert_eq!(result.0[0].content[0].data["value"], "{x} A A");
}
