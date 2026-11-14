use super::run::result_message;
use crate::{Message, message::Role, tool::Call};
use serde_json::json;

#[test]
fn child_private_think_is_not_forwarded_as_a_function_result() {
    let call: Call = serde_json::from_value(json!({
        "call_id": "call_child",
        "name": "agent_spawn",
        "arguments": {}
    }))
    .unwrap();
    let mut message = Message::new(Role::Assistant);
    message.content.extend([
        crate::model::content::part::Part {
            r#type: "think".into(),
            data: json!({"value":"private reasoning"})
                .as_object()
                .unwrap()
                .clone(),
        },
        crate::model::content::part::Part {
            r#type: "text".into(),
            data: json!({"value":"child result"}).as_object().unwrap().clone(),
        },
    ]);

    let result = result_message(message, &call);

    assert_eq!(result.role, Role::Function);
    assert_eq!(result.content.len(), 1);
    assert_eq!(result.content[0].r#type, "text");
    assert_eq!(result.content[0].data["value"], "child result");
    assert_eq!(result.metadata["call_id"], "call_child");
}
