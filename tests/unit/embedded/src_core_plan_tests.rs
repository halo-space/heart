use super::{decode_plan, stream_request};
use crate::components::agent::{Input, Payload};
use crate::components::{
    message::{Message, Messages, Role},
    model::content::part::{Part, object},
    plan::Plan,
};
use serde_json::{Value, json};

fn valid_plan() -> Value {
    json!({
        "version": 1,
        "nodes": {"research": {"name": "research", "objective": "查找依据"}},
        "edges": []
    })
}

fn assistant_text(value: &str) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: object(json!({"value": value})),
    });
    message
}

#[test]
fn complete_message_decoder_accepts_only_a_complete_valid_plan() {
    let message = assistant_text(&valid_plan().to_string());
    let decoded = decode_plan(&message).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), valid_plan());

    let mut invalid = valid_plan();
    invalid["edges"] = json!([{"from":"research","to":"missing"}]);
    assert_eq!(
        decode_plan(&assistant_text(&invalid.to_string()))
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn complete_message_decoder_rejects_patch_and_runtime_fields() {
    for value in [
        json!({"nodes": {}}),
        json!({"version":1,"nodes":{},"edges":[],"state":{}}),
        json!({
            "version": 1,
            "nodes": {"research": {"name":"research", "objective":"x", "kind":"tool"}},
            "edges": []
        }),
    ] {
        assert!(decode_plan(&assistant_text(&value.to_string())).is_err());
    }
    let mut non_text = Message::new(Role::Assistant);
    non_text.values.insert("patch".into(), json!({}));
    assert!(decode_plan(&non_text).is_err());
}

#[test]
fn planner_request_contains_query_current_graph_and_feedback_as_data() {
    let input = Input {
        metadata: serde_json::from_value(json!({"chat_id": 11, "session_id": 7})).unwrap(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"整理资料"}),
            mime_type: None,
        }],
    };
    let plan: Plan = serde_json::from_value(valid_plan()).unwrap();
    let feedback = Messages::new([assistant_text("补充来源")]);
    let request = stream_request(&input, Some(&plan), Some(&feedback), false).unwrap();
    assert!(!request.stream);
    assert_eq!(
        request.response_format,
        Some(crate::components::model::chat::ResponseFormat::JsonObject)
    );
    assert_eq!(request.messages.0[1].metadata["session_id"], 7);
    let context = request.messages.0[2].content[0].data["value"]
        .as_str()
        .unwrap();
    let context: Value = serde_json::from_str(context).unwrap();
    assert_eq!(context["current_plan"], valid_plan());
    assert_eq!(context["feedback"], serde_json::to_value(feedback).unwrap());
}
