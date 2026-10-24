use futures_executor::block_on;
use halo::agent::{Cancellation, Message, Messages};
use halo::event::Event;
use halo::message::Role;
use halo::model::content::part::Part;
use halo::model::token::Usage;
use halo::node::{Echo, Node, Signal};
use serde_json::json;

#[test]
fn message_keeps_values_metadata_and_usage_as_siblings() {
    let mut message = Message::function(json!({"answer": "ok"}));
    message.metadata.insert("source".into(), json!("node_a"));
    message.usage = Some(Usage {
        input: Some(3),
        output: Some(2),
        total: Some(5),
        input_details: None,
        output_details: None,
        details: Default::default(),
    });
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "ok"}).as_object().unwrap().clone(),
    });

    assert_eq!(message.role, Role::Function);
    assert_eq!(message.values["answer"], "ok");
    assert_eq!(message.metadata["source"], "node_a");
    assert_eq!(
        message.usage.as_ref().and_then(|usage| usage.total),
        Some(5)
    );
}

#[test]
fn public_component_error_aliases_and_usage_details_are_stable() {
    let error = halo::agent::Error::new("INVALID_ARGUMENTS", "bad input");
    let _: halo::tool::Error = error.clone();
    let _: halo::function::Error = error.clone();
    let _: halo::prompt::Error = error;

    let usage = Usage {
        input: Some(10),
        output: Some(4),
        total: Some(14),
        input_details: Some(halo::model::token::InputDetails { cached: Some(2) }),
        output_details: Some(halo::model::token::OutputDetails { think: Some(1) }),
        details: Default::default(),
    };
    assert_eq!(
        usage
            .input_details
            .as_ref()
            .and_then(|details| details.cached),
        Some(2)
    );
    assert_eq!(
        usage
            .output_details
            .as_ref()
            .and_then(|details| details.think),
        Some(1)
    );
}

#[test]
fn cancellation_is_shared_and_node_returns_complete() {
    let cancellation = Cancellation::new();
    let child = cancellation.clone();
    child.cancel();
    assert!(cancellation.is_cancelled());

    let input = Messages::new([Message::new(Role::User)]);
    let result = block_on(Echo.call(input.clone(), &cancellation));
    assert_eq!(result, Ok(Signal::Complete(input)));
}

#[test]
fn independent_events_preserve_the_public_wire_format() {
    let event: Event = Event::Complete {
        message: Message::new(Role::Assistant),
        usage: None,
        finish_reason: None,
    };
    let encoded = serde_json::to_value(&event).unwrap();
    assert_eq!(
        encoded,
        json!({"Complete": {
            "message": serde_json::to_value(Message::new(Role::Assistant)).unwrap(),
            "usage": null,
            "finish_reason": null
        }})
    );
    assert_eq!(serde_json::from_value::<Event>(encoded).unwrap(), event);

    let delta = Event::Delta {
        message: Message::new(Role::Assistant),
    };
    let encoded = serde_json::to_value(&delta).unwrap();
    assert_eq!(
        encoded,
        json!({"Delta": {
            "message": serde_json::to_value(Message::new(Role::Assistant)).unwrap()
        }})
    );
    assert_eq!(serde_json::from_value::<Event>(encoded).unwrap(), delta);
}

#[test]
fn content_uses_the_canonical_nested_part_path() {
    let _part = halo::model::content::part::Part {
        r#type: "text".into(),
        data: json!({"value": "ok"}).as_object().unwrap().clone(),
    };
}

#[test]
fn integration_components_are_implementable_without_runtime_state() {
    struct Skill;
    impl halo::skill::Skill for Skill {
        fn definition(&self) -> &halo::skill::Definition {
            static DEFINITION: std::sync::OnceLock<halo::skill::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo::skill::Definition {
                name: "echo".into(),
                description: Some("echoes messages".into()),
                metadata: Default::default(),
            })
        }

        async fn read(&self, _cancellation: &Cancellation) -> Result<String, halo::skill::Error> {
            Ok("skill instructions".into())
        }
    }

    struct Permission;
    impl halo::permission::Permission for Permission {
        async fn check(
            &self,
            request: &halo::permission::Request,
            _cancellation: &Cancellation,
        ) -> Result<halo::permission::Decision, halo::permission::Error> {
            if request.action == "read" {
                Ok(halo::permission::Decision {
                    behavior: halo::permission::Behavior::Allow,
                    reason: None,
                })
            } else {
                Ok(halo::permission::Decision {
                    behavior: halo::permission::Behavior::Deny,
                    reason: Some("denied".into()),
                })
            }
        }
    }

    struct Sandbox;
    impl halo::sandbox::Sandbox for Sandbox {
        async fn execute(
            &self,
            request: halo::sandbox::Request,
            _cancellation: &Cancellation,
        ) -> Result<halo::sandbox::Response, halo::sandbox::Error> {
            Ok(halo::sandbox::Response {
                data: request.input,
                metadata: request.metadata,
            })
        }
    }

    struct Reasoner;
    impl halo::reason::Reasoner for Reasoner {
        async fn reason(
            &self,
            input: Messages,
            _cancellation: &Cancellation,
        ) -> Result<Messages, halo::reason::Error> {
            Ok(input)
        }
    }

    struct Compressor;
    impl halo::compression::Compressor for Compressor {
        async fn compress(
            &self,
            input: Messages,
            _cancellation: &Cancellation,
        ) -> Result<Messages, halo::compression::Error> {
            Ok(input)
        }
    }

    let cancellation = Cancellation::new();
    let input = Messages::new([Message::new(Role::User)]);
    assert_eq!(
        block_on(halo::skill::Skill::read(&Skill, &cancellation)).unwrap(),
        "skill instructions"
    );
    assert!(
        block_on(halo::permission::Permission::check(
            &Permission,
            &halo::permission::Request {
                action: "read".into(),
                resource: "file".into(),
                metadata: Default::default(),
            },
            &cancellation,
        ))
        .is_ok()
    );
    assert_eq!(
        block_on(halo::sandbox::Sandbox::execute(
            &Sandbox,
            halo::sandbox::Request {
                operation: "identity".into(),
                input: json!("ok"),
                metadata: Default::default(),
            },
            &cancellation,
        ))
        .unwrap()
        .data,
        json!("ok")
    );
    assert_eq!(
        block_on(halo::reason::Reasoner::reason(
            &Reasoner,
            input.clone(),
            &cancellation,
        ))
        .unwrap(),
        input.clone()
    );
    assert_eq!(
        block_on(halo::compression::Compressor::compress(
            &Compressor,
            input,
            &cancellation,
        ))
        .unwrap()
        .as_slice()
        .len(),
        1
    );
}

#[test]
fn mcp_and_workspace_are_explicit_adapters() {
    struct Mcp;
    impl halo::mcp::Server for Mcp {
        fn definition(&self) -> &halo::mcp::Definition {
            static DEFINITION: std::sync::OnceLock<halo::mcp::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo::mcp::Definition {
                name: "weather".into(),
                description: Some("weather tools".into()),
                metadata: Default::default(),
            })
        }

        fn tools(&self) -> &[halo::tool::ToolDefinition] {
            static TOOLS: std::sync::OnceLock<Vec<halo::tool::ToolDefinition>> =
                std::sync::OnceLock::new();
            TOOLS.get_or_init(|| {
                vec![halo::tool::ToolDefinition {
                    name: "current_weather".into(),
                    description: "Read weather".into(),
                    parameters: serde_json::Map::from_iter([("type".into(), json!("object"))]),
                }]
            })
        }

        async fn call(
            &self,
            call: halo::tool::Call,
            _cancellation: &Cancellation,
        ) -> Result<Message, halo::mcp::Error> {
            Ok(Message::function(json!({"name": call.name})))
        }
    }

    struct Workspace;
    impl halo::workspace::Workspace for Workspace {
        fn definition(&self) -> &halo::workspace::Definition {
            static DEFINITION: std::sync::OnceLock<halo::workspace::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo::workspace::Definition {
                mode: halo::workspace::Mode::Server,
                path: None,
                metadata: Default::default(),
            })
        }

        fn tool(&self) -> Vec<halo::tool::ToolDefinition> {
            Vec::new()
        }

        fn mcp(&self) -> Vec<halo::mcp::Definition> {
            Vec::new()
        }

        fn skill(&self) -> Vec<halo::skill::Definition> {
            Vec::new()
        }

        fn execute<'a>(
            &'a self,
            _call: halo::tool::Call,
            _cancellation: &'a Cancellation,
        ) -> halo::workspace::ToolFuture<'a> {
            Box::pin(async {
                Err(halo::Error::new(
                    "NOT_IMPLEMENTED",
                    "contract fixture has no workspace tools",
                ))
            })
        }
    }

    let cancellation = Cancellation::new();
    assert_eq!(halo::mcp::Server::definition(&Mcp).name, "weather");
    assert_eq!(halo::mcp::Server::tools(&Mcp).len(), 1);
    assert_eq!(
        block_on(halo::mcp::Server::call(
            &Mcp,
            halo::tool::Call {
                call_id: "call_1".into(),
                name: "current_weather".into(),
                arguments: Default::default(),
            },
            &cancellation,
        ))
        .unwrap()
        .values["name"],
        "current_weather"
    );
    assert_eq!(
        halo::workspace::Workspace::definition(&Workspace).mode,
        halo::workspace::Mode::Server
    );
    assert!(
        halo::workspace::Workspace::definition(&Workspace)
            .path
            .is_none()
    );
}
