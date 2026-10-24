use futures_executor::block_on;
use halo::agent::event::Event;
use halo::agent::message::Role;
use halo::agent::model::content::part::Part;
use halo::agent::model::token::Usage;
use halo::agent::node::{Echo, Node, Signal};
use halo::agent::{Cancellation, Message, Messages};
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
    let _: halo::agent::tool::Error = error.clone();
    let _: halo::agent::function::Error = error.clone();
    let _: halo::agent::prompt::Error = error;

    let usage = Usage {
        input: Some(10),
        output: Some(4),
        total: Some(14),
        input_details: Some(halo::agent::model::token::InputDetails { cached: Some(2) }),
        output_details: Some(halo::agent::model::token::OutputDetails { think: Some(1) }),
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
    let _part = halo::agent::model::content::part::Part {
        r#type: "text".into(),
        data: json!({"value": "ok"}).as_object().unwrap().clone(),
    };
}

#[test]
fn integration_components_are_implementable_without_runtime_state() {
    struct Skill;
    impl halo::agent::skill::Skill for Skill {
        fn definition(&self) -> &halo::agent::skill::Definition {
            static DEFINITION: std::sync::OnceLock<halo::agent::skill::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo::agent::skill::Definition {
                name: "echo".into(),
                description: Some("echoes messages".into()),
                metadata: Default::default(),
            })
        }

        async fn read(
            &self,
            _cancellation: &Cancellation,
        ) -> Result<String, halo::agent::skill::Error> {
            Ok("skill instructions".into())
        }
    }

    struct Permission;
    impl halo::agent::permission::Permission for Permission {
        async fn check(
            &self,
            request: &halo::agent::permission::Request,
            _cancellation: &Cancellation,
        ) -> Result<halo::agent::permission::Decision, halo::agent::permission::Error> {
            if request.action == "read" {
                Ok(halo::agent::permission::Decision {
                    behavior: halo::agent::permission::Behavior::Allow,
                    reason: None,
                })
            } else {
                Ok(halo::agent::permission::Decision {
                    behavior: halo::agent::permission::Behavior::Deny,
                    reason: Some("denied".into()),
                })
            }
        }
    }

    struct Sandbox;
    impl halo::agent::sandbox::Sandbox for Sandbox {
        async fn execute(
            &self,
            request: halo::agent::sandbox::Request,
            _cancellation: &Cancellation,
        ) -> Result<halo::agent::sandbox::Response, halo::agent::sandbox::Error> {
            Ok(halo::agent::sandbox::Response {
                data: request.input,
                metadata: request.metadata,
            })
        }
    }

    struct Reasoner;
    impl halo::agent::reason::Reasoner for Reasoner {
        async fn reason(
            &self,
            input: Messages,
            _cancellation: &Cancellation,
        ) -> Result<Messages, halo::agent::reason::Error> {
            Ok(input)
        }
    }

    struct Compressor;
    impl halo::agent::compression::Compressor for Compressor {
        async fn compress(
            &self,
            input: Messages,
            _cancellation: &Cancellation,
        ) -> Result<Messages, halo::agent::compression::Error> {
            Ok(input)
        }
    }

    let cancellation = Cancellation::new();
    let input = Messages::new([Message::new(Role::User)]);
    assert_eq!(
        block_on(halo::agent::skill::Skill::read(&Skill, &cancellation)).unwrap(),
        "skill instructions"
    );
    assert!(
        block_on(halo::agent::permission::Permission::check(
            &Permission,
            &halo::agent::permission::Request {
                action: "read".into(),
                resource: "file".into(),
                metadata: Default::default(),
            },
            &cancellation,
        ))
        .is_ok()
    );
    assert_eq!(
        block_on(halo::agent::sandbox::Sandbox::execute(
            &Sandbox,
            halo::agent::sandbox::Request {
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
        block_on(halo::agent::reason::Reasoner::reason(
            &Reasoner,
            input.clone(),
            &cancellation,
        ))
        .unwrap(),
        input.clone()
    );
    assert_eq!(
        block_on(halo::agent::compression::Compressor::compress(
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
    impl halo::agent::mcp::Server for Mcp {
        fn definition(&self) -> &halo::agent::mcp::Definition {
            static DEFINITION: std::sync::OnceLock<halo::agent::mcp::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo::agent::mcp::Definition {
                name: "weather".into(),
                description: Some("weather tools".into()),
                metadata: Default::default(),
            })
        }

        fn tools(&self) -> &[halo::agent::tool::ToolDefinition] {
            static TOOLS: std::sync::OnceLock<Vec<halo::agent::tool::ToolDefinition>> =
                std::sync::OnceLock::new();
            TOOLS.get_or_init(|| {
                vec![halo::agent::tool::ToolDefinition {
                    name: "current_weather".into(),
                    description: "Read weather".into(),
                    parameters: serde_json::Map::from_iter([("type".into(), json!("object"))]),
                }]
            })
        }

        async fn call(
            &self,
            call: halo::agent::tool::Call,
            _cancellation: &Cancellation,
        ) -> Result<Message, halo::agent::mcp::Error> {
            Ok(Message::function(json!({"name": call.name})))
        }
    }

    struct Workspace;
    impl halo::agent::workspace::Workspace for Workspace {
        fn definition(&self) -> &halo::agent::workspace::Definition {
            static DEFINITION: std::sync::OnceLock<halo::agent::workspace::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo::agent::workspace::Definition {
                mode: halo::agent::workspace::Mode::Server,
                path: None,
                metadata: Default::default(),
            })
        }

        fn tool(&self) -> Vec<halo::agent::tool::ToolDefinition> {
            Vec::new()
        }

        fn mcp(&self) -> Vec<halo::agent::mcp::Definition> {
            Vec::new()
        }

        fn skill(&self) -> Vec<halo::agent::skill::Definition> {
            Vec::new()
        }

        fn execute<'a>(
            &'a self,
            _call: halo::agent::tool::Call,
            _cancellation: &'a Cancellation,
        ) -> halo::agent::workspace::ToolFuture<'a> {
            Box::pin(async {
                Err(halo::agent::Error::new(
                    "NOT_IMPLEMENTED",
                    "contract fixture has no workspace tools",
                ))
            })
        }
    }

    let cancellation = Cancellation::new();
    assert_eq!(halo::agent::mcp::Server::definition(&Mcp).name, "weather");
    assert_eq!(halo::agent::mcp::Server::tools(&Mcp).len(), 1);
    assert_eq!(
        block_on(halo::agent::mcp::Server::call(
            &Mcp,
            halo::agent::tool::Call {
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
        halo::agent::workspace::Workspace::definition(&Workspace).mode,
        halo::agent::workspace::Mode::Server
    );
    assert!(
        halo::agent::workspace::Workspace::definition(&Workspace)
            .path
            .is_none()
    );
}
