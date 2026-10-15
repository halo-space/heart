use futures_executor::block_on;
use halo_agents::event::Event;
use halo_agents::message::Role;
use halo_agents::model::content::part::Part;
use halo_agents::model::token::Usage;
use halo_agents::node::{Echo, Node, Signal};
use halo_agents::{Cancellation, Message, Messages};
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
    let error = halo_agents::Error::new("INVALID_ARGUMENTS", "bad input");
    let _: halo_agents::tool::Error = error.clone();
    let _: halo_agents::function::Error = error.clone();
    let _: halo_agents::prompt::Error = error;

    let usage = Usage {
        input: Some(10),
        output: Some(4),
        total: Some(14),
        input_details: Some(halo_agents::model::token::InputDetails { cached: Some(2) }),
        output_details: Some(halo_agents::model::token::OutputDetails { think: Some(1) }),
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
    let result = block_on(Echo.call(input.clone(), cancellation));
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
    let _part = halo_agents::model::content::part::Part {
        r#type: "text".into(),
        data: json!({"value": "ok"}).as_object().unwrap().clone(),
    };
}

#[test]
fn integration_components_are_implementable_without_runtime_state() {
    struct Skill;
    impl halo_agents::skill::Skill for Skill {
        fn definition(&self) -> &halo_agents::skill::Definition {
            static DEFINITION: std::sync::OnceLock<halo_agents::skill::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo_agents::skill::Definition {
                name: "echo".into(),
                description: Some("echoes messages".into()),
                metadata: Default::default(),
            })
        }

        async fn run(
            &self,
            input: Messages,
            _cancellation: &Cancellation,
        ) -> Result<Messages, halo_agents::skill::Error> {
            Ok(input)
        }
    }

    struct Permission;
    impl halo_agents::permission::Permission for Permission {
        async fn check(
            &self,
            request: &halo_agents::permission::Request,
            _cancellation: &Cancellation,
        ) -> Result<halo_agents::permission::Decision, halo_agents::permission::Error> {
            if request.action == "read" {
                Ok(halo_agents::permission::Decision {
                    behavior: halo_agents::permission::Behavior::Allow,
                    reason: None,
                })
            } else {
                Ok(halo_agents::permission::Decision {
                    behavior: halo_agents::permission::Behavior::Deny,
                    reason: Some("denied".into()),
                })
            }
        }
    }

    struct Sandbox;
    impl halo_agents::sandbox::Sandbox for Sandbox {
        async fn execute(
            &self,
            request: halo_agents::sandbox::Request,
            _cancellation: &Cancellation,
        ) -> Result<halo_agents::sandbox::Response, halo_agents::sandbox::Error> {
            Ok(halo_agents::sandbox::Response {
                data: request.input,
                metadata: request.metadata,
            })
        }
    }

    struct Reasoner;
    impl halo_agents::reason::Reasoner for Reasoner {
        async fn reason(
            &self,
            input: Messages,
            _cancellation: &Cancellation,
        ) -> Result<Messages, halo_agents::reason::Error> {
            Ok(input)
        }
    }

    struct Compressor;
    impl halo_agents::compression::Compressor for Compressor {
        async fn compress(
            &self,
            input: Messages,
            _cancellation: &Cancellation,
        ) -> Result<Messages, halo_agents::compression::Error> {
            Ok(input)
        }
    }

    let cancellation = Cancellation::new();
    let input = Messages::new([Message::new(Role::User)]);
    assert_eq!(
        block_on(halo_agents::skill::Skill::run(
            &Skill,
            input.clone(),
            &cancellation
        ))
        .unwrap(),
        input
    );
    assert!(
        block_on(halo_agents::permission::Permission::check(
            &Permission,
            &halo_agents::permission::Request {
                action: "read".into(),
                resource: "file".into(),
                metadata: Default::default(),
            },
            &cancellation,
        ))
        .is_ok()
    );
    assert_eq!(
        block_on(halo_agents::sandbox::Sandbox::execute(
            &Sandbox,
            halo_agents::sandbox::Request {
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
        block_on(halo_agents::reason::Reasoner::reason(
            &Reasoner,
            input.clone(),
            &cancellation,
        ))
        .unwrap(),
        input.clone()
    );
    assert_eq!(
        block_on(halo_agents::compression::Compressor::compress(
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
    impl halo_agents::mcp::Server for Mcp {
        fn definition(&self) -> &halo_agents::mcp::Definition {
            static DEFINITION: std::sync::OnceLock<halo_agents::mcp::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo_agents::mcp::Definition {
                name: "weather".into(),
                description: Some("weather tools".into()),
                metadata: Default::default(),
            })
        }

        async fn list_tools(
            &self,
            _cancellation: &Cancellation,
        ) -> Result<Vec<halo_agents::tool::ToolDefinition>, halo_agents::mcp::Error> {
            Ok(vec![halo_agents::tool::ToolDefinition {
                name: "current_weather".into(),
                description: "Read weather".into(),
                parameters: serde_json::Map::from_iter([("type".into(), json!("object"))]),
            }])
        }

        async fn call(
            &self,
            call: halo_agents::tool::Call,
            _cancellation: &Cancellation,
        ) -> Result<Message, halo_agents::mcp::Error> {
            Ok(Message::function(json!({"name": call.name})))
        }
    }

    struct Workspace;
    impl halo_agents::workspace::Workspace for Workspace {
        fn definition(&self) -> &halo_agents::workspace::Definition {
            static DEFINITION: std::sync::OnceLock<halo_agents::workspace::Definition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| halo_agents::workspace::Definition {
                mode: halo_agents::workspace::Mode::Server,
                path: None,
                metadata: Default::default(),
            })
        }

        async fn read(
            &self,
            _path: &str,
            _cancellation: &Cancellation,
        ) -> Result<Vec<u8>, halo_agents::workspace::Error> {
            Ok(Vec::new())
        }

        async fn write(
            &self,
            _path: &str,
            _data: Vec<u8>,
            _cancellation: &Cancellation,
        ) -> Result<(), halo_agents::workspace::Error> {
            Ok(())
        }

        async fn delete(
            &self,
            _path: &str,
            _cancellation: &Cancellation,
        ) -> Result<(), halo_agents::workspace::Error> {
            Ok(())
        }
    }

    let cancellation = Cancellation::new();
    assert_eq!(halo_agents::mcp::Server::definition(&Mcp).name, "weather");
    assert_eq!(
        block_on(halo_agents::mcp::Server::list_tools(&Mcp, &cancellation))
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        block_on(halo_agents::mcp::Server::call(
            &Mcp,
            halo_agents::tool::Call {
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
        halo_agents::workspace::Workspace::definition(&Workspace).mode,
        halo_agents::workspace::Mode::Server
    );
    assert!(
        block_on(halo_agents::workspace::Workspace::write(
            &Workspace,
            "remote/file",
            vec![1],
            &cancellation,
        ))
        .is_ok()
    );
}
