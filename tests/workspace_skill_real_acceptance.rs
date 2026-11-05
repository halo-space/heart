//! End-to-end Skill discovery and on-demand reading with a real DeepSeek model.

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use halo::{
    agent::core::workspace::Local,
    agent::{
        Agent, Cancellation, Error, Input, Payload, event::Event, middleware::Middleware,
        model::chat, provider::deepseek::DeepSeek,
    },
    harness,
};
use serde_json::json;

struct TemporaryWorkspace(PathBuf);

impl TemporaryWorkspace {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("heart-live-skill-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(root.join("skills/field-guide"))
            .expect("create real local Workspace Skill directory");
        std::fs::create_dir_all(root.join("skills/field-guide/references"))
            .expect("create Skill resource directory");
        std::fs::write(
            root.join("skills/field-guide/SKILL.md"),
            "---\nname: field-guide\ndescription: Read this guide to answer the verification question.\n---\nRead references/verification.md and use its exact verification code. Do not guess.\n",
        )
        .expect("write real Skill instructions to the Workspace filesystem");
        std::fs::write(
            root.join("skills/field-guide/references/verification.md"),
            "The verification code is ORBIT-7319.\n",
        )
        .expect("write real nested Skill resource to the Workspace filesystem");
        Self(root)
    }
}

impl Drop for TemporaryWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct RequireSkillCalls {
    count: Arc<AtomicUsize>,
}

impl Middleware for RequireSkillCalls {
    async fn before_model(
        &self,
        input: &mut serde_json::Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let has_skill_tool = input["tools"]
            .as_array()
            .is_some_and(|tools| tools.iter().any(|tool| tool["name"] == "skill"));
        if has_skill_tool && self.count.fetch_add(1, Ordering::SeqCst) < 2 {
            input["tool_choice"] = json!({"Named": {"name": "skill"}});
        }
        Ok(())
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and accesses the live DeepSeek API"]
async fn real_model_discovers_and_reads_a_local_workspace_skill() {
    let directory = TemporaryWorkspace::new();
    let workspace = Local::new(&directory.0).expect("open local Workspace");
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct the live DeepSeek provider");
    let model = provider
        .chat(
            chat::Config::new(
                std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
            )
            .options(serde_json::Map::from_iter([(
                "thinking".into(),
                json!({"type": "disabled"}),
            )])),
        )
        .expect("construct live Chat model");
    let agent = harness::builder()
        .model(model)
        .system_prompt(
            "Follow the user request exactly. The only valid way to answer is to call the `skill` tool. First call it with {\"skill\":\"field-guide\"} to read the Skill instructions. Then call the same `skill` tool with {\"skill\":\"field-guide\",\"resource\":\"references/verification.md\"}. Do not answer before both tool calls and never guess the verification code.",
        )
        .workspace(workspace)
        .middleware(RequireSkillCalls {
            count: Arc::new(AtomicUsize::new(0)),
        })
        .with_max_iters(5)
        .build()
        .expect("build Harness Agent with local Workspace");
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"Use the skill tool now. Read field-guide, then read references/verification.md through the skill tool, and reply with only the exact verification code from that resource."}),
            mime_type: None,
        }],
        ..Default::default()
    };

    let mut events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("run against the live model and local Workspace");
    let mut final_text = String::new();
    let mut model_completions = 0;
    let mut skill_reader_called = false;
    let mut nested_resource_read = false;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.expect("Agent event") {
            model_completions += 1;
            skill_reader_called |= message.content.iter().any(|part| {
                part.r#type == "tool_call"
                    && part.data.get("name").and_then(|name| name.as_str()) == Some("skill")
            });
            nested_resource_read |= message.content.iter().any(|part| {
                part.r#type == "tool_call"
                    && part.data.get("name").and_then(|name| name.as_str()) == Some("skill")
                    && part.data["arguments"]["resource"] == "references/verification.md"
            });
            if message.role == halo::agent::message::Role::Assistant {
                final_text = message
                    .content
                    .iter()
                    .filter(|part| part.r#type == "text")
                    .filter_map(|part| part.data.get("value").and_then(|value| value.as_str()))
                    .collect::<String>();
            }
        }
    }

    assert!(
        model_completions >= 2,
        "expected a Skill tool interaction and a final model answer; got {model_completions} completions"
    );
    assert!(
        skill_reader_called,
        "the live model must actually call the Workspace Skill reader"
    );
    assert!(
        nested_resource_read,
        "the live model must actually read the referenced nested Skill resource"
    );
    assert_eq!(
        final_text.trim(),
        "ORBIT-7319",
        "the answer must come from the actual local Skill instructions"
    );
    println!("LIVE_WORKSPACE_SKILL_RESULT={final_text}");
}
