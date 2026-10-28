//! End-to-end Skill discovery and on-demand reading with a real DeepSeek model.

use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use halo::{
    agent::core::workspace::Local,
    agent::{
        Agent, Cancellation, Input, Payload, event::Event, model::chat,
        provider::deepseek::DeepSeek,
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
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct live Chat model");
    let agent = harness::builder()
        .model(model)
        .system_prompt(
            "Follow the user request exactly. You must discover the named Workspace Skill, read its instructions, then read any referenced resource before answering. Never guess the requested verification code.",
        )
        .workspace(workspace)
        .with_max_iters(5)
        .build()
        .expect("build Harness Agent with local Workspace");
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"Read the field-guide Skill and its referenced verification.md resource, then tell me its verification code. Reply with only the code."}),
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
