//! Local Workspace tool selection and side effects with a live model.

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

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "heart-live-workspace-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create isolated Local Workspace");
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and accesses the live DeepSeek API"]
async fn real_model_writes_user_content_through_local_workspace_tool() {
    let directory = Directory::new();
    let workspace = Local::new(&directory.0).expect("open isolated Local Workspace");
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct live DeepSeek provider");
    let model = provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct live Chat");
    let agent = harness::builder()
        .model(model)
        .workspace(workspace)
        .system_prompt(
            "When asked to save text, you must call workspace_write with the exact requested path and content. Do not claim success unless the tool result confirms it.",
        )
        .with_max_iters(3)
        .build()
        .expect("build harness Agent");
    let input = Input {
        metadata: json!({"chat_id":9_100_001,"agent_id":9_100_002,"session_id":9_100_003})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请使用工作区工具将精确文本 `REAL-WORKSPACE-ORBIT-7319` 保存到文件 `evidence.txt`，然后告诉我保存结果。"}),
            mime_type: None,
        }],
    };
    let events: Vec<_> = agent
        .run(input, false, Cancellation::new())
        .await
        .expect("run live Agent")
        .collect()
        .await;

    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Ok(Event::Complete { .. })))
    );
    assert_eq!(
        std::fs::read_to_string(directory.0.join("evidence.txt"))
            .expect("the live model must create the requested file"),
        "REAL-WORKSPACE-ORBIT-7319"
    );
}
