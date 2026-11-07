//! AgentScope console example rewritten with Heart's Harness.
//!
//! This is a real interactive example: DeepSeek drives a Harness Agent,
//! Workspace supplies local file tools, and a selective Permission policy asks
//! before write/delete/execute operations. Set `DEEPSEEK_API_KEY` before run.
//!
//! ```text
//! cargo run --example console_harness
//! ```

use std::sync::atomic::{AtomicI64, Ordering};

use futures_util::StreamExt;
use halo::agent::core::workspace::Local;
use halo::agent::event::Event;
use halo::agent::memory::short::session::Store as _;
use halo::agent::memory::short::{self, session};
use halo::agent::model::chat;
use halo::agent::provider::deepseek::DeepSeek;
use halo::agent::{Agent, Cancellation, Error, Input, Payload};
use halo::{agent, harness};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader, Lines, Stdin};

static NEXT_ID: AtomicI64 = AtomicI64::new(9_500_000);

fn next_id() -> i64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn text(message: &halo::agent::Message) -> String {
    message
        .content
        .iter()
        .filter(|part| part.r#type == "text" || part.r#type == "think")
        .filter_map(|part| part.data.get("value").and_then(Value::as_str))
        .collect()
}

fn input(chat_id: i64, value: String) -> Input {
    Input {
        metadata: json!({
            "tenant_id": 9_500_000,
            "user_id": 9_500_003,
            "agent_id": 9_500_002,
            "chat_id": chat_id,
            "session_id": 9_500_001,
        })
        .as_object()
        .expect("static metadata object")
        .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value": value}),
            mime_type: None,
        }],
    }
}

fn reply(chat_id: i64, reply_id: i64, approved: bool) -> Input {
    Input {
        metadata: json!({
            "tenant_id": 9_500_000,
            "user_id": 9_500_003,
            "agent_id": 9_500_002,
            "chat_id": chat_id,
            "session_id": 9_500_001,
            "reply_id": reply_id,
        })
        .as_object()
        .expect("static reply metadata object")
        .clone(),
        payload: vec![Payload {
            r#type: "json".into(),
            data: json!({"value": {"approved": approved}}),
            mime_type: None,
        }],
    }
}

type InputLines = Lines<BufReader<Stdin>>;

async fn run_turn<M>(
    agent: &harness::Agent<M>,
    chat_id: i64,
    value: String,
    lines: &mut InputLines,
) -> Result<(), Error>
where
    M: chat::Chat + 'static,
{
    let mut events = agent
        .run(input(chat_id, value), true, Cancellation::new())
        .await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => {
                for part in message.content {
                    if let Some(value) = part.data.get("value").and_then(Value::as_str) {
                        match part.r#type.as_str() {
                            "think" => eprint!("\x1b[90m{value}\x1b[0m"),
                            "text" => print!("{value}"),
                            "tool_call" => println!("\n[tool] {value}"),
                            _ => {}
                        }
                    }
                }
            }
            Event::Complete {
                message,
                usage,
                finish_reason,
            } => {
                if let Some(reply_id) = message.metadata.get("reply_id").and_then(Value::as_i64) {
                    println!("\n\n[permission] {}", text(&message));
                    print!("Allow this operation? [y/N] ");
                    std::io::Write::flush(&mut std::io::stdout())
                        .map_err(|error| Error::new("IO", error.to_string()))?;
                    let approved = lines
                        .next_line()
                        .await
                        .map_err(|error| Error::new("IO", error.to_string()))?
                        .is_some_and(|line| {
                            matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                        });
                    agent
                        .reply(reply(chat_id, reply_id, approved), Cancellation::new())
                        .await?;
                } else if message.role == halo::agent::message::Role::Tool {
                    println!("\n[tool-result] {}", text(&message));
                } else {
                    println!("\n[complete] finish={finish_reason:?} usage={usage:?}");
                }
            }
        }
    }
    println!();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .map_err(|_| Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required"))?;
    let root = std::env::var("HEART_WORKDIR").unwrap_or_else(|_| "./heart-workspace".into());
    std::fs::create_dir_all(&root).map_err(|error| Error::new("IO", error.to_string()))?;

    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(api_key)
        .build()?;
    let model = provider.chat(
        chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        )
        .options(serde_json::Map::from_iter([(
            "thinking".into(),
            json!({"type": "disabled"}),
        )])),
    )?;
    let workspace = Local::new(&root).map_err(|error| Error::new("IO", error.to_string()))?;
    let memory = agent::memory::Config::new(
        agent::core::memory::short::InMemory::new(),
        agent::core::memory::InMemory::new(),
    )
    .with_scopes(["user", "agent"]);
    memory
        .session()
        .create(
            session::Session {
                id: 9_500_001,
                tenant_id: 9_500_000,
                user_id: 9_500_003,
                r#type: short::Type::Agent,
                title: Some("Heart Harness console".into()),
                summary: None,
                usage: None,
                status: session::Status::Active,
                metadata: Default::default(),
                created_time: 0,
                updated_time: 0,
            },
            &Cancellation::new(),
        )
        .await?;
    let permission = agent::core::permission::Permission::new(
        |request: agent::permission::Request, _: Cancellation| async move {
            let read_only = matches!(
                request.resource.as_str(),
                "workspace_read" | "workspace_list" | "workspace_glob" | "workspace_grep"
            );
            Ok(agent::permission::Decision {
                behavior: if read_only {
                    agent::permission::Behavior::Allow
                } else {
                    agent::permission::Behavior::Ask
                },
                reason: Some(if read_only {
                    "read-only Workspace operation".into()
                } else {
                    "this Workspace operation changes state or executes code".into()
                }),
            })
        },
    );
    let agent = harness::builder()
        .model(model)
        .workspace(workspace)
        .permission(permission)
        .memory(memory)
        .with_id_fn(next_id)
        .with_max_iters(20)
        .with_max_concurrency(4)
        .system_prompt(
            "你是一个运行在本地 Workspace 中的长任务助手。使用真实 Workspace 工具读取、搜索和修改文件；\
             不要编造工具结果。需要写入、删除或执行操作时，等待 Permission 审批后再继续。\
             记忆用户明确要求记住的偏好，并在后续轮次使用已有上下文。",
        )
        .build()?;

    println!("Heart Harness console. 输入 exit 或 quit 退出。Workspace: {root}");
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut chat_id = 9_500_100;
    loop {
        print!("\n> ");
        std::io::Write::flush(&mut std::io::stdout())
            .map_err(|error| Error::new("IO", error.to_string()))?;
        let Some(line) = lines
            .next_line()
            .await
            .map_err(|error| Error::new("IO", error.to_string()))?
        else {
            break;
        };
        if matches!(line.trim().to_ascii_lowercase().as_str(), "exit" | "quit") {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        run_turn(&agent, chat_id, line, &mut lines).await?;
        chat_id += 1;
    }
    Ok(())
}
