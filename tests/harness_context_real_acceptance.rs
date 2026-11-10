//! Real-model Harness Context acceptance.
//!
//! These tests use a live DeepSeek request. They are ignored unless the
//! application supplies `DEEPSEEK_API_KEY`.

use futures_util::StreamExt;
use halo::agent::{Agent, Cancellation, Error, Input, Message, Payload, event::Event};
use halo::agent::{message::Role, model::chat, provider::deepseek::DeepSeek};
use halo::harness;
use serde_json::{Map, Value, json};
use tempfile::tempdir;

fn chat() -> impl chat::Chat {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real Chat")
}

fn input(chat_id: i64, session_id: i64, text: &str) -> Input {
    Input {
        metadata: Map::from_iter([
            ("chat_id".into(), json!(chat_id)),
            ("session_id".into(), json!(session_id)),
        ]),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value": text}),
            mime_type: None,
        }],
    }
}

fn item(
    id: &str,
    session_id: i64,
    content: &str,
    kind: harness::state::Kind,
) -> harness::state::Item {
    harness::state::Item {
        id: id.into(),
        kind,
        title: id.into(),
        content: Value::String(content.into()),
        scope: harness::state::Scope::Session,
        session_id: Some(session_id),
        source: Some("acceptance".into()),
        version: 0,
        created_time: 1,
        updated_time: 1,
        metadata: Map::new(),
    }
}

fn change(id: &str) -> harness::state::Change {
    harness::state::Change {
        id: id.into(),
        trigger: "acceptance".into(),
        version: 0,
        changes: vec![id.into()],
        evidence: vec!["live-model-request".into()],
        outcome: "accepted".into(),
        created_time: 1,
    }
}

fn text(message: &Message) -> String {
    message
        .content
        .iter()
        .filter(|part| part.r#type == "text")
        .filter_map(|part| part.data.get("value").and_then(Value::as_str))
        .collect()
}

async fn complete(agent: &impl Agent, input: Input) -> Result<String, Error> {
    let events = agent.run(input, false, Cancellation::new()).await?;
    futures_util::pin_mut!(events);
    let mut answer = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event?
            && message.role == Role::Assistant
        {
            answer = Some(text(&message));
        }
    }
    answer.ok_or_else(|| Error::new("INVALID_RESPONSE", "live Harness run returned no answer"))
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_harness_refreshes_context_after_refinement() {
    let directory = tempdir().expect("create local Harness Workspace");
    let workspace = halo::agent::core::workspace::Local::new(directory.path()).unwrap();
    let writer = harness::state::Store::new(workspace.clone());
    let session_id = 77_001;
    writer
        .refine(
            item(
                "marker",
                session_id,
                "当前标记是 ORBIT-ALPHA。回答时只输出当前标记。",
                harness::state::Kind::Memory,
            ),
            change("initial"),
            &Cancellation::new(),
        )
        .await
        .unwrap();

    let agent = harness::builder()
        .model(chat())
        .state(harness::state::Store::new(workspace.clone()))
        .system_prompt("只根据当前 Harness Context 回答，不要猜测或保留旧版本。")
        .build()
        .unwrap();
    let first = complete(
        &agent,
        input(77_001, session_id, "当前标记是什么？只输出标记本身。"),
    )
    .await
    .unwrap();
    assert!(first.contains("ORBIT-ALPHA"), "first live answer: {first}");

    writer
        .refine(
            item(
                "marker",
                session_id,
                "当前标记是 ORBIT-BETA。回答时只输出当前标记。",
                harness::state::Kind::Memory,
            ),
            change("updated"),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    let second = complete(
        &agent,
        input(77_002, session_id, "当前标记是什么？只输出标记本身。"),
    )
    .await
    .unwrap();
    assert!(
        second.contains("ORBIT-BETA"),
        "second live answer: {second}"
    );
    assert!(
        !second.contains("ORBIT-ALPHA"),
        "stale context leaked: {second}"
    );
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY to construct the real provider"]
async fn harness_rejects_an_unregistered_persisted_subagent_before_model_call() {
    let directory = tempdir().expect("create local Harness Workspace");
    let workspace = halo::agent::core::workspace::Local::new(directory.path()).unwrap();
    harness::state::Store::new(workspace.clone())
        .refine(
            item(
                "missing-member",
                77_101,
                "不存在的成员",
                harness::state::Kind::Subagent,
            ),
            change("unknown-member"),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    let agent = harness::builder()
        .model(chat())
        .state(harness::state::Store::new(workspace))
        .build()
        .unwrap();
    let saved = agent
        .load_state(&Cancellation::new())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(saved.subagents(), vec!["missing-member"]);
    assert!(
        saved.context_for(Some(77_101), &[]).is_err(),
        "persisted unknown SubAgent should fail context validation"
    );
    let result = agent
        .run(
            input(77_101, 77_101, "回答一个简单问题。"),
            false,
            Cancellation::new(),
        )
        .await;
    let mut events = match result {
        Ok(events) => events,
        Err(error) => {
            assert_eq!(error.code, "INVALID_ARGUMENTS");
            return;
        }
    };
    let error = match events.next().await {
        Some(Err(error)) => error,
        Some(Ok(_)) => panic!("an unknown persisted SubAgent must fail before the model call"),
        None => panic!("unknown persisted SubAgent stream ended without an error"),
    };
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_harness_goal_is_refreshed_before_each_model_request() {
    let directory = tempdir().expect("create local Harness Workspace");
    let workspace = halo::agent::core::workspace::Local::new(directory.path()).unwrap();
    let writer = harness::state::Store::new(workspace.clone());
    let session_id = 77_201;
    writer
        .update_goal(
            harness::state::Goal {
                id: "goal-1".into(),
                objective: "GOAL-OMEGA：完成这项任务".into(),
                status: harness::state::GoalStatus::Active,
                created_time: 1,
                updated_time: 1,
            },
            change("goal-active"),
            &Cancellation::new(),
        )
        .await
        .unwrap();

    let agent = harness::builder()
        .model(chat())
        .state(harness::state::Store::new(workspace.clone()))
        .system_prompt("只根据当前 Context 回答。")
        .build()
        .unwrap();
    let active = complete(
        &agent,
        input(
            77_201,
            session_id,
            "当前任务目标是什么？只输出目标中的英文标记。",
        ),
    )
    .await
    .unwrap();
    assert!(
        active.contains("GOAL-OMEGA"),
        "active goal answer: {active}"
    );

    writer
        .update_goal(
            harness::state::Goal {
                id: "goal-1".into(),
                objective: "GOAL-OMEGA：完成这项任务".into(),
                status: harness::state::GoalStatus::Paused,
                created_time: 1,
                updated_time: 2,
            },
            change("goal-paused"),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    let paused = complete(
        &agent,
        input(77_202, session_id, "当前是否有 active goal？只回答有或无。"),
    )
    .await
    .unwrap();
    assert!(paused.contains('无'), "paused goal was injected: {paused}");
}
