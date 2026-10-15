use std::{
    future::pending,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures_executor::block_on;
use halo_agents::{
    Cancellation, Error, Message, Messages, core, message::Role, skill::Skill as _,
    workspace::Workspace as _,
};
use serde_json::json;

const MARKDOWN: &str = "---\nname: research\ndescription: >-\n  Research the requested topic\n  and report its sources.\nmetadata:\n  author: user\nallowed-tools: search\n---\n# Instructions\nUse explicit sources.\n";

fn echo(
    input: Messages,
    _: Cancellation,
) -> impl std::future::Future<Output = Result<Messages, Error>> {
    std::future::ready(Ok(input))
}

#[test]
fn skill_parses_yaml_and_retains_definition_metadata() {
    let skill = core::skill::Skill::from_markdown(MARKDOWN, echo).unwrap();
    assert_eq!(skill.definition().name, "research");
    assert_eq!(
        skill.definition().description.as_deref(),
        Some("Research the requested topic and report its sources.")
    );
    assert_eq!(skill.definition().metadata["author"], json!("user"));
    assert_eq!(
        skill.definition().metadata["allowed-tools"],
        json!("search")
    );
}

#[test]
fn selected_skill_passes_instructions_and_input_to_explicit_executor() {
    let skill = core::skill::Skill::from_markdown(MARKDOWN, echo).unwrap();
    let mut user = Message::new(Role::User);
    user.values.insert("query".into(), json!("weather"));
    user.metadata.insert("chat_id".into(), json!(1));
    let output = block_on(skill.run(Messages::new([user.clone()]), &Cancellation::new())).unwrap();
    assert_eq!(output.0.len(), 2);
    assert_eq!(output.0[0].role, Role::System);
    assert_eq!(
        output.0[0].content[0].data["value"],
        json!("# Instructions\nUse explicit sources.\n")
    );
    assert_eq!(output.0[1], user);
}

#[test]
fn skill_rejects_invalid_headers_and_does_not_execute_while_loading() {
    let count = Arc::new(AtomicUsize::new(0));
    let calls = count.clone();
    let executor = move |input: Messages, _: Cancellation| {
        calls.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok::<Messages, Error>(input))
    };
    core::skill::Skill::from_markdown(MARKDOWN, executor).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    for markdown in [
        "no header",
        "---\nname: research\n",
        "---\nname: Bad_Name\ndescription: test\n---\ntext",
        "---\nname: research\ndescription: \"\"\n---\ntext",
        "---\nname: research\nname: duplicate\ndescription: test\n---\ntext",
        "---\nname: research\ndescription: test\n---\n",
    ] {
        assert!(
            core::skill::Skill::from_markdown(markdown, echo).is_err(),
            "{markdown}"
        );
    }
}

#[test]
fn skill_accepts_crlf_and_bom_without_changing_instructions() {
    let markdown = format!("\u{feff}{}", MARKDOWN.replace('\n', "\r\n"));
    let skill = core::skill::Skill::from_markdown(&markdown, echo).unwrap();
    assert_eq!(skill.definition().name, "research");
    let output = block_on(skill.run(Messages::default(), &Cancellation::new())).unwrap();
    assert_eq!(
        output.0[0].content[0].data["value"],
        json!("# Instructions\r\nUse explicit sources.\r\n")
    );
}

#[test]
fn skill_loads_from_server_workspace_and_checks_directory_name() {
    let workspace = core::workspace::InMemory::new();
    let cancel = Cancellation::new();
    block_on(workspace.write("research/SKILL.md", MARKDOWN.as_bytes().to_vec(), &cancel)).unwrap();
    let skill = block_on(core::skill::Skill::load(
        &workspace,
        "research/SKILL.md",
        echo,
        &cancel,
    ))
    .unwrap();
    assert_eq!(skill.definition().name, "research");
    block_on(workspace.write("other/SKILL.md", MARKDOWN.as_bytes().to_vec(), &cancel)).unwrap();
    assert_eq!(
        block_on(core::skill::Skill::load(
            &workspace,
            "other/SKILL.md",
            echo,
            &cancel
        ))
        .err()
        .unwrap()
        .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(
        block_on(core::skill::Skill::load(
            &workspace,
            "missing/SKILL.md",
            echo,
            &cancel
        ))
        .err()
        .unwrap()
        .code,
        "NOT_FOUND"
    );
}

#[test]
fn skill_returns_executor_error_without_wrapping() {
    let skill = core::skill::Skill::from_markdown(MARKDOWN, |_: Messages, _: Cancellation| async {
        Err(Error::new("BUSINESS_ERROR", "original").with_details("detail"))
    })
    .unwrap();
    assert_eq!(
        block_on(skill.run(Messages::default(), &Cancellation::new())).unwrap_err(),
        Error::new("BUSINESS_ERROR", "original").with_details("detail")
    );
}

#[tokio::test]
async fn skill_cancellation_wakes_executor_and_does_not_return_partial_messages() {
    let skill = core::skill::Skill::from_markdown(MARKDOWN, |_: Messages, _: Cancellation| {
        pending::<Result<Messages, Error>>()
    })
    .unwrap();
    let cancel = Cancellation::new();
    let token = cancel.clone();
    let task = tokio::spawn(async move {
        tokio::task::yield_now().await;
        token.cancel();
    });
    assert_eq!(
        tokio::time::timeout(
            Duration::from_secs(1),
            skill.run(Messages::default(), &cancel)
        )
        .await
        .unwrap()
        .unwrap_err()
        .code,
        "CANCELLED"
    );
    task.await.unwrap();
}
