use futures_executor::block_on;
use halo::agent::Cancellation;
use halo::{agent::core, agent::skill::Skill as _};
use serde_json::json;

const MARKDOWN: &str = "---\nname: research\ndescription: >-\n  Research the requested topic\n  and report its sources.\nmetadata:\n  author: user\nallowed-tools: search\n---\n# Instructions\nUse explicit sources.\n";

#[test]
fn skill_parses_yaml_and_retains_definition_metadata() {
    let skill = core::skill::Skill::from_markdown(MARKDOWN).unwrap();
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
    assert_eq!(
        block_on(skill.read(&Cancellation::new())).unwrap(),
        "# Instructions\nUse explicit sources.\n"
    );
}

#[test]
fn skill_rejects_invalid_headers_and_empty_instructions() {
    for markdown in [
        "no header",
        "---\nname: research\n",
        "---\nname: Bad_Name\ndescription: test\n---\ntext",
        "---\nname: research\ndescription: \"\"\n---\ntext",
        "---\nname: research\nname: duplicate\ndescription: test\n---\ntext",
        "---\nname: research\ndescription: test\n---\n",
    ] {
        assert!(
            core::skill::Skill::from_markdown(markdown).is_err(),
            "{markdown}"
        );
    }
}

#[test]
fn skill_accepts_crlf_and_bom_without_changing_instructions() {
    let markdown = format!("\u{feff}{}", MARKDOWN.replace('\n', "\r\n"));
    let skill = core::skill::Skill::from_markdown(&markdown).unwrap();
    assert_eq!(skill.definition().name, "research");
    assert_eq!(
        block_on(skill.read(&Cancellation::new())).unwrap(),
        "# Instructions\r\nUse explicit sources.\r\n"
    );
}

#[tokio::test]
async fn skill_loads_from_local_workspace_and_checks_directory_name() {
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let directory = Directory(std::env::temp_dir().join(format!(
        "heart-skill-local-{}-{}",
        std::process::id(),
        rand::random::<u64>()
    )));
    std::fs::create_dir(&directory.0).unwrap();
    let workspace = core::workspace::Local::new(&directory.0).unwrap();
    let cancel = Cancellation::new();
    workspace
        .write("research/SKILL.md", MARKDOWN.as_bytes().to_vec(), &cancel)
        .await
        .unwrap();
    let skill = core::skill::Skill::load(&workspace, "research/SKILL.md", &cancel)
        .await
        .unwrap();
    assert_eq!(skill.definition().name, "research");
    workspace
        .write("other/SKILL.md", MARKDOWN.as_bytes().to_vec(), &cancel)
        .await
        .unwrap();
    assert_eq!(
        core::skill::Skill::load(&workspace, "other/SKILL.md", &cancel)
            .await
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(
        core::skill::Skill::load(&workspace, "missing/SKILL.md", &cancel)
            .await
            .err()
            .unwrap()
            .code,
        "NOT_FOUND"
    );
}

#[test]
fn skill_read_observes_cancellation() {
    let skill = core::skill::Skill::from_markdown(MARKDOWN).unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        block_on(skill.read(&cancellation)).unwrap_err().code,
        "CANCELLED"
    );
}
