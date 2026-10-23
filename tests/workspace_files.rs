use std::{path::PathBuf, sync::Arc};

use futures_util::future::join_all;
use halo::agent::{Cancellation, Error, Message};
use halo::{agent::toolkit::Toolkit, core, tool::Call};
use serde_json::{Value, json};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "heart-file-tools-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

async fn run(capabilities: &Toolkit, name: &str, arguments: Value) -> Result<Message, Error> {
    capabilities
        .execute(
            Call {
                call_id: format!("call_{name}"),
                name: format!("workspace_{name}"),
                arguments: arguments.as_object().unwrap().clone(),
            },
            &Cancellation::new(),
        )
        .await
        .map(|(message, _)| message)
}

async fn populate(workspace: &core::workspace::Local) {
    let cancellation = Cancellation::new();
    for (path, content) in [
        ("readme.txt", "alpha\nbeta\nbeta\n"),
        ("src/a.rs", "first\nsecond\nlast\n"),
        ("src/deep/b.rs", "ALPHA\nBeta\n"),
        (".git/config", "alpha secret\n"),
    ] {
        workspace
            .write(path, content.as_bytes().to_vec(), &cancellation)
            .await
            .unwrap();
    }
    workspace
        .write(
            "binary.dat",
            vec![0, b'a', b'l', b'p', b'h', b'a'],
            &cancellation,
        )
        .await
        .unwrap();
    workspace
        .write("invalid.dat", vec![255, 254], &cancellation)
        .await
        .unwrap();
}

async fn check_listing(workspace: &core::workspace::Local) {
    let cancellation = Cancellation::new();
    assert_eq!(
        workspace.list(".", false, &cancellation).await.unwrap(),
        [".git", "binary.dat", "invalid.dat", "readme.txt", "src"]
    );
    assert_eq!(
        workspace.list("src", false, &cancellation).await.unwrap(),
        ["src/a.rs", "src/deep"]
    );
    assert_eq!(
        workspace.list("src", true, &cancellation).await.unwrap(),
        ["src/a.rs", "src/deep/b.rs"]
    );
    assert_eq!(
        workspace
            .list("missing", true, &cancellation)
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(
        workspace
            .list("readme.txt", false, &cancellation)
            .await
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    for path in ["../escape", "/outside", "src/../escape"] {
        assert_eq!(
            workspace
                .list(path, true, &cancellation)
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
}

#[tokio::test]
async fn native_local_list_has_stable_relative_paths_and_rejects_invalid_directories() {
    let directory = Directory::new();
    let local = core::workspace::Local::new(&directory.0).unwrap();
    populate(&local).await;
    check_listing(&local).await;
}

#[tokio::test]
async fn list_and_glob_filter_nested_files_and_paginate_in_stable_order() {
    let directory = Directory::new();
    let workspace = core::workspace::Local::new(&directory.0).unwrap();
    populate(&workspace).await;
    let capabilities = Toolkit::from(workspace.toolkit(Toolkit::new()));
    let result = run(&capabilities, "list", json!({"path":"src"}))
        .await
        .unwrap();
    assert_eq!(result.values["entries"], json!(["src/a.rs", "src/deep"]));
    let result = run(
        &capabilities,
        "glob",
        json!({"pattern":"**/*.{rs,txt}","offset":1,"head_limit":1}),
    )
    .await
    .unwrap();
    assert_eq!(result.values["files"], json!(["src/a.rs"]));
    assert_eq!(result.values["truncated"], true);
    let result = run(
        &capabilities,
        "glob",
        json!({"path":"src","pattern":"*.rs","head_limit":0}),
    )
    .await
    .unwrap();
    assert_eq!(result.values["files"], json!(["src/a.rs", "src/deep/b.rs"]));
    let result = run(
        &capabilities,
        "glob",
        json!({"path":"src","pattern":"deep/?.rs"}),
    )
    .await
    .unwrap();
    assert_eq!(result.values["files"], json!(["src/deep/b.rs"]));
    let result = run(&capabilities, "list", json!({"offset":999}))
        .await
        .unwrap();
    assert_eq!(result.values["entries"], json!([]));
    assert_eq!(result.values["truncated"], false);
}

#[tokio::test]
async fn grep_supports_modes_case_filtering_single_files_and_skips_binary_vcs_data() {
    let directory = Directory::new();
    let workspace = core::workspace::Local::new(&directory.0).unwrap();
    populate(&workspace).await;
    let capabilities = Toolkit::from(workspace);
    let result = run(
        &capabilities,
        "grep",
        json!({"pattern":"^alpha$","case_insensitive":true}),
    )
    .await
    .unwrap();
    assert_eq!(
        result.values["files"],
        json!(["readme.txt", "src/deep/b.rs"])
    );
    let result = run(
        &capabilities,
        "grep",
        json!({"pattern":"beta","case_insensitive":true,"output_mode":"count"}),
    )
    .await
    .unwrap();
    assert_eq!(
        result.values["counts"],
        json!([{"path":"readme.txt","count":2},{"path":"src/deep/b.rs","count":1}])
    );
    let result = run(
        &capabilities,
        "grep",
        json!({"pattern":"beta","case_insensitive":true,"glob":"*.rs"}),
    )
    .await
    .unwrap();
    assert_eq!(result.values["files"], json!(["src/deep/b.rs"]));
    let result = run(&capabilities, "grep", json!({"path":"readme.txt","pattern":"beta","output_mode":"content","head_limit":1,"offset":1})).await.unwrap();
    assert_eq!(
        result.values["matches"],
        json!([{"path":"readme.txt","line":3,"content":"beta","matched":true}])
    );
    assert_eq!(result.values["truncated"], false);
    let result = run(&capabilities, "grep", json!({"pattern":"not-present"}))
        .await
        .unwrap();
    assert_eq!(result.values["files"], json!([]));
}

#[tokio::test]
async fn grep_multiline_and_overlapping_context_keep_original_line_numbers() {
    let directory = Directory::new();
    let workspace = core::workspace::Local::new(&directory.0).unwrap();
    populate(&workspace).await;
    let capabilities = Toolkit::from(workspace);
    let result = run(&capabilities, "grep", json!({"path":"src/a.rs","pattern":"first.*second","multiline":true,"output_mode":"content","context":1})).await.unwrap();
    assert_eq!(
        result.values["matches"],
        json!([
            {"path":"src/a.rs","line":1,"content":"first","matched":true},
            {"path":"src/a.rs","line":2,"content":"second","matched":true},
            {"path":"src/a.rs","line":3,"content":"last","matched":false}
        ])
    );
    let result = run(
        &capabilities,
        "grep",
        json!({"path":"readme.txt","pattern":"beta","output_mode":"content","context":1}),
    )
    .await
    .unwrap();
    assert_eq!(result.values["matches"].as_array().unwrap().len(), 3);
    let result = run(
        &capabilities,
        "grep",
        json!({"path":"src/a.rs","pattern":"first.*second"}),
    )
    .await
    .unwrap();
    assert_eq!(result.values["files"], json!([]));
}

#[tokio::test]
async fn exact_edits_preserve_files_on_missing_or_ambiguous_matches() {
    let directory = Directory::new();
    let capabilities = Toolkit::from(core::workspace::Local::new(&directory.0).unwrap());
    run(
        &capabilities,
        "write",
        json!({"path":"draft.txt","content":"alpha alpha"}),
    )
    .await
    .unwrap();
    for (old, new, code) in [
        ("missing", "x", "NOT_FOUND"),
        ("alpha", "x", "CONFLICT"),
        ("", "x", "INVALID_ARGUMENTS"),
        ("alpha", "alpha", "INVALID_ARGUMENTS"),
    ] {
        let error = run(
            &capabilities,
            "edit",
            json!({"path":"draft.txt","old_string":old,"new_string":new}),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code, code);
        let result = run(&capabilities, "read", json!({"path":"draft.txt"}))
            .await
            .unwrap();
        assert_eq!(result.values["content"], "alpha alpha");
    }
    let result = run(
        &capabilities,
        "edit",
        json!({"path":"draft.txt","old_string":"alpha","new_string":"β","replace_all":true}),
    )
    .await
    .unwrap();
    assert_eq!(result.values["replacements"], 2);
    let result = run(&capabilities, "read", json!({"path":"draft.txt"}))
        .await
        .unwrap();
    assert_eq!(result.values["content"], "β β");
}

#[tokio::test]
async fn concurrent_local_edits_do_not_overwrite_each_others_successful_changes() {
    let directory = Directory::new();
    let capabilities = Arc::new(Toolkit::from(
        core::workspace::Local::new(&directory.0).unwrap(),
    ));
    let original = (0..20)
        .map(|idx| format!("[token:{idx}]\n"))
        .collect::<String>();
    run(
        &capabilities,
        "write",
        json!({"path":"draft.txt","content":original}),
    )
    .await
    .unwrap();
    let jobs = (0..20).map(|idx| {
        let capabilities = Arc::clone(&capabilities);
        async move {
            run(&capabilities, "edit", json!({"path":"draft.txt","old_string":format!("[token:{idx}]"),"new_string":format!("[done:{idx}]")})).await
        }
    });
    let results = join_all(jobs).await;
    assert!(results.iter().all(Result::is_ok));
    let result = run(&capabilities, "read", json!({"path":"draft.txt"}))
        .await
        .unwrap();
    let content = result.values["content"].as_str().unwrap();
    assert!(!content.contains("[token:"));
    for idx in 0..20 {
        assert!(content.contains(&format!("[done:{idx}]")));
    }
}

#[tokio::test]
async fn invalid_search_and_scope_arguments_return_errors_without_commands() {
    let directory = Directory::new();
    let capabilities = Toolkit::from(core::workspace::Local::new(&directory.0).unwrap());
    for (name, args) in [
        ("glob", json!({"pattern":"["})),
        ("glob", json!({"pattern":"../*"})),
        ("glob", json!({"pattern":"*","path":"../escape"})),
        ("grep", json!({"pattern":"("})),
        ("grep", json!({"pattern":"x","output_mode":"unsupported"})),
        ("grep", json!({"pattern":"x","head_limit":-1})),
        (
            "grep",
            json!({"pattern":"x","path":"/outside","glob":"*.rs"}),
        ),
        (
            "grep",
            json!({"pattern":"x","path":"../escape","glob":"*.rs"}),
        ),
        ("list", json!({"path":"/outside"})),
    ] {
        assert_eq!(
            run(&capabilities, name, args).await.unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let error = capabilities
        .execute(
            Call {
                call_id: "cancelled".into(),
                name: "workspace_list".into(),
                arguments: Default::default(),
            },
            &cancellation,
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, "CANCELLED");
}

#[cfg(unix)]
#[tokio::test]
async fn local_listing_glob_and_grep_do_not_follow_outside_symlinks() {
    let inside = Directory::new();
    let outside = Directory::new();
    std::fs::write(outside.0.join("secret.txt"), "outside secret").unwrap();
    std::os::unix::fs::symlink(&outside.0, inside.0.join("linked")).unwrap();
    let workspace = core::workspace::Local::new(&inside.0).unwrap();
    assert!(
        workspace
            .list(".", true, &Cancellation::new())
            .await
            .unwrap()
            .is_empty()
    );
    let capabilities = Toolkit::from(workspace);
    let result = run(&capabilities, "glob", json!({"pattern":"**/*"}))
        .await
        .unwrap();
    assert_eq!(result.values["files"], json!([]));
    let result = run(&capabilities, "grep", json!({"pattern":"secret"}))
        .await
        .unwrap();
    assert_eq!(result.values["files"], json!([]));
    assert_eq!(
        run(
            &capabilities,
            "grep",
            json!({"path":"linked","pattern":"secret"})
        )
        .await
        .unwrap_err()
        .code,
        "PERMISSION_DENIED"
    );
}
