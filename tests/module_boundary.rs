use std::path::{Path, PathBuf};

fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn source(root: &str) -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut content = String::new();
    for path in rust_files(&manifest.join(root)) {
        content.push_str(&std::fs::read_to_string(path).expect("module source is readable"));
        content.push('\n');
    }
    content
}

#[test]
fn core_does_not_depend_on_runtime() {
    let source = source("src/core");
    assert!(!source.contains("crate::runtime"));
    assert!(!source.contains("use crate::runtime"));
}

#[test]
fn components_do_not_depend_on_business_layers() {
    let source = source("src/components");
    assert!(!source.contains("crate::core"));
    assert!(!source.contains("crate::runtime"));
    assert!(!source.contains("crate::harness"));
}

#[test]
fn sandbox_implementation_lives_in_core() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(manifest.join("src/core/sandbox/local.rs").is_file());
    assert!(!manifest.join("src/components/sandbox/local.rs").exists());

    let components = source("src/components/sandbox.rs");
    assert!(!components.contains("pub mod local"));
    assert!(!components.contains("pub use local::Local"));
}
