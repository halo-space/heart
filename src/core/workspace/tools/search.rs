use std::sync::Arc;

use globset::{GlobBuilder, GlobMatcher};
use regex::RegexBuilder;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use super::{
    Call, Files, Message, Tool, ToolDefinition, arguments, definition, reading, tool_message,
};
use crate::{Cancellation, Error, core::operation::check};

fn root() -> String {
    ".".into()
}
fn default_limit() -> usize {
    250
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListArguments {
    #[serde(default = "root")]
    path: String,
    #[serde(default)]
    recursive: bool,
    #[serde(default = "default_limit")]
    head_limit: usize,
    #[serde(default)]
    offset: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobArguments {
    pattern: String,
    #[serde(default = "root")]
    path: String,
    #[serde(default = "default_limit")]
    head_limit: usize,
    #[serde(default)]
    offset: usize,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Content,
    #[default]
    FilesWithMatches,
    Count,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GrepArguments {
    pattern: String,
    #[serde(default = "root")]
    path: String,
    #[serde(default)]
    output_mode: Mode,
    glob: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    multiline: bool,
    #[serde(default)]
    context: usize,
    #[serde(default = "default_limit")]
    head_limit: usize,
    #[serde(default)]
    offset: usize,
}

fn properties() -> Value {
    json!({
        "path":{"type":"string","default":".","description":"Workspace-relative directory; . denotes the root. Grep also accepts a file."},
        "head_limit":{"type":"integer","minimum":0,"default":250,"description":"Maximum returned records; 0 means unlimited."},
        "offset":{"type":"integer","minimum":0,"default":0}
    })
}

fn schema(extra: Value, required: &[&str]) -> Value {
    let mut properties = properties();
    properties
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    json!({"type":"object","properties":properties,"required":required,"additionalProperties":false})
}

pub(super) struct List {
    workspace: Arc<Files>,
    access: Arc<RwLock<()>>,
    definition: ToolDefinition,
}
pub(super) struct Glob {
    workspace: Arc<Files>,
    access: Arc<RwLock<()>>,
    definition: ToolDefinition,
}
pub(super) struct Grep {
    workspace: Arc<Files>,
    access: Arc<RwLock<()>>,
    definition: ToolDefinition,
}

impl List {
    pub(super) fn new(workspace: Arc<Files>, access: Arc<RwLock<()>>) -> Self {
        Self {
            workspace,
            access,
            definition: definition(
                "workspace_list",
                "List workspace-relative paths in stable lexical order. Non-recursive listing includes immediate files and directories; recursive listing returns files only. Symbolic links are not followed.",
                schema(json!({"recursive":{"type":"boolean","default":false}}), &[]),
            ),
        }
    }
}
impl Glob {
    pub(super) fn new(workspace: Arc<Files>, access: Arc<RwLock<()>>) -> Self {
        Self {
            workspace,
            access,
            definition: definition(
                "workspace_glob",
                "Find workspace files with *, ?, **, character classes and brace alternatives. Patterns are relative to path; patterns without a slash match basenames at any depth. Results have stable lexical ordering.",
                schema(
                    json!({"pattern":{"type":"string","minLength":1}}),
                    &["pattern"],
                ),
            ),
        }
    }
}
impl Grep {
    pub(super) fn new(workspace: Arc<Files>, access: Arc<RwLock<()>>) -> Self {
        Self {
            workspace,
            access,
            definition: definition(
                "workspace_grep",
                "Search UTF-8 workspace files with Rust regular expressions. Return files_with_matches (default), content with 1-based line numbers and optional context, or matching-line count per file. Supports glob filtering, case_insensitive and multiline. Skips binary/non-UTF-8 files and VCS metadata; does not apply gitignore rules. Records sort by path then line. No shell is required.",
                schema(
                    json!({
                        "pattern":{"type":"string"},
                        "output_mode":{"type":"string","enum":["content","files_with_matches","count"],"default":"files_with_matches"},
                        "glob":{"type":"string","minLength":1},
                        "case_insensitive":{"type":"boolean","default":false},
                        "multiline":{"type":"boolean","default":false},
                        "context":{"type":"integer","minimum":0,"default":0}
                    }),
                    &["pattern"],
                ),
            ),
        }
    }
}

impl Tool for List {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let args: ListArguments = arguments(call)?;
        let _guard = reading(&self.access, cancellation).await?;
        let paths = self
            .workspace
            .list(&args.path, args.recursive, cancellation)
            .await?;
        let (entries, truncated) = page(paths, args.offset, args.head_limit);
        Ok(tool_message(
            json!({"entries":entries,"truncated":truncated}),
        ))
    }
}

impl Tool for Glob {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let args: GlobArguments = arguments(call)?;
        let pattern = pattern(&args.pattern)?;
        let _guard = reading(&self.access, cancellation).await?;
        let paths = self.workspace.list(&args.path, true, cancellation).await?;
        let mut matches = Vec::new();
        for path in paths {
            check(cancellation)?;
            if matches_path(&pattern, &args.pattern, &args.path, &path) {
                matches.push(path);
            }
        }
        let (files, truncated) = page(matches, args.offset, args.head_limit);
        Ok(tool_message(json!({"files":files,"truncated":truncated})))
    }
}

impl Tool for Grep {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let args: GrepArguments = arguments(call)?;
        let regex = RegexBuilder::new(&args.pattern)
            .case_insensitive(args.case_insensitive)
            .multi_line(true)
            .dot_matches_new_line(args.multiline)
            .build()
            .map_err(|error| {
                Error::new("INVALID_ARGUMENTS", "invalid search regular expression")
                    .with_details(error.to_string())
            })?;
        let glob = args.glob.as_deref().map(pattern).transpose()?;
        let _guard = reading(&self.access, cancellation).await?;
        let (paths, mut single_file) =
            match self.workspace.list(&args.path, true, cancellation).await {
                Ok(paths) => (paths, None),
                Err(error) if error.code == "INVALID_ARGUMENTS" => {
                    // Validate a file before filters can skip it: an absolute or
                    // traversal path must not become an empty successful search.
                    let bytes = self.workspace.read(&args.path, cancellation).await?;
                    (vec![args.path.clone()], Some(bytes))
                }
                Err(error) => return Err(error),
            };
        let maximum = if args.head_limit == 0 {
            usize::MAX
        } else {
            args.offset
                .saturating_add(args.head_limit)
                .saturating_add(1)
        };
        let mut records = Vec::new();
        for path in paths {
            check(cancellation)?;
            if path
                .split('/')
                .any(|part| matches!(part, ".git" | ".hg" | ".svn" | ".bzr"))
            {
                continue;
            }
            if let (Some(glob), Some(raw)) = (&glob, &args.glob)
                && !matches_path(glob, raw, &args.path, &path)
            {
                continue;
            }
            let bytes = match single_file.take() {
                Some(bytes) => bytes,
                None => self.workspace.read(&path, cancellation).await?,
            };
            if bytes.contains(&0) {
                continue;
            }
            let Ok(content) = String::from_utf8(bytes) else {
                continue;
            };
            let lines: Vec<_> = content.lines().collect();
            let mut matched = vec![false; lines.len()];
            if args.multiline {
                let mut starts = vec![0];
                starts.extend(content.match_indices('\n').map(|(idx, _)| idx + 1));
                for found in regex.find_iter(&content) {
                    check(cancellation)?;
                    let from = starts
                        .partition_point(|start| *start <= found.start())
                        .saturating_sub(1);
                    let to = starts
                        .partition_point(|start| {
                            *start <= found.end().saturating_sub(1).max(found.start())
                        })
                        .saturating_sub(1);
                    for value in matched.iter_mut().take(to.saturating_add(1)).skip(from) {
                        *value = true;
                    }
                }
            } else {
                for (line, value) in lines.iter().zip(&mut matched) {
                    check(cancellation)?;
                    *value = regex.is_match(line);
                }
            }
            let count = matched.iter().filter(|value| **value).count();
            if count == 0 {
                continue;
            }
            match args.output_mode {
                Mode::FilesWithMatches => records.push(json!(path)),
                Mode::Count => records.push(json!({"path":path,"count":count})),
                Mode::Content => {
                    let mut ranges: Vec<(usize, usize)> = Vec::new();
                    for (idx, value) in matched.iter().enumerate() {
                        if *value {
                            let from = idx.saturating_sub(args.context);
                            let to = idx
                                .saturating_add(args.context)
                                .saturating_add(1)
                                .min(matched.len());
                            if let Some(last) = ranges.last_mut().filter(|last| last.1 >= from) {
                                last.1 = to;
                            } else {
                                ranges.push((from, to));
                            }
                        }
                    }
                    for (from, to) in ranges {
                        for idx in from..to {
                            check(cancellation)?;
                            records.push(json!({"path":path,"line":idx+1,"content":lines[idx],"matched":matched[idx]}));
                            if records.len() >= maximum {
                                break;
                            }
                        }
                        if records.len() >= maximum {
                            break;
                        }
                    }
                }
            }
            if records.len() >= maximum {
                break;
            }
        }
        let (records, truncated) = page(records, args.offset, args.head_limit);
        let key = match args.output_mode {
            Mode::Content => "matches",
            Mode::FilesWithMatches => "files",
            Mode::Count => "counts",
        };
        let mut result = serde_json::Map::new();
        result.insert(key.into(), json!(records));
        result.insert("truncated".into(), json!(truncated));
        Ok(tool_message(Value::Object(result)))
    }
}

fn pattern(raw: &str) -> Result<GlobMatcher, Error> {
    if raw.is_empty()
        || raw.starts_with('/')
        || raw.contains('\0')
        || raw.contains('\\')
        || raw.split('/').any(|part| part == "..")
    {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "glob must be a workspace-relative pattern",
        ));
    }
    GlobBuilder::new(raw)
        .literal_separator(true)
        .backslash_escape(false)
        .build()
        .map(|glob| glob.compile_matcher())
        .map_err(|error| {
            Error::new("INVALID_ARGUMENTS", "invalid glob pattern").with_details(error.to_string())
        })
}

fn matches_path(pattern: &GlobMatcher, raw: &str, base: &str, path: &str) -> bool {
    let relative = if base == "." {
        path
    } else {
        path.strip_prefix(base)
            .and_then(|rest| rest.strip_prefix('/'))
            .unwrap_or(path)
    };
    let candidate = if raw.contains('/') {
        relative
    } else {
        relative.rsplit('/').next().unwrap_or(relative)
    };
    pattern.is_match(candidate)
}

fn page<T>(items: Vec<T>, offset: usize, limit: usize) -> (Vec<T>, bool) {
    let available = items.len().saturating_sub(offset);
    let count = if limit == 0 {
        available
    } else {
        limit.min(available)
    };
    (
        items.into_iter().skip(offset).take(count).collect(),
        available > count,
    )
}
