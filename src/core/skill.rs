use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    Cancellation, Error, Message, Messages,
    function::Function,
    message::Role,
    model::content::Part,
    skill::{Definition, Skill as Contract},
    workspace::Workspace,
};

/// A selected Skill's instructions and explicit executor. Loading a Skill does
/// not execute scripts, fetch references, or grant tool/permission access.
pub struct Skill<F> {
    definition: Definition,
    instructions: String,
    executor: F,
}

impl<F> Skill<F> {
    pub fn new(definition: Definition, instructions: String, executor: F) -> Result<Self, Error> {
        if definition.name.trim().is_empty() || instructions.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "skill name and instructions are required",
            ));
        }
        Ok(Self {
            definition,
            instructions,
            executor,
        })
    }

    /// Parses standard SKILL.md YAML frontmatter followed by Markdown. The
    /// instructions stay private; metadata remains metadata, not an executor.
    pub fn from_markdown(markdown: &str, executor: F) -> Result<Self, Error> {
        if markdown.len() > 4 * 1024 * 1024 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "skill document is too large",
            ));
        }
        let markdown = markdown.strip_prefix('\u{feff}').unwrap_or(markdown);
        let mut lines = markdown.split_inclusive('\n');
        if lines.next().map(str::trim_end) != Some("---") {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "skill requires YAML frontmatter",
            ));
        }
        let mut header = String::new();
        let mut closed = false;
        for line in lines.by_ref() {
            if line.trim_end() == "---" {
                closed = true;
                break;
            }
            header.push_str(line);
        }
        if !closed {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "skill frontmatter is not closed",
            ));
        }
        #[derive(Deserialize)]
        struct Header {
            name: String,
            description: String,
            #[serde(default)]
            metadata: Map<String, Value>,
            #[serde(flatten)]
            extra: Map<String, Value>,
        }
        let header: Header = serde_saphyr::from_str(&header)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid skill YAML frontmatter"))?;
        let name = &header.name;
        if name.is_empty()
            || name.len() > 64
            || name.starts_with('-')
            || name.ends_with('-')
            || name.contains("--")
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            || header.description.trim().is_empty()
            || header.description.chars().count() > 1024
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "invalid skill name or description",
            ));
        }
        let mut metadata = header.metadata;
        for (key, value) in header.extra {
            metadata.entry(key).or_insert(value);
        }
        Self::new(
            Definition {
                name: header.name,
                description: Some(header.description),
                metadata,
            },
            lines.collect(),
            executor,
        )
    }

    /// Reads one explicit resource from a caller-selected Workspace. Local and
    /// server resources use the same contract; no hidden workspace lookup.
    pub async fn load<W: Workspace>(
        workspace: &W,
        path: &str,
        executor: F,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        super::operation::check(cancellation)?;
        let data =
            super::operation::cancellable(workspace.read(path, cancellation), cancellation).await?;
        let markdown = std::str::from_utf8(&data)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "skill document must be UTF-8"))?;
        let skill = Self::from_markdown(markdown, executor)?;
        // Standard directory skills must match their parent directory. A
        // single explicitly named resource may also be used independently.
        if path.ends_with("/SKILL.md") {
            let parent = path.rsplit('/').nth(1).unwrap_or("");
            if parent != skill.definition.name {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "skill name must match its directory",
                ));
            }
        }
        super::operation::check(cancellation)?;
        Ok(skill)
    }
}

impl<F: Function<Messages, Messages>> Contract for Skill<F> {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    async fn run(
        &self,
        mut input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error> {
        super::operation::check(cancellation)?;
        let mut instructions = Message::new(Role::System);
        instructions.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":self.instructions}),
        });
        input.0.insert(0, instructions);
        super::operation::cancellable(self.executor.call(input, cancellation), cancellation).await
    }
}
