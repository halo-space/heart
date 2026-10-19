use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    Cancellation, Error,
    skill::{Definition, Skill as Contract},
};

/// A Skill's instructions and resource references. Loading or reading it does
/// not execute scripts, fetch references, or grant tool/permission access.
pub struct Skill {
    definition: Definition,
    instructions: String,
}

impl Skill {
    pub fn new(definition: Definition, instructions: String) -> Result<Self, Error> {
        if definition.name.trim().is_empty() || instructions.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "skill name and instructions are required",
            ));
        }
        Ok(Self {
            definition,
            instructions,
        })
    }

    /// Parses standard SKILL.md YAML frontmatter followed by Markdown. The
    /// body remains readable Markdown; metadata does not grant permissions.
    pub fn from_markdown(markdown: &str) -> Result<Self, Error> {
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
        )
    }

    /// Load one file from an explicit Local workspace. Database-backed skills
    /// can use `new` or `from_markdown` with the data retrieved by their backend;
    /// the common Workspace contract does not prescribe a file loading API.
    pub async fn load(
        workspace: &super::workspace::Local,
        path: &str,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        super::operation::check(cancellation)?;
        let data =
            super::operation::cancellable(workspace.read(path, cancellation), cancellation).await?;
        let markdown = std::str::from_utf8(&data)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "skill document must be UTF-8"))?;
        let skill = Self::from_markdown(markdown)?;
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

impl Contract for Skill {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    async fn read(&self, cancellation: &Cancellation) -> Result<String, Error> {
        super::operation::check(cancellation)?;
        let instructions = self.instructions.clone();
        super::operation::check(cancellation)?;
        Ok(instructions)
    }
}
