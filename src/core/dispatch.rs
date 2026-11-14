//! Internal assembly for the Agent's Tool, MCP, Skill and Workspace views.
//!
//! The concrete adapters are split by responsibility: `dispatch::tool` owns
//! executable Tool/MCP/Workspace tools, while `skill::load` and
//! `skill::reader` own Skill discovery and instruction reading. This module
//! only combines them.

mod call;
mod inputs;
mod tool;

use std::collections::BTreeSet;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::Cancellation;
use crate::components::agent::Error;
use crate::components::mcp::Server;
use crate::components::message::Message;
use crate::components::skill::{Definition as SkillDefinition, Loader, Skill};
#[cfg(test)]
use crate::components::tool::Tool;
use crate::components::tool::{Call, Evaluation, ToolDefinition, validate_call};
use crate::core::skill::{load, reader};
use crate::workspace::Workspace;
pub(crate) use inputs::ToolInputs;

/// Internal dispatch table for heterogeneous Tools, MCP servers and Skills.
///
/// This type is an implementation detail. It is never the public way to
/// configure an Agent or Workspace. Public callers mount each concrete kind
/// with `.tools(...)`, `.mcps(...)` and `.skills(...)`; this table only keeps
/// their tools together while an Agent runs. One shared `skill`
/// Tool can return a selected Skill's full instructions; the other Tools and
/// MCP servers remain independently defined and executable. Cloning the table
/// shares configured tools and Workspace-owned Skill loader caches; it does not
/// copy or materialize files.
#[derive(Clone)]
pub struct Dispatch {
    tools: Vec<Arc<dyn tool::Tool>>,
    mcp_list: Vec<crate::mcp::Definition>,
    skills: Vec<Arc<dyn load::Skill>>,
    skill_loaders: Vec<Arc<dyn load::Loader>>,
    skill_reader: bool,
    workspace: Option<Arc<dyn Workspace>>,
}

impl Default for Dispatch {
    fn default() -> Self {
        Self::new()
    }
}

impl Dispatch {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            mcp_list: Vec::new(),
            skills: Vec::new(),
            skill_loaders: Vec::new(),
            skill_reader: false,
            workspace: None,
        }
    }

    /// Adapt one concrete Workspace to the internal heterogeneous dispatch
    /// table. The Workspace remains the public execution entrance; this
    /// adapter is only used while assembling an Agent or Harness.
    pub(crate) fn workspace<W>(workspace: W) -> Self
    where
        W: Workspace + 'static,
    {
        let workspace: Arc<dyn Workspace> = Arc::new(workspace);
        let tools = workspace
            .tool()
            .into_iter()
            .map(|definition| {
                Arc::new(tool::WorkspaceTool {
                    workspace: Arc::clone(&workspace),
                    definition,
                }) as Arc<dyn tool::Tool>
            })
            .collect();
        Self {
            tools,
            mcp_list: workspace.mcp(),
            skills: Vec::new(),
            skill_loaders: Vec::new(),
            skill_reader: false,
            workspace: Some(workspace),
        }
    }

    /// Combine two independently assembled tool sets. Local Tool and Skill
    /// names and already-discovered MCP Tool names are checked when the set
    /// is prepared for a run.
    pub fn merge(mut self, other: Self) -> Result<Self, Error> {
        if self.workspace.is_some() && other.workspace.is_some() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "only one Workspace can be attached to an Agent",
            ));
        }
        self = self.append(other);
        self.validate_names(&self.tools)?;
        self.validate_skill_names()?;
        Ok(self)
    }

    pub(crate) fn append(mut self, other: Self) -> Self {
        self.tools.extend(other.tools);
        self.mcp_list.extend(other.mcp_list);
        self.skills.extend(other.skills);
        self.skill_loaders.extend(other.skill_loaders);
        self.skill_reader |= other.skill_reader;
        if self.workspace.is_none() {
            self.workspace = other.workspace;
        }
        self
    }

    /// Add one or more Tools. Repeated calls may add different concrete Tool
    /// types; a single Tool is accepted as well.
    pub fn tools<I>(mut self, tools: I) -> Self
    where
        I: ToolInputs,
    {
        tools.append_to(&mut self);
        self
    }

    /// Register MCP servers that have already connected and discovered their
    /// tools. `Server::http`/`Server::stdio` perform that work before this
    /// method can be called; Agent runs never perform MCP discovery.
    pub fn mcps<S>(mut self, servers: impl IntoIterator<Item = S>) -> Self
    where
        S: Server + 'static,
    {
        for server in servers {
            let tool = tool::McpServer {
                server: Arc::new(server),
            };
            self.mcp_list.push(tool.server.definition().clone());
            self.tools.extend(tool.tools());
        }
        self
    }

    /// Add Skills without executing them. Repeated calls may add different
    /// concrete types. Skill names are unique within the Skill list and do
    /// not share the Tool namespace; `skill` is reserved for the shared reader.
    pub fn skills<S>(mut self, skills: impl IntoIterator<Item = S>) -> Result<Self, Error>
    where
        S: Skill + 'static,
    {
        for skill in skills {
            let definition = skill.definition().clone();
            if definition.name.trim().is_empty() {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "skill name must not be empty",
                ));
            }
            self.skills.push(Arc::new(load::Direct {
                skill: Arc::new(skill),
                definition,
            }));
        }
        self.validate_skill_names()?;
        Ok(self)
    }

    /// Bind a Workspace-owned Skill loader. Discovery and reading happen
    /// asynchronously for each run; the dispatch table never assumes local
    /// file paths.
    pub(crate) fn skill<S: Loader + 'static>(mut self, loader: S) -> Self {
        self.skill_loaders.push(Arc::new(load::adapter::Loader {
            loader: Arc::new(loader),
            cache: Arc::new(RwLock::new(None)),
        }));
        self
    }

    pub(crate) async fn with_discovered_skills(
        &self,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        let base = self.with_workspace_skills(cancellation).await?;
        let mut skills = base.skills.clone();
        for loader in &base.skill_loaders {
            skills.extend(
                crate::cancellation::cancellable(
                    loader.discover(cancellation.clone()),
                    cancellation,
                )
                .await?,
            );
        }
        let view = Self {
            tools: base.tools.clone(),
            mcp_list: base.mcp_list.clone(),
            skills,
            skill_loaders: Vec::new(),
            skill_reader: base.skill_reader,
            workspace: None,
        };
        view.validate_skill_names()?;
        Ok(view)
    }

    async fn with_workspace_skills(&self, cancellation: &Cancellation) -> Result<Self, Error> {
        let Some(workspace) = &self.workspace else {
            return Ok(self.clone());
        };
        let definitions = workspace.skill(cancellation).await?;
        let mut skills = self.skills.clone();
        skills.extend(definitions.into_iter().map(|definition| {
            Arc::new(load::Workspace {
                workspace: Arc::clone(workspace),
                definition,
            }) as Arc<dyn load::Skill>
        }));
        Ok(Self {
            tools: self.tools.clone(),
            mcp_list: self.mcp_list.clone(),
            skills,
            skill_loaders: self.skill_loaders.clone(),
            skill_reader: self.skill_reader,
            workspace: None,
        })
    }

    /// Add the Skill list to the Agent system prompt. Tool and MCP schemas
    /// are passed separately through the model's ordinary `tools` parameter.
    pub(crate) fn system_prompt(&self, prompt: Option<String>) -> Option<String> {
        if self.skills.is_empty() {
            return prompt;
        }
        let mut skill_text = String::from(
            "Available Skills are instruction sets. Read a Skill with the `skill` tool when its description is relevant, then follow its instructions using the available Tools and MCP servers. A Skill does not execute those tools.\n<skills>\n",
        );
        for skill in &self.skills {
            let definition = skill.definition();
            skill_text.push_str("<skill><name>");
            skill_text.push_str(&escape_prompt_text(&definition.name));
            skill_text.push_str("</name><description>");
            skill_text.push_str(&escape_prompt_text(
                definition.description.as_deref().unwrap_or(""),
            ));
            skill_text.push_str("</description></skill>\n");
        }
        skill_text.push_str("</skills>");
        Some(match prompt {
            Some(prompt) if !prompt.is_empty() => format!("{prompt}\n\n{skill_text}"),
            _ => skill_text,
        })
    }

    fn validate_names(&self, tools: &[Arc<dyn tool::Tool>]) -> Result<(), Error> {
        let mut names = BTreeSet::new();
        for name in tools.iter().map(|tool| tool.definition().name) {
            if !names.insert(name) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "Tool names must be unique across configured Tools and MCP servers",
                ));
            }
        }
        if !self.skills.is_empty() && !self.skill_reader && names.contains(load::READER_NAME) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "the skill Tool name is reserved when Skills are configured",
            ));
        }
        Ok(())
    }

    fn validate_skill_names(&self) -> Result<(), Error> {
        let mut names = BTreeSet::new();
        for skill in &self.skills {
            if skill.definition().name.trim().is_empty() {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "Skill name must not be empty",
                ));
            }
            if !names.insert(skill.definition().name.as_str()) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "Skill names must be unique",
                ));
            }
        }
        Ok(())
    }

    pub(crate) async fn prepared(&self, cancellation: &Cancellation) -> Result<Self, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "tool discovery cancelled"));
        }
        let discovered = self
            .with_workspace_skills(cancellation)
            .await?
            .with_discovered_skills(cancellation)
            .await?;
        let mut tools = discovered.tools.clone();
        for tool in &tools {
            let definition = tool.definition();
            definition.validate()?;
        }
        discovered.validate_names(&tools)?;
        let skill_reader = !discovered.skills.is_empty();
        if skill_reader && !discovered.skill_reader {
            tools.push(Arc::new(tool::adapter::Tool {
                tool: Arc::new(reader::Reader::new(discovered.skills.clone())),
            }));
        }
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "tool discovery cancelled"));
        }
        Ok(Self {
            tools,
            mcp_list: discovered.mcp_list,
            skills: discovered.skills,
            skill_loaders: Vec::new(),
            skill_reader,
            workspace: None,
        })
    }

    /// Return currently prepared Tool definitions synchronously. MCP
    /// definitions are available as soon as the server is attached; Skill
    /// sources are still loaded when an Agent obtains its private prepared
    /// view.
    pub(crate) fn tool_list(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|tool| tool.definition()).collect()
    }

    /// Prepared Agent tool list; not inferred from a generated Plan.
    pub(crate) fn names(&self) -> BTreeSet<String> {
        self.tools
            .iter()
            .map(|tool| tool.definition().name)
            .collect()
    }

    pub(crate) fn skill_list(&self) -> Vec<SkillDefinition> {
        self.skills
            .iter()
            .map(|skill| skill.definition().clone())
            .collect()
    }

    pub(crate) fn mcp_list(&self) -> Vec<crate::mcp::Definition> {
        self.mcp_list.clone()
    }
}

fn escape_prompt_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_core_dispatch_tests.rs"]
mod tests;
