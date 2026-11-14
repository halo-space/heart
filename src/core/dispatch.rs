//! Internal assembly for the Agent's Tool, MCP, Skill and Workspace views.
//!
//! The concrete adapters are split by responsibility: `dispatch::tool` owns
//! executable Tool/MCP/Workspace tools, while `skill::load` and
//! `skill::reader` own Skill discovery and instruction reading. This module
//! only combines them.

mod call;
mod tool;

use std::collections::BTreeSet;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::Cancellation;
use crate::components::agent::Error;
use crate::components::mcp::Server;
use crate::components::message::Message;
use crate::components::skill::{Definition as SkillDefinition, Loader, Skill};
use crate::components::tool::{Call, Evaluation, Tool, ToolDefinition, validate_call};
use crate::core::skill::{load, reader};
use crate::workspace::Workspace;

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
#[doc(hidden)]
pub struct Dispatch {
    tools: Vec<Arc<dyn tool::Tool>>,
    mcp_catalog: Vec<crate::mcp::Definition>,
    skills: Vec<Arc<dyn load::Skill>>,
    skill_loaders: Vec<Arc<dyn load::Loader>>,
    skill_reader: bool,
    workspace: Option<Arc<dyn Workspace>>,
}

/// Internal inputs accepted by the public Agent/Workspace `.tools(...)`
/// methods. The public tool point is plural
/// even when one concrete Tool is registered; this keeps the assembly API
/// consistent for one or many Tools.
pub trait ToolInputs {
    fn append_to(self, dispatch: &mut Dispatch);
}

impl<T> ToolInputs for T
where
    T: Tool + 'static,
{
    fn append_to(self, dispatch: &mut Dispatch) {
        dispatch.tools.push(Arc::new(tool::adapter::Tool {
            tool: Arc::new(self),
        }));
    }
}

impl<T, const N: usize> ToolInputs for [T; N]
where
    T: Tool + 'static,
{
    fn append_to(self, dispatch: &mut Dispatch) {
        for tool in self {
            dispatch.tools.push(Arc::new(tool::adapter::Tool {
                tool: Arc::new(tool),
            }));
        }
    }
}

impl<T> ToolInputs for Vec<T>
where
    T: Tool + 'static,
{
    fn append_to(self, dispatch: &mut Dispatch) {
        for tool in self {
            dispatch.tools.push(Arc::new(tool::adapter::Tool {
                tool: Arc::new(tool),
            }));
        }
    }
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
            mcp_catalog: Vec::new(),
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
            mcp_catalog: workspace.mcp(),
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
        self.mcp_catalog.extend(other.mcp_catalog);
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
            self.mcp_catalog.push(tool.server.definition().clone());
            self.tools.extend(tool.tools());
        }
        self
    }

    /// Add Skills without executing them. Repeated calls may add different
    /// concrete types. Skill names are unique within the Skill catalog and do
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
            mcp_catalog: base.mcp_catalog.clone(),
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
            mcp_catalog: self.mcp_catalog.clone(),
            skills,
            skill_loaders: self.skill_loaders.clone(),
            skill_reader: self.skill_reader,
            workspace: None,
        })
    }

    /// Add the Skill catalog to the Agent system prompt. Tool and MCP schemas
    /// are passed separately through the model's ordinary `tools` parameter.
    pub(crate) fn system_prompt(&self, prompt: Option<String>) -> Option<String> {
        if self.skills.is_empty() {
            return prompt;
        }
        let mut catalog = String::from(
            "Available Skills are instruction sets. Read a Skill with the `skill` tool when its description is relevant, then follow its instructions using the available Tools and MCP servers. A Skill does not execute those tools.\n<skills>\n",
        );
        for skill in &self.skills {
            let definition = skill.definition();
            catalog.push_str("<skill><name>");
            catalog.push_str(&escape_prompt_text(&definition.name));
            catalog.push_str("</name><description>");
            catalog.push_str(&escape_prompt_text(
                definition.description.as_deref().unwrap_or(""),
            ));
            catalog.push_str("</description></skill>\n");
        }
        catalog.push_str("</skills>");
        Some(match prompt {
            Some(prompt) if !prompt.is_empty() => format!("{prompt}\n\n{catalog}"),
            _ => catalog,
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
            mcp_catalog: discovered.mcp_catalog,
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
    pub(crate) fn tool_catalog(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|tool| tool.definition()).collect()
    }

    /// Prepared Agent tool catalog; not inferred from a generated Plan.
    pub(crate) fn names(&self) -> BTreeSet<String> {
        self.tools
            .iter()
            .map(|tool| tool.definition().name)
            .collect()
    }

    pub(crate) fn skill_catalog(&self) -> Vec<SkillDefinition> {
        self.skills
            .iter()
            .map(|skill| skill.definition().clone())
            .collect()
    }

    pub(crate) fn mcp_catalog(&self) -> Vec<crate::mcp::Definition> {
        self.mcp_catalog.clone()
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
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use futures_executor::block_on;
    use serde_json::{Map, json};

    use super::*;
    use crate::components::message::Role;

    struct Weather {
        definition: ToolDefinition,
        called: Arc<AtomicBool>,
    }

    impl Tool for Weather {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            self.called.store(true, Ordering::SeqCst);
            Ok(Message {
                role: Role::Tool,
                content: Vec::new(),
                values: Map::from_iter([(String::from("temperature"), json!(22))]),
                metadata: Map::new(),
                usage: None,
            })
        }
    }

    fn weather(called: Arc<AtomicBool>) -> Weather {
        Weather {
            definition: ToolDefinition {
                name: "weather".into(),
                description: "weather lookup".into(),
                parameters: serde_json::from_value(json!({
                    "type": "object",
                    "properties": {}
                }))
                .unwrap(),
            },
            called,
        }
    }

    #[test]
    fn dispatch_exports_definitions_and_executes_by_name() {
        let called = Arc::new(AtomicBool::new(false));
        let dispatch = Dispatch::new().tools(weather(called.clone()));
        assert_eq!(dispatch.len(), 1);
        assert_eq!(dispatch.tool_catalog()[0].name, "weather");
        let (message, evaluation) = block_on(dispatch.execute(
            Call {
                call_id: "call_1".into(),
                name: "weather".into(),
                arguments: Map::new(),
            },
            &Cancellation::new(),
        ))
        .unwrap();
        assert!(called.load(Ordering::SeqCst));
        assert_eq!(message.values["temperature"], 22);
        assert!(evaluation.passed);
    }

    #[test]
    fn dispatch_rejects_unknown_tools_without_invoking_registered_tools() {
        let called = Arc::new(AtomicBool::new(false));
        let dispatch = Dispatch::new().tools(weather(called.clone()));
        let error = block_on(dispatch.execute(
            Call {
                call_id: "call_1".into(),
                name: "unknown".into(),
                arguments: Map::new(),
            },
            &Cancellation::new(),
        ))
        .unwrap_err();
        assert_eq!(error.code, "NOT_FOUND");
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn prepared_view_exposes_tools_and_skills_together() {
        let skill = crate::core::skill::Skill::new(
            SkillDefinition {
                name: "research".into(),
                description: Some("Research a topic and cite sources.".into()),
                ..Default::default()
            },
            "Use sources.".into(),
        )
        .unwrap();
        let dispatch = Dispatch::new()
            .tools(weather(Arc::new(AtomicBool::new(false))))
            .skills(vec![skill])
            .unwrap();
        let cancellation = Cancellation::new();
        let prepared = block_on(dispatch.prepared(&cancellation)).unwrap();
        let definitions = prepared.tool_catalog();
        assert_eq!(definitions.len(), 2);
        assert_eq!(definitions[0].name, "weather");
        assert_eq!(definitions[1].name, load::READER_NAME);
        let prompt = prepared.system_prompt(None).unwrap();
        assert!(prompt.contains("research"));
        assert!(prompt.contains("Research a topic and cite sources."));

        let (call, output) = block_on(prepared.call(
            Call {
                call_id: "call_skill".into(),
                name: load::READER_NAME.into(),
                arguments: Map::from_iter([("skill".into(), json!("research"))]),
            },
            &cancellation,
        ))
        .unwrap();
        assert_eq!(call.name, load::READER_NAME);
        assert_eq!(output.values["instructions"], "Use sources.");

        // Preparing an already prepared view remains idempotent.
        let prepared_again = block_on(prepared.prepared(&cancellation)).unwrap();
        assert_eq!(prepared_again.tool_catalog().len(), 2);
    }

    struct RetryingEval {
        definition: ToolDefinition,
        calls: Arc<std::sync::atomic::AtomicUsize>,
        evaluations: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Tool for RetryingEval {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }
        async fn before(&self, call: &mut Call, _cancellation: &Cancellation) -> Result<(), Error> {
            call.arguments.insert("city".into(), json!("Beijing"));
            Ok(())
        }
        async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Message::function(json!({"temperature": 22})))
        }
        async fn eval(
            &self,
            call: &Call,
            message: &Message,
            _cancellation: &Cancellation,
        ) -> Result<Evaluation, Error> {
            assert_eq!(call.arguments["city"], "Beijing");
            assert_eq!(message.values["temperature"], 22);
            if self.evaluations.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(Error::new("TIMEOUT", "evaluation transport failed"))
            } else {
                Ok(Evaluation::failed(crate::Messages::new([
                    Message::function(json!({"feedback": "too warm"})),
                ])))
            }
        }
    }

    #[test]
    fn caller_can_retry_only_eval_using_the_retained_call_and_result() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let evaluations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let dispatch = Dispatch::new().tools(RetryingEval {
            definition: weather(Arc::new(AtomicBool::new(false))).definition,
            calls: calls.clone(),
            evaluations: evaluations.clone(),
        });
        let cancellation = Cancellation::new();
        let (call, message) = block_on(dispatch.call(
            Call {
                call_id: "call_1".into(),
                name: "weather".into(),
                arguments: Map::new(),
            },
            &cancellation,
        ))
        .unwrap();
        assert_eq!(
            block_on(dispatch.evaluate(&call, &message, &cancellation))
                .unwrap_err()
                .code,
            "TIMEOUT"
        );
        let result = block_on(dispatch.evaluate(&call, &message, &cancellation)).unwrap();
        assert!(!result.passed);
        assert!(result.feedback.is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
    }
}
