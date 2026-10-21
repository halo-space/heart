//! Internal mounted Tool/MCP/Skill dispatch used by the concrete Agent
//! implementations. This module is deliberately hidden from the public
//! component contract; callers use the Agent and Workspace `.tools(...)`,
//! `.mcps(...)` and `.skills(...)` methods.

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::Cancellation;
use crate::components::agent::Error;
use crate::components::mcp::Server;
use crate::components::message::Message;
use crate::components::skill::{Definition as SkillDefinition, Loader, Skill};
use crate::components::tool::{
    Call, Evaluation, Tool, ToolDefinition, prepare_call, validate_call,
};

type CallFuture = Pin<Box<dyn Future<Output = Result<(Call, Message), Error>> + Send + 'static>>;
type PrepareFuture = Pin<Box<dyn Future<Output = Result<Call, Error>> + Send + 'static>>;
type EvaluationFuture = Pin<Box<dyn Future<Output = Result<Evaluation, Error>> + Send + 'static>>;
type SkillFuture = Pin<Box<dyn Future<Output = Result<String, Error>> + Send + 'static>>;
type ResourceFuture = Pin<Box<dyn Future<Output = Result<Vec<u8>, Error>> + Send + 'static>>;
type SkillDiscoveryFuture =
    Pin<Box<dyn Future<Output = Result<Vec<Arc<dyn SkillEntry>>, Error>> + Send + 'static>>;

trait SkillEntry: Send + Sync {
    fn definition(&self) -> &SkillDefinition;
    fn read(&self, cancellation: Cancellation) -> SkillFuture;
    fn read_resource(&self, _path: String, _cancellation: Cancellation) -> ResourceFuture {
        Box::pin(async {
            Err(Error::new(
                "NOT_FOUND",
                "Skill has no Workspace resource source",
            ))
        })
    }
}

trait SkillLoader: Send + Sync {
    fn discover(&self, cancellation: Cancellation) -> SkillDiscoveryFuture;
}

struct TypedLoader<S> {
    loader: Arc<S>,
}

impl<S: Loader + 'static> SkillLoader for TypedLoader<S> {
    fn discover(&self, cancellation: Cancellation) -> SkillDiscoveryFuture {
        let loader = Arc::clone(&self.loader);
        Box::pin(async move {
            let definitions = loader.list(&cancellation).await?;
            Ok(definitions
                .into_iter()
                .map(|definition| {
                    Arc::new(SourcedSkill {
                        loader: Arc::clone(&loader),
                        definition,
                    }) as Arc<dyn SkillEntry>
                })
                .collect())
        })
    }
}

struct SourcedSkill<S> {
    loader: Arc<S>,
    definition: SkillDefinition,
}

impl<S: Loader + 'static> SkillEntry for SourcedSkill<S> {
    fn definition(&self) -> &SkillDefinition {
        &self.definition
    }

    fn read(&self, cancellation: Cancellation) -> SkillFuture {
        let loader = Arc::clone(&self.loader);
        let name = self.definition.name.clone();
        Box::pin(async move { loader.read(&name, &cancellation).await })
    }

    fn read_resource(&self, path: String, cancellation: Cancellation) -> ResourceFuture {
        let loader = Arc::clone(&self.loader);
        let name = self.definition.name.clone();
        Box::pin(async move { loader.read_resource(&name, &path, &cancellation).await })
    }
}

struct TypedSkill<S> {
    skill: Arc<S>,
    definition: SkillDefinition,
}

impl<S: Skill + 'static> SkillEntry for TypedSkill<S> {
    fn definition(&self) -> &SkillDefinition {
        &self.definition
    }

    fn read(&self, cancellation: Cancellation) -> SkillFuture {
        let skill = Arc::clone(&self.skill);
        Box::pin(async move { skill.read(&cancellation).await })
    }
}

const SKILL_TOOL_NAME: &str = "skill";

struct SkillReader {
    skills: Vec<Arc<dyn SkillEntry>>,
    definition: ToolDefinition,
}

impl SkillReader {
    fn new(skills: Vec<Arc<dyn SkillEntry>>) -> Self {
        Self {
            skills,
            definition: ToolDefinition {
                name: SKILL_TOOL_NAME.into(),
                description: "Read a Skill's full instructions by exact name, or a text resource relative to that Skill with `resource`. Skills explain how to use available capabilities; reading does not execute them.".into(),
                parameters: serde_json::from_value(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "skill": {
                            "type": "string",
                            "description": "The exact name of the registered Skill to read."
                        },
                        "resource": {
                            "type": "string",
                            "description": "Optional path to a text resource inside that Skill."
                        }
                    },
                    "required": ["skill"],
                    "additionalProperties": false
                }))
                .expect("static Skill parameters are an object"),
            },
        }
    }
}

impl Tool for SkillReader {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let name = call
            .arguments
            .get("skill")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "skill name is required"))?;
        let skill = self
            .skills
            .iter()
            .find(|skill| skill.definition().name == name)
            .ok_or_else(|| Error::new("NOT_FOUND", "skill is not registered"))?;
        if let Some(path) = call
            .arguments
            .get("resource")
            .and_then(serde_json::Value::as_str)
        {
            if path.is_empty()
                || path.contains(['\\', '\0'])
                || path
                    .split('/')
                    .any(|part| part.is_empty() || part == "." || part == "..")
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "Skill resource requires a normalized relative path",
                ));
            }
            let data = await_stage(
                skill.read_resource(path.to_owned(), cancellation.clone()),
                cancellation,
            )
            .await?;
            let content = String::from_utf8(data).map_err(|_| {
                Error::new("INVALID_ARGUMENTS", "Skill resource must be UTF-8 text")
            })?;
            Ok(Message::function(serde_json::json!({
                "name": name, "resource": path, "content": content
            })))
        } else {
            let instructions = await_stage(skill.read(cancellation.clone()), cancellation).await?;
            Ok(Message::function(serde_json::json!({
                "name": name, "instructions": instructions
            })))
        }
    }
}

struct TypedMcp<S> {
    server: Arc<S>,
}

impl<S: Server + 'static> TypedMcp<S> {
    fn entries(&self) -> Vec<Arc<dyn Entry>> {
        let server = Arc::clone(&self.server);
        server
            .tools()
            .iter()
            .cloned()
            .map(|definition| {
                Arc::new(McpTool {
                    server: Arc::clone(&server),
                    definition,
                }) as Arc<dyn Entry>
            })
            .collect()
    }
}

struct McpTool<S> {
    server: Arc<S>,
    definition: ToolDefinition,
}

impl<S: Server + 'static> Entry for McpTool<S> {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn call(&self, call: Call, cancellation: Cancellation) -> CallFuture {
        let server = Arc::clone(&self.server);
        let definition = self.definition.clone();
        Box::pin(async move {
            validate_call(&definition, &call)?;
            let message = server.call(call.clone(), &cancellation).await?;
            Ok((call, message))
        })
    }

    fn prepare(&self, call: Call, _cancellation: Cancellation) -> PrepareFuture {
        // No local before hook or argument rewrite for an MCP tool. call
        // retains its existing validation immediately before server dispatch.
        Box::pin(async move { Ok(call) })
    }

    fn evaluate(
        &self,
        _call: Call,
        _message: Message,
        _cancellation: Cancellation,
    ) -> EvaluationFuture {
        // MCP has no local Tool hooks: preserve its direct Message/Error.
        Box::pin(async { Ok(Evaluation::passed()) })
    }
}

trait Entry: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn prepare(&self, call: Call, cancellation: Cancellation) -> PrepareFuture;
    fn call(&self, call: Call, cancellation: Cancellation) -> CallFuture;
    fn after(&self, call: Call, message: Message, _cancellation: Cancellation) -> CallFuture {
        Box::pin(async move { Ok((call, message)) })
    }
    fn evaluate(
        &self,
        call: Call,
        message: Message,
        cancellation: Cancellation,
    ) -> EvaluationFuture;
}

struct TypedEntry<T> {
    tool: Arc<T>,
}

impl<T> Entry for TypedEntry<T>
where
    T: Tool + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.tool.definition().clone()
    }

    fn call(&self, call: Call, cancellation: Cancellation) -> CallFuture {
        let tool = Arc::clone(&self.tool);
        Box::pin(async move {
            let message = tool.call(&call, &cancellation).await?;
            Ok((call, message))
        })
    }

    fn after(&self, call: Call, mut message: Message, cancellation: Cancellation) -> CallFuture {
        let tool = Arc::clone(&self.tool);
        Box::pin(async move {
            tool.after(&call, &mut message, &cancellation).await?;
            Ok((call, message))
        })
    }

    fn prepare(&self, call: Call, cancellation: Cancellation) -> PrepareFuture {
        let tool = Arc::clone(&self.tool);
        Box::pin(async move { prepare_call(tool.as_ref(), call, &cancellation).await })
    }

    fn evaluate(
        &self,
        call: Call,
        message: Message,
        cancellation: Cancellation,
    ) -> EvaluationFuture {
        let tool = Arc::clone(&self.tool);
        Box::pin(async move { tool.eval(&call, &message, &cancellation).await })
    }
}

/// Internal Agent mount table for heterogeneous Tools, MCP servers and Skills.
///
/// This type is an implementation detail. It is never the public way to
/// configure an Agent or Workspace. Public callers mount each concrete kind
/// with `.tools(...)`, `.mcps(...)` and `.skills(...)`; this table only keeps
/// their dispatch entries together while an Agent runs. One shared `skill`
/// Tool can return a selected Skill's full instructions; the other Tools and
/// MCP servers remain independently defined and executable. Cloning the table
/// shares mounted entries and Workspace-owned Skill loader caches; it does not
/// copy or materialize files.
#[derive(Clone)]
#[doc(hidden)]
pub struct Capabilities {
    entries: Vec<Arc<dyn Entry>>,
    skills: Vec<Arc<dyn SkillEntry>>,
    skill_loaders: Vec<Arc<dyn SkillLoader>>,
    skill_reader: bool,
}

/// Internal inputs accepted by the public Agent/Workspace `.tools(...)`
/// methods. The public entry point is plural
/// even when one concrete Tool is mounted; this keeps the assembly API
/// consistent for one or many Tools.
#[doc(hidden)]
pub trait ToolInputs {
    fn append_to(self, capabilities: &mut Capabilities);
}

impl<T> ToolInputs for T
where
    T: Tool + 'static,
{
    fn append_to(self, capabilities: &mut Capabilities) {
        capabilities.entries.push(Arc::new(TypedEntry {
            tool: Arc::new(self),
        }));
    }
}

impl<T, const N: usize> ToolInputs for [T; N]
where
    T: Tool + 'static,
{
    fn append_to(self, capabilities: &mut Capabilities) {
        for tool in self {
            capabilities.entries.push(Arc::new(TypedEntry {
                tool: Arc::new(tool),
            }));
        }
    }
}

impl<T> ToolInputs for Vec<T>
where
    T: Tool + 'static,
{
    fn append_to(self, capabilities: &mut Capabilities) {
        for tool in self {
            capabilities.entries.push(Arc::new(TypedEntry {
                tool: Arc::new(tool),
            }));
        }
    }
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::new()
    }
}

impl Capabilities {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            skills: Vec::new(),
            skill_loaders: Vec::new(),
            skill_reader: false,
        }
    }

    /// Combine two independently assembled capability sets. Local Tool and
    /// Skill names and already-discovered MCP Tool names are checked when the
    /// The capability set is prepared for a run.
    pub fn merge(mut self, other: Self) -> Result<Self, Error> {
        self.entries.extend(other.entries);
        self.skills.extend(other.skills);
        self.skill_loaders.extend(other.skill_loaders);
        self.skill_reader |= other.skill_reader;
        self.validate_names(&self.entries)?;
        self.validate_skill_names()?;
        Ok(self)
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
            let mounted = TypedMcp {
                server: Arc::new(server),
            };
            self.entries.extend(mounted.entries());
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
            self.skills.push(Arc::new(TypedSkill {
                skill: Arc::new(skill),
                definition,
            }));
        }
        self.validate_skill_names()?;
        Ok(self)
    }

    /// Bind a Workspace-owned Skill loader. Discovery and reading happen
    /// asynchronously for each run; the capability set never assumes local
    /// file paths.
    pub(crate) fn skill<S: Loader + 'static>(mut self, loader: S) -> Self {
        self.skill_loaders.push(Arc::new(TypedLoader {
            loader: Arc::new(loader),
        }));
        self
    }

    pub(crate) fn retain_skill_loaders(mut self, other: &Self) -> Self {
        self.skill_loaders
            .extend(other.skill_loaders.iter().cloned());
        self
    }

    pub(crate) async fn with_discovered_skills(
        &self,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        let mut skills = self.skills.clone();
        for loader in &self.skill_loaders {
            skills.extend(await_stage(loader.discover(cancellation.clone()), cancellation).await?);
        }
        let view = Self {
            entries: self.entries.clone(),
            skills,
            skill_loaders: Vec::new(),
            skill_reader: self.skill_reader,
        };
        view.validate_skill_names()?;
        Ok(view)
    }

    /// Add the Skill catalog to the Agent system prompt. Tool and MCP schemas
    /// are passed separately through the model's ordinary `tools` parameter.
    pub(crate) fn system_prompt(&self, prompt: Option<String>) -> Option<String> {
        if self.skills.is_empty() {
            return prompt;
        }
        let mut catalog = String::from(
            "Available Skills are instruction sets. Read a Skill with the `skill` tool when its description is relevant, then follow its instructions using the available Tools and MCP capabilities. A Skill does not execute those capabilities.\n<skills>\n",
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

    fn validate_names(&self, entries: &[Arc<dyn Entry>]) -> Result<(), Error> {
        let mut names = BTreeSet::new();
        for name in entries.iter().map(|entry| entry.definition().name) {
            if !names.insert(name) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "Tool names must be unique across mounted Tools and MCP servers",
                ));
            }
        }
        if !self.skills.is_empty() && !self.skill_reader && names.contains(SKILL_TOOL_NAME) {
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
        let discovered = self.with_discovered_skills(cancellation).await?;
        let mut entries = discovered.entries.clone();
        for entry in &entries {
            let definition = entry.definition();
            definition.validate()?;
        }
        discovered.validate_names(&entries)?;
        let skill_reader = !discovered.skills.is_empty();
        if skill_reader && !discovered.skill_reader {
            entries.push(Arc::new(TypedEntry {
                tool: Arc::new(SkillReader::new(discovered.skills.clone())),
            }));
        }
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "tool discovery cancelled"));
        }
        Ok(Self {
            entries,
            skills: discovered.skills,
            skill_loaders: Vec::new(),
            skill_reader,
        })
    }

    /// Return currently prepared Tool definitions synchronously. MCP
    /// definitions are available as soon as the server is attached; Skill
    /// sources are still loaded when an Agent obtains its private prepared
    /// view.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.entries
            .iter()
            .map(|entry| entry.definition())
            .collect()
    }

    /// Prepared Agent capability catalog; not inferred from a generated Plan.
    pub(crate) fn capability_names(&self) -> BTreeSet<String> {
        self.entries
            .iter()
            .map(|entry| entry.definition().name)
            .collect()
    }

    pub async fn execute(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<(Message, Evaluation), Error> {
        let prepared = self.prepared(cancellation).await?;
        let (call, message) = prepared.call(call, cancellation).await?;
        let evaluation = prepared.evaluate(&call, &message, cancellation).await?;
        Ok((message, evaluation))
    }

    pub(crate) async fn call(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<(Call, Message), Error> {
        self.advance_call(crate::core::lifecycle::Phase::Before, call, cancellation)
            .await
    }

    pub(crate) async fn prepare_call(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<Call, Error> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        await_stage(entry.prepare(call, cancellation.clone()), cancellation).await
    }

    async fn advance_call(
        &self,
        phase: crate::core::lifecycle::Phase,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<(Call, Message), Error> {
        use crate::core::lifecycle::{Methods, Phase, advance};
        let (call, message) = advance(
            phase,
            Phase::Complete,
            call,
            None,
            Methods {
                before: |call| self.prepare_call(call, cancellation),
                call: |call| async {
                    self.invoke(call, cancellation)
                        .await
                        .map(|(_, message)| message)
                },
                after: |call, message| async {
                    self.finish(call, message, cancellation)
                        .await
                        .map(|(_, message)| message)
                },
                // Standalone capability calls have no Runtime owner.
                // They do not persist recovery facts or attempt stage retries.
                commit: |_: Phase, _: &Call, _: Option<&Message>| async { Ok(()) },
            },
            cancellation,
        )
        .await?;
        Ok((call, message.expect("completed Tool call has a result")))
    }

    // Native Agent Steps retain each successful boundary before entering the
    // next user hook. The public convenience lifecycle still runs all stages.
    pub(crate) async fn invoke(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<(Call, Message), Error> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        await_stage(entry.call(call, cancellation.clone()), cancellation).await
    }

    pub(crate) async fn finish(
        &self,
        call: Call,
        message: Message,
        cancellation: &Cancellation,
    ) -> Result<(Call, Message), Error> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        await_stage(
            entry.after(call, message, cancellation.clone()),
            cancellation,
        )
        .await
    }

    /// Preflight before optional Agent permission checks; no Tool/MCP hooks.
    pub(crate) fn validate(&self, call: &Call) -> Result<(), Error> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        validate_call(&entry.definition(), call)
    }

    pub(crate) async fn evaluate(
        &self,
        call: &Call,
        message: &Message,
        cancellation: &Cancellation,
    ) -> Result<Evaluation, Error> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        await_stage(
            entry.evaluate(call.clone(), message.clone(), cancellation.clone()),
            cancellation,
        )
        .await
    }

    /// Whether the local/prepared Tool view is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Count currently prepared Tools, including the shared Skill reader.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

async fn await_stage<T>(
    future: impl Future<Output = Result<T, Error>>,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|context| {
        cancellation.register(context.waker());
        if cancellation.is_cancelled() {
            return std::task::Poll::Ready(Err(Error::new(
                "CANCELLED",
                "Capability operation cancelled",
            )));
        }
        match future.as_mut().poll(context) {
            std::task::Poll::Ready(_) if cancellation.is_cancelled() => std::task::Poll::Ready(
                Err(Error::new("CANCELLED", "Capability operation cancelled")),
            ),
            result => result,
        }
    })
    .await
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
    fn mounted_capabilities_exports_definitions_and_executes_by_name() {
        let called = Arc::new(AtomicBool::new(false));
        let capabilities = Capabilities::new().tools(weather(called.clone()));
        assert_eq!(capabilities.len(), 1);
        assert_eq!(capabilities.definitions()[0].name, "weather");
        let (message, evaluation) = block_on(capabilities.execute(
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
    fn mounted_capabilities_rejects_unknown_tools_without_invoking_registered_tools() {
        let called = Arc::new(AtomicBool::new(false));
        let capabilities = Capabilities::new().tools(weather(called.clone()));
        let error = block_on(capabilities.execute(
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
        let capabilities = Capabilities::new()
            .tools(weather(Arc::new(AtomicBool::new(false))))
            .skills(vec![skill])
            .unwrap();
        let cancellation = Cancellation::new();
        let prepared = block_on(capabilities.prepared(&cancellation)).unwrap();
        let definitions = prepared.definitions();
        assert_eq!(definitions.len(), 2);
        assert_eq!(definitions[0].name, "weather");
        assert_eq!(definitions[1].name, SKILL_TOOL_NAME);
        let prompt = prepared.system_prompt(None).unwrap();
        assert!(prompt.contains("research"));
        assert!(prompt.contains("Research a topic and cite sources."));

        let (call, output) = block_on(prepared.call(
            Call {
                call_id: "call_skill".into(),
                name: SKILL_TOOL_NAME.into(),
                arguments: Map::from_iter([("skill".into(), json!("research"))]),
            },
            &cancellation,
        ))
        .unwrap();
        assert_eq!(call.name, SKILL_TOOL_NAME);
        assert_eq!(output.values["instructions"], "Use sources.");

        // Preparing an already prepared view remains idempotent.
        let prepared_again = block_on(prepared.prepared(&cancellation)).unwrap();
        assert_eq!(prepared_again.definitions().len(), 2);
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
        let capabilities = Capabilities::new().tools(RetryingEval {
            definition: weather(Arc::new(AtomicBool::new(false))).definition,
            calls: calls.clone(),
            evaluations: evaluations.clone(),
        });
        let cancellation = Cancellation::new();
        let (call, message) = block_on(capabilities.call(
            Call {
                call_id: "call_1".into(),
                name: "weather".into(),
                arguments: Map::new(),
            },
            &cancellation,
        ))
        .unwrap();
        assert_eq!(
            block_on(capabilities.evaluate(&call, &message, &cancellation))
                .unwrap_err()
                .code,
            "TIMEOUT"
        );
        let result = block_on(capabilities.evaluate(&call, &message, &cancellation)).unwrap();
        assert!(!result.passed);
        assert!(result.feedback.is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
    }
}
