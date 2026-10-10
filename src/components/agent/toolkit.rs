use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::components::agent::Error;
use crate::components::mcp::Server;
use crate::components::message::{Message, Messages};
use crate::components::skill::{Definition as SkillDefinition, Skill};
use crate::components::tool::{
    Call, Evaluation, Tool, ToolDefinition, call_prepared, prepare_call, validate_call,
};
use crate::runtime::cancellation::Cancellation;

type CallFuture = Pin<Box<dyn Future<Output = Result<(Call, Message), Error>>>>;
type PrepareFuture = Pin<Box<dyn Future<Output = Result<Call, Error>>>>;
type EvaluationFuture = Pin<Box<dyn Future<Output = Result<Evaluation, Error>>>>;
type DiscoveryFuture = Pin<Box<dyn Future<Output = Result<Vec<Arc<dyn Entry>>, Error>>>>;
type SkillFuture = Pin<Box<dyn Future<Output = Result<Messages, Error>>>>;

trait SkillEntry: Send + Sync {
    fn definition(&self) -> &SkillDefinition;
    fn run(&self, input: Messages, cancellation: Cancellation) -> SkillFuture;
}

struct TypedSkill<S> {
    skill: Arc<S>,
    definition: SkillDefinition,
}

impl<S: Skill + 'static> SkillEntry for TypedSkill<S> {
    fn definition(&self) -> &SkillDefinition {
        &self.definition
    }

    fn run(&self, input: Messages, cancellation: Cancellation) -> SkillFuture {
        let skill = Arc::clone(&self.skill);
        Box::pin(async move { skill.run(input, &cancellation).await })
    }
}

trait Mcp: Send + Sync {
    fn discover(&self, cancellation: Cancellation) -> DiscoveryFuture;
}

struct TypedMcp<S> {
    server: Arc<S>,
}

impl<S: Server + 'static> Mcp for TypedMcp<S> {
    fn discover(&self, cancellation: Cancellation) -> DiscoveryFuture {
        let server = Arc::clone(&self.server);
        Box::pin(async move {
            let definitions = server.list_tools(&cancellation).await?;
            Ok(definitions
                .into_iter()
                .map(|definition| {
                    Arc::new(McpTool {
                        server: Arc::clone(&server),
                        definition,
                    }) as Arc<dyn Entry>
                })
                .collect())
        })
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
        Box::pin(async move { call_prepared(tool.as_ref(), call, &cancellation).await })
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

/// Agent assembly-level collection of heterogeneous Tools, MCP servers and Skills.
///
/// A Toolkit exposes Tool definitions and delegates explicitly selected calls. It does
/// not become a Workflow Node and does not own Runtime execution state.
pub struct Toolkit {
    entries: Vec<Arc<dyn Entry>>,
    mcps: Vec<Arc<dyn Mcp>>,
    skills: Vec<Arc<dyn SkillEntry>>,
}

impl Default for Toolkit {
    fn default() -> Self {
        Self::new()
    }
}

impl Toolkit {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            mcps: Vec::new(),
            skills: Vec::new(),
        }
    }

    /// Add one Tool. Repeated calls may add different concrete Tool types.
    pub fn tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.entries.push(Arc::new(TypedEntry {
            tool: Arc::new(tool),
        }));
        self
    }

    /// Add several Tools of one concrete type.
    pub fn tools<T>(mut self, tools: impl IntoIterator<Item = T>) -> Self
    where
        T: Tool + 'static,
    {
        for tool in tools {
            self = self.tool(tool);
        }
        self
    }

    /// Register already initialized MCP servers; no connection or discovery
    /// occurs during fluent configuration. Different server types can be
    /// supplied through repeated calls. Discovery happens once per Agent run
    /// or standalone execute, not on each ReAct model iteration.
    pub fn mcps<S>(mut self, servers: impl IntoIterator<Item = S>) -> Self
    where
        S: Server + 'static,
    {
        for server in servers {
            self.mcps.push(Arc::new(TypedMcp {
                server: Arc::new(server),
            }));
        }
        self
    }

    /// Add Skills without executing them. Repeated calls may add different
    /// concrete types. Skill names and local Tool names must be unique;
    /// collisions with discovered MCP Tools are checked during preparation.
    /// Skills retain their own Messages -> Messages contract, not a Tool schema.
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
        self.validate_names(&self.entries)?;
        Ok(self)
    }

    /// Run one explicitly selected Skill. Preserve its complete Messages and
    /// original Error; do not discover unrelated MCP servers or retry it.
    /// This does not enable automatic model selection of Skills.
    pub async fn run(
        &self,
        name: &str,
        input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "skill call cancelled"));
        }
        self.validate_names(&self.entries)?;
        let skill = self
            .skills
            .iter()
            .find(|skill| skill.definition().name == name)
            .ok_or_else(|| Error::new("NOT_FOUND", "skill is not registered"))?;
        await_stage(skill.run(input, cancellation.clone()), cancellation).await
    }

    fn validate_names(&self, entries: &[Arc<dyn Entry>]) -> Result<(), Error> {
        let mut names = BTreeSet::new();
        for name in entries.iter().map(|entry| entry.definition().name).chain(
            self.skills
                .iter()
                .map(|skill| skill.definition().name.clone()),
        ) {
            if !names.insert(name) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "names must be unique across Toolkit capabilities",
                ));
            }
        }
        Ok(())
    }

    pub(crate) async fn prepared(&self, cancellation: &Cancellation) -> Result<Self, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "tool discovery cancelled"));
        }
        let mut entries = self.entries.clone();
        for server in &self.mcps {
            entries.extend(await_stage(server.discover(cancellation.clone()), cancellation).await?);
        }
        for entry in &entries {
            let definition = entry.definition();
            definition.validate()?;
        }
        self.validate_names(&entries)?;
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "tool discovery cancelled"));
        }
        Ok(Self {
            entries,
            mcps: Vec::new(),
            skills: self.skills.clone(),
        })
    }

    /// Return locally registered Tool definitions synchronously. MCP discovery
    /// is asynchronous: Agent run obtains a private prepared view including
    /// discovered definitions, without mutating the reusable configuration.
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
            .chain(
                self.skills
                    .iter()
                    .map(|skill| skill.definition().name.clone()),
            )
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
        let call = self.prepare_call(call, cancellation).await?;
        self.call_prepared(call, cancellation).await
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

    pub(crate) async fn call_prepared(
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

    /// Whether the local/prepared Tool view is empty. Skills are not Tools.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Count local/prepared Tools, not Skills or undiscovered MCP Tools.
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
                "Toolkit operation cancelled",
            )));
        }
        match future.as_mut().poll(context) {
            std::task::Poll::Ready(_) if cancellation.is_cancelled() => {
                std::task::Poll::Ready(Err(Error::new("CANCELLED", "Toolkit operation cancelled")))
            }
            result => result,
        }
    })
    .await
}

/// Construct an empty Agent Toolkit.
pub fn new() -> Toolkit {
    Toolkit::new()
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
    fn toolkit_exports_definitions_and_executes_by_name() {
        let called = Arc::new(AtomicBool::new(false));
        let toolkit = new().tool(weather(called.clone()));
        assert_eq!(toolkit.len(), 1);
        assert_eq!(toolkit.definitions()[0].name, "weather");
        let (message, evaluation) = block_on(toolkit.execute(
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
    fn toolkit_rejects_unknown_tools_without_invoking_registered_tools() {
        let called = Arc::new(AtomicBool::new(false));
        let toolkit = new().tool(weather(called.clone()));
        let error = block_on(toolkit.execute(
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
    fn prepared_view_retains_skills_without_executing_them() {
        let called = Arc::new(AtomicBool::new(false));
        let flag = called.clone();
        let skill = crate::core::skill::Skill::new(
            SkillDefinition {
                name: "research".into(),
                ..Default::default()
            },
            "Use sources.".into(),
            move |input: Messages, _: Cancellation| {
                flag.store(true, Ordering::SeqCst);
                std::future::ready(Ok::<_, Error>(input))
            },
        )
        .unwrap();
        let toolkit = new().skills(vec![skill]).unwrap();
        let cancellation = Cancellation::new();
        let prepared = block_on(toolkit.prepared(&cancellation)).unwrap();
        assert!(!called.load(Ordering::SeqCst));
        assert!(prepared.definitions().is_empty());
        let output =
            block_on(prepared.run("research", Messages::default(), &cancellation)).unwrap();
        assert_eq!(output.0[0].role, Role::System);
        assert!(called.load(Ordering::SeqCst));
        // Reusable assembly is not consumed by preparation.
        assert!(block_on(toolkit.run("research", Messages::default(), &cancellation)).is_ok());
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
        let toolkit = new().tool(RetryingEval {
            definition: weather(Arc::new(AtomicBool::new(false))).definition,
            calls: calls.clone(),
            evaluations: evaluations.clone(),
        });
        let cancellation = Cancellation::new();
        let (call, message) = block_on(toolkit.call(
            Call {
                call_id: "call_1".into(),
                name: "weather".into(),
                arguments: Map::new(),
            },
            &cancellation,
        ))
        .unwrap();
        assert_eq!(
            block_on(toolkit.evaluate(&call, &message, &cancellation))
                .unwrap_err()
                .code,
            "TIMEOUT"
        );
        let result = block_on(toolkit.evaluate(&call, &message, &cancellation)).unwrap();
        assert!(!result.passed);
        assert!(result.feedback.is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
    }
}
