//! Adapters for executable Tool, MCP and Workspace capabilities.
//!
//! `dispatch` owns composition; this module owns the concrete call lifecycle
//! boundary for each executable entry.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::Cancellation;
use crate::components::agent::Error;
use crate::components::mcp::Server;
use crate::components::message::Message;
use crate::components::tool::{Call, Evaluation, ToolDefinition, prepare_call, validate_call};
use crate::workspace::Workspace;

pub(crate) type CallFuture =
    Pin<Box<dyn Future<Output = Result<(Call, Message), Error>> + Send + 'static>>;
pub(crate) type PrepareFuture = Pin<Box<dyn Future<Output = Result<Call, Error>> + Send + 'static>>;
pub(crate) type EvaluationFuture =
    Pin<Box<dyn Future<Output = Result<Evaluation, Error>> + Send + 'static>>;

pub(crate) struct McpServer<S> {
    pub(crate) server: Arc<S>,
}

impl<S: Server + 'static> McpServer<S> {
    pub(crate) fn entries(&self) -> Vec<Arc<dyn Tool>> {
        let server = Arc::clone(&self.server);
        server
            .tools()
            .iter()
            .cloned()
            .map(|definition| {
                Arc::new(Mcp {
                    server: Arc::clone(&server),
                    definition,
                }) as Arc<dyn Tool>
            })
            .collect()
    }
}

struct Mcp<S> {
    server: Arc<S>,
    definition: ToolDefinition,
}

impl<S: Server + 'static> Tool for Mcp<S> {
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

pub(crate) trait Tool: Send + Sync {
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

pub(crate) mod adapter {
    use std::sync::Arc;

    pub(crate) struct Tool<T> {
        pub(crate) tool: Arc<T>,
    }
}

impl<T> Tool for adapter::Tool<T>
where
    T: crate::components::tool::Tool + 'static,
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

pub(crate) struct WorkspaceTool {
    pub(crate) workspace: Arc<dyn Workspace>,
    pub(crate) definition: ToolDefinition,
}

impl Tool for WorkspaceTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn prepare(&self, call: Call, _cancellation: Cancellation) -> PrepareFuture {
        Box::pin(async move { Ok(call) })
    }

    fn call(&self, call: Call, cancellation: Cancellation) -> CallFuture {
        let workspace = Arc::clone(&self.workspace);
        Box::pin(async move {
            let (message, evaluation) = workspace.execute(call.clone(), &cancellation).await?;
            if !evaluation.passed {
                return Err(Error::new(
                    "EXECUTION_FAILED",
                    "Workspace Tool evaluation failed",
                ));
            }
            Ok((call, message))
        })
    }

    fn evaluate(
        &self,
        _call: Call,
        _message: Message,
        _cancellation: Cancellation,
    ) -> EvaluationFuture {
        Box::pin(async { Ok(Evaluation::passed()) })
    }
}
