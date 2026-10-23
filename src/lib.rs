//! Halo Agent/Workflow execution kernel.
//!
//! This crate starts from the frozen public contracts. Runtime scheduling is
//! intentionally added after these contracts compile and are covered by tests.

extern crate self as halo;

pub mod agent;
pub mod cancellation;
mod components;
pub mod core;
pub(crate) mod pkg;
pub mod runtime;

pub use components::checkpoint;
pub use components::compression;
/// Document contracts and the agreed concrete construction paths. Contract
/// definitions remain under components; implementations remain under core.
pub mod document {
    pub use crate::components::document::*;
    pub use crate::core::document::{LocalLoader, chunker, graph, parser, raptor, transformer};
}
pub use components::error::Error;
pub use components::evaluation;
pub use components::event;
pub use components::function;
pub mod harness;
pub use components::knowledge;
pub use components::mcp;
pub use components::memory;
pub use components::message::{self, Message, Messages};
pub mod middleware {
    pub use crate::components::middleware::*;
    pub use crate::core::middleware::Retry;
}
pub use components::model;
pub mod node {
    pub use crate::components::node::*;
    pub use crate::core::node::Echo;
}
pub use cancellation::Cancellation;
pub use components::permission;
pub use components::plan;
pub use components::prompt;
pub use components::reason;
pub use components::sandbox;
pub use components::skill;
pub mod tool {
    pub use crate::components::tool::*;
    pub use crate::core::tool::execute;
}
pub use components::workspace;
pub use core::provider;
/// Optional compile-time check that a function or component method can be
/// registered as a Workflow Node. Ordinary functions work without the
/// attribute. Components may use `call = method` or `signal = method` on an
/// inherent impl, without a wrapping Node object.
///
/// ```
/// use halo::agent::{Cancellation, Error, node, workflow::WorkflowBuilder};
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Deserialize, JsonSchema)]
/// struct Input { value: i64 }
/// #[derive(Serialize, JsonSchema)]
/// struct Output { value: i64 }
///
/// #[node]
/// async fn calculate(input: Input, _: Cancellation) -> Result<Output, Error> {
///     Ok(Output { value: input.value + 1 })
/// }
///
/// let mut builder = WorkflowBuilder::new("example");
/// builder.add_node("calculate", calculate)?;
/// # Ok::<(), halo::agent::workflow::GraphError>(())
/// ```
///
/// ```compile_fail
/// use halo::agent::{Cancellation, Error, node};
/// #[node]
/// fn synchronous(_: i64, _: Cancellation) -> Result<i64, Error> { Ok(1) }
/// ```
pub use heart_node_macros::node;

// Compile the attribute in the defining library too, not only downstream
// examples. This is a test-only assertion, with no node or runtime created.
#[cfg(test)]
const _: () = {
    #[crate::node]
    async fn macro_internal_contract(
        input: crate::Messages,
        _: crate::Cancellation,
    ) -> Result<crate::Messages, crate::Error> {
        Ok(input)
    }
};
