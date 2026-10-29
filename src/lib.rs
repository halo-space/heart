//! Halo Agent/Workflow execution kernel.
//!
//! This crate starts from the frozen public contracts. Runtime scheduling is
//! intentionally added after these contracts compile and are covered by tests.

extern crate self as halo;

pub mod agent;
pub(crate) mod cancellation;
mod components;
pub(crate) mod core;
pub(crate) mod pkg;
pub(crate) mod runtime;

pub(crate) use components::checkpoint;
pub(crate) use components::compression;
pub(crate) mod document {
    pub use crate::components::document::*;
}
pub use components::error::Error;
pub(crate) use components::evaluation;
pub(crate) use components::event;
pub mod harness;
pub(crate) use components::mcp;
pub(crate) use components::memory;
pub(crate) use components::message::{self, Message, Messages};
pub(crate) mod middleware {
    pub use crate::components::middleware::*;
}
pub(crate) use components::model;
pub(crate) mod node {
    pub use crate::components::node::*;
    #[allow(unused_imports)]
    pub use crate::core::node::Echo;
}
pub use cancellation::Cancellation;
pub(crate) use components::permission;
pub(crate) use components::plan;
pub(crate) use components::reason;
pub(crate) use components::sandbox;
pub(crate) use components::skill;
pub(crate) mod tool {
    pub use crate::components::tool::*;
    #[allow(unused_imports)]
    pub use crate::core::tool::execute;
}
pub(crate) use components::workspace;
/// Optional compile-time check that a function or component method can be
/// registered as a Workflow Node. Ordinary functions work without the
/// attribute. Components may use `call = method` or `signal = method` on an
/// inherent impl, without a wrapping Node object.
///
/// ```
/// use halo::{node, agent::{Cancellation, Error, workflow::WorkflowBuilder}};
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
/// use halo::{node, agent::{Cancellation, Error}};
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
