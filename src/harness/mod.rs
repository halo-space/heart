//! Reserved namespace for a future high-level Agent assembly, in the style of DeepAgents.
//!
//! Harness is not a general lifecycle layer shared by Agent and Workflow.
//! Agent and Workflow own separate execution capabilities and reuse Core
//! implementations directly. This namespace is reserved for higher-level
//! Agent assemblies; shared execution and lifecycle mechanisms live with the
//! Runtime or concrete Agent implementation that owns their semantics.
//!
//! The high-level Harness assembly itself is not implemented yet.
