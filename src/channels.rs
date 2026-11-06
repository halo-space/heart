//! External messaging channel adapters.
//!
//! Channels own platform transport and message conversion. The application
//! supplies the handler that binds an inbound message to an Agent, Workflow,
//! or Harness entry point.

pub mod dingtalk;
pub mod feishu;
