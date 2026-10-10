pub mod chat;
pub mod content;
pub mod embed;
pub mod token;

/// Model capabilities share one error fact shape. Runtime control fields do
/// not enter this error type.
pub type Error = crate::components::error::Error;
