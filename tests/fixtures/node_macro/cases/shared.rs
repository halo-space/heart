use heart::{Cancellation, Error};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

#[derive(Deserialize, JsonSchema)]
struct Input { value: i64 }
#[derive(Serialize, JsonSchema)]
struct Output { value: i64 }
