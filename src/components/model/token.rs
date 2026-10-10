use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, JsonSchema)]
pub struct InputDetails {
    pub cached: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, JsonSchema)]
pub struct OutputDetails {
    pub think: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, JsonSchema)]
pub struct Usage {
    pub input: Option<u64>,
    pub output: Option<u64>,
    pub total: Option<u64>,
    pub input_details: Option<InputDetails>,
    pub output_details: Option<OutputDetails>,
    pub details: Map<String, Value>,
}

/// Internal aggregation of settled calls; provider extensions stay on each
/// original Message/Attempt, not on an aggregate with ambiguous semantics.
pub(crate) fn merge(left: Option<Usage>, right: Option<Usage>) -> Option<Usage> {
    fn add(left: Option<u64>, right: Option<u64>) -> Option<u64> {
        match (left, right) {
            (Some(a), Some(b)) => a.checked_add(b),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }
    match (left, right) {
        (Some(a), Some(b)) => Some(Usage {
            input: add(a.input, b.input),
            output: add(a.output, b.output),
            total: add(a.total, b.total),
            input_details: (a.input_details.is_some() || b.input_details.is_some()).then(|| {
                InputDetails {
                    cached: add(
                        a.input_details.and_then(|v| v.cached),
                        b.input_details.and_then(|v| v.cached),
                    ),
                }
            }),
            output_details: (a.output_details.is_some() || b.output_details.is_some()).then(|| {
                OutputDetails {
                    think: add(
                        a.output_details.and_then(|v| v.think),
                        b.output_details.and_then(|v| v.think),
                    ),
                }
            }),
            details: Map::new(),
        }),
        (a, b) => a.or(b).map(|mut usage| {
            usage.details.clear();
            usage
        }),
    }
}
