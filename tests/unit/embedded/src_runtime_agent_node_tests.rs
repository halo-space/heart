use super::*;
use crate::components::model::token::{InputDetails, OutputDetails};
use serde_json::Map;
use serde_json::json;

#[test]
fn usage_is_recomputable_and_keeps_standard_details_only_in_aggregate() {
    let mut state = State::default();
    state.begin_execution(1).unwrap();
    let first = Usage {
        input: Some(10),
        output: Some(4),
        total: Some(14),
        input_details: Some(InputDetails { cached: Some(3) }),
        output_details: Some(OutputDetails { think: Some(2) }),
        details: Map::from_iter([("provider_private".into(), json!(99))]),
    };
    state
        .set_attempt_result(
            Status::Failed,
            Some(first.clone()),
            Some(Error::new("TIMEOUT", "failed")),
        )
        .unwrap();
    assert!(state.usage.as_ref().unwrap().details.is_empty());
    state.add_attempt().unwrap();
    let second = Usage {
        input: Some(20),
        output: Some(6),
        total: Some(26),
        input_details: Some(InputDetails { cached: Some(5) }),
        output_details: Some(OutputDetails { think: Some(3) }),
        details: Map::from_iter([("provider_private".into(), json!(100))]),
    };
    for _ in 0..2 {
        state
            .set_attempt_result(Status::Succeeded, Some(second.clone()), None)
            .unwrap();
        let usage = state.usage.as_ref().unwrap();
        assert_eq!(usage.total, Some(40));
        assert_eq!(usage.input_details.as_ref().unwrap().cached, Some(8));
        assert_eq!(usage.output_details.as_ref().unwrap().think, Some(5));
        assert!(usage.details.is_empty());
    }
    let execution = state.execution.as_ref().unwrap();
    assert_eq!(execution.attempts[0].usage, Some(first));
    assert_eq!(execution.attempts[1].usage, Some(second));
    assert!(execution.attempts[0].created_time.is_some());
    assert!(execution.attempts[0].finished_time.is_some());
    let missing =
        crate::components::model::token::merge(Some(Usage::default()), Some(Usage::default()))
            .unwrap();
    assert_eq!(missing.input, None);
    assert_eq!(missing.input_details, None);
    assert_eq!(missing.output_details, None);
}
