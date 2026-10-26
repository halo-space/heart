//! Pure plan/evaluation contract tests. Real model behavior is exercised by
//! `component_real_acceptance`; this file deliberately does not fabricate Chat
//! responses or implement a substitute Chat provider.

use futures_executor::block_on;
use halo::agent::evaluation::{Evaluation, Evaluator as _};
use halo::agent::message::Role;
use halo::agent::model::content::part::Part;
use halo::agent::plan::Plan;
use halo::agent::{Cancellation, Error, Message, Messages};
use serde_json::{Value, json};
use std::rc::Rc;

fn plan_value() -> Value {
    json!({
        "version": 2,
        "nodes": {
            "a": {"name": "search", "objective": "查询天气"},
            "b": {"name": "summarize", "objective": "总结天气"}
        },
        "edges": [{"from": "a", "to": "b"}]
    })
}

fn evaluation(passed: bool) -> Evaluation {
    let feedback = (!passed).then(|| {
        let mut message = Message::new(Role::User);
        message.content.push(Part {
            r#type: "text".into(),
            data: serde_json::from_value(json!({"value":"需要补充来源"})).unwrap(),
        });
        Messages::new([message])
    });
    Evaluation { passed, feedback }
}

#[test]
fn plan_json_round_trips_as_the_complete_graph_contract() {
    let source = plan_value();
    let plan: Plan = serde_json::from_value(source.clone()).unwrap();
    plan.validate().unwrap();
    assert_eq!(serde_json::to_value(plan).unwrap(), source);
}

#[test]
fn plan_rejects_missing_or_extra_runtime_fields() {
    let mut missing = plan_value();
    missing["edges"][0].as_object_mut().unwrap().remove("to");
    let mut extra = plan_value();
    extra["nodes"]["a"]["exec_id"] = json!(17);
    for value in [missing, extra] {
        let plan: Result<Plan, _> = serde_json::from_value(value);
        assert!(plan.is_err());
    }
}

#[test]
fn plan_rejects_invalid_dag_shapes() {
    let mut self_edge = plan_value();
    self_edge["edges"] = json!([{"from":"a","to":"a"}]);
    let mut duplicate_edge = plan_value();
    duplicate_edge["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({"from":"a","to":"b"}));
    let mut dangling = plan_value();
    dangling["edges"] = json!([{"from":"a","to":"missing"}]);
    let mut cycle = plan_value();
    cycle["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({"from":"b","to":"a"}));
    let mut no_version = plan_value();
    no_version["version"] = json!(0);

    for value in [self_edge, duplicate_edge, dangling, cycle, no_version] {
        let plan: Plan = serde_json::from_value(value).unwrap();
        assert!(plan.validate().is_err());
    }
}

#[test]
fn evaluation_contract_round_trips_business_feedback_without_result_wrapper() {
    for expected in [evaluation(true), evaluation(false)] {
        let encoded = serde_json::to_value(&expected).unwrap();
        assert_eq!(
            serde_json::from_value::<Evaluation>(encoded).unwrap(),
            expected
        );
    }
}

#[test]
fn user_function_evaluator_supports_native_non_send_futures_and_feedback() {
    let evaluator = |_: halo::agent::Input, _: Messages, _: Cancellation| async move {
        let marker = Rc::new(1);
        std::future::ready(()).await;
        assert_eq!(*marker, 1);
        Ok::<_, Error>(evaluation(false))
    };
    let result = Messages::default();
    assert_eq!(
        block_on(evaluator.evaluate(
            &halo::agent::Input::default(),
            &result,
            &Cancellation::new(),
        ))
        .unwrap(),
        evaluation(false)
    );
}

#[test]
fn user_function_evaluator_preserves_technical_errors() {
    let expected =
        Error::new("UNAVAILABLE", "evaluation backend unavailable").with_details("backend");
    let evaluator = {
        let error = expected.clone();
        move |_: halo::agent::Input, _: Messages, _: Cancellation| {
            let error = error.clone();
            async move { Err::<Evaluation, _>(error) }
        }
    };
    assert_eq!(
        block_on(evaluator.evaluate(
            &halo::agent::Input::default(),
            &Messages::default(),
            &Cancellation::new(),
        ))
        .unwrap_err(),
        expected
    );
}
