use crate::runtime::agent::lifecycle::model_unfinished;
use crate::runtime::agent::{
    model_steps,
    node::{Status, Step},
};
use crate::{
    Error, Message,
    message::Role,
    model::{chat::Request, content::part::Part},
};
use serde_json::{Value, json};

fn active_model_steps() -> Vec<Step> {
    let mut user = Message::new(Role::User);
    user.values
        .insert("text".into(), json!("explain the observed output"));
    let mut steps = Vec::new();
    model_steps::input(
        &mut steps,
        Request {
            messages: crate::Messages::new([user]),
            stream: true,
            ..Default::default()
        },
    );
    steps[0].stage = Some("call".into());
    model_steps::calling(&mut steps);
    steps
}

fn observed_partial() -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: serde_json::from_value(json!({"value":"received so far"})).unwrap(),
    });
    message
}

#[test]
fn unfinished_output_is_retained_only_on_the_actual_attempt() {
    let mut steps = active_model_steps();
    let partial = observed_partial();

    let usage = model_unfinished(&mut steps, &partial).expect("partial output is billable");
    model_steps::stopped(
        &mut steps,
        &Error::new("CANCELLED", "request was cancelled"),
    );

    let step = &steps[0];
    let attempt = &step.attempts[0];
    assert_eq!(step.status, "cancelled");
    assert!(
        step.message.is_none(),
        "partial data is not a successful result"
    );
    assert_eq!(attempt.status, Status::Cancelled);
    assert_eq!(attempt.usage, Some(usage.clone()));
    assert_eq!(attempt.message.as_deref().unwrap().content, partial.content);
    assert_eq!(attempt.message.as_deref().unwrap().usage, Some(usage));
    assert!(model_steps::validate(&steps, 0).is_ok());
}

#[test]
fn provider_usage_is_retained_and_runtime_metadata_does_not_change_estimates() {
    let mut plain_steps = active_model_steps();
    let mut padded_steps = active_model_steps();
    let plain = observed_partial();
    let mut padded = plain.clone();
    padded
        .metadata
        .insert("noise".into(), json!("m".repeat(20_000)));
    padded
        .values
        .insert("noise".into(), json!("v".repeat(20_000)));

    let estimated = model_unfinished(&mut plain_steps, &plain).unwrap();
    let padded_estimate = model_unfinished(&mut padded_steps, &padded).unwrap();
    assert_eq!(estimated, padded_estimate);
    assert!(estimated.input.unwrap() > 0);
    assert!(estimated.output.unwrap() > 0);

    let mut provider_steps = active_model_steps();
    let mut provider_partial = observed_partial();
    provider_partial.usage = Some(crate::model::token::Usage {
        input: Some(31),
        output: Some(12),
        total: Some(43),
        ..Default::default()
    });
    let provider_usage = model_unfinished(&mut provider_steps, &provider_partial).unwrap();
    assert_eq!(provider_usage.input, Some(31));
    assert_eq!(provider_usage.output, Some(12));
    assert_eq!(provider_usage.total, Some(43));
    assert_eq!(
        provider_steps[0].attempts[0]
            .message
            .as_ref()
            .unwrap()
            .usage,
        Some(provider_usage)
    );
}

#[test]
fn failed_attempt_keeps_partial_error_and_rejects_inconsistent_saved_facts() {
    let mut steps = active_model_steps();
    let partial = observed_partial();
    model_unfinished(&mut steps, &partial).unwrap();
    let error = Error::new("NETWORK", "provider stream ended unexpectedly");
    model_steps::stopped(&mut steps, &error);

    assert_eq!(steps[0].attempts[0].status, Status::Failed);
    assert_eq!(steps[0].attempts[0].error, Some(error));
    assert_eq!(
        steps[0].attempts[0].message.as_ref().unwrap().content,
        partial.content
    );
    assert!(model_steps::validate(&steps, 0).is_ok());

    let mut corrupt = steps;
    corrupt[0].attempts[0].message.as_mut().unwrap().usage = None;
    let scene: Value = serde_json::to_value(&corrupt).unwrap();
    let decoded: Vec<Step> = serde_json::from_value(scene).unwrap();
    let error = model_steps::validate(&decoded, 0).unwrap_err();
    assert_eq!(error.code, "INVALID_STATE");
}

#[test]
fn absent_partial_output_does_not_create_a_synthetic_attempt_message() {
    let mut steps = active_model_steps();
    assert!(model_unfinished(&mut steps, &Message::new(Role::Assistant)).is_none());
    assert!(steps[0].attempts[0].message.is_none());
    assert!(steps[0].attempts[0].usage.is_none());
    assert!(model_steps::validate(&steps, 0).is_ok());
}
