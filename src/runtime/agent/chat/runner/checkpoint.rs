//! Chat Agent checkpoint serialization and recovery helpers.

use super::*;

pub(super) fn checkpoint_state(
    messages: &Messages,
    usage: Option<crate::model::token::Usage>,
    scene: (
        Option<&Input>,
        Option<&crate::runtime::agent::preparation::State>,
        Option<&crate::runtime::agent::completion::State>,
    ),
    result: &Result<Event, Error>,
    root: &Root,
    steps: &[Step],
    stream: bool,
) -> Value {
    let (input, preparation, completion) = scene;
    let status = match result {
        Ok(Event::Complete {
            finish_reason: Some(reason),
            ..
        }) if reason == "cancelled" => crate::runtime::agent::status::Status::Cancelled,
        Ok(Event::Complete { .. }) => crate::runtime::agent::status::Status::Succeeded,
        Err(error) if error.code == "CANCELLED" => crate::runtime::agent::status::Status::Cancelled,
        Err(error) if error.code == "WAITING" => crate::runtime::agent::status::Status::Waiting,
        Err(_) => crate::runtime::agent::status::Status::Failed,
        Ok(Event::Delta { .. }) => crate::runtime::agent::status::Status::Running,
    };
    let mut state = json!({
        "chat_id": input.and_then(|input| input.metadata.get("chat_id")).and_then(Value::as_i64),
        "type": "agent",
        "status": status,
        "stream": stream,
        "messages": messages,
        "usage": usage,
        "query": input,
        "metadata": input.map(|input| input.metadata.clone()),
        "execution": root.snapshot(status, usage.clone(), result.as_ref().err().cloned()),
        "steps": steps,
        "waits": [],
    });
    if let Some(preparation) = preparation {
        state["preparation"] =
            serde_json::to_value(preparation).expect("preparation serialization");
    }
    crate::runtime::agent::checkpoint::record_result(&mut state, result);
    if let Some(completion) = completion {
        state["completion"] = serde_json::to_value(completion).expect("completion serialization");
    }
    state
}

pub(super) fn waiting_checkpoint_state(
    messages: &Messages,
    usage: Option<crate::model::token::Usage>,
    scene: (
        Option<&Input>,
        Option<&crate::runtime::agent::preparation::State>,
        Option<&crate::runtime::agent::completion::State>,
    ),
    root: &Root,
    steps: &[Step],
    stream: bool,
) -> Value {
    let mut state = checkpoint_state(
        messages,
        usage.clone(),
        scene,
        &Ok(Event::Delta {
            message: Message::new(Role::Assistant),
        }),
        root,
        steps,
        stream,
    );
    state["status"] = serde_json::to_value(crate::runtime::agent::status::Status::Waiting)
        .expect("serializable waiting status");
    state["execution"] = serde_json::to_value(root.snapshot(
        crate::runtime::agent::status::Status::Waiting,
        usage,
        None,
    ))
    .expect("serializable root execution");
    state
}

pub(super) fn checkpoint_future(
    checkpoint: Option<CheckpointSave>,
    checkpoint_id: Option<i64>,
    state: Value,
) -> CheckpointFuture {
    Box::pin(async move {
        if crate::runtime::agent::scope::capture(state.clone()) {
            Ok(())
        } else if let (Some(checkpoint), Some(id)) = (checkpoint, checkpoint_id) {
            checkpoint.save(id, state, Cancellation::new()).await
        } else {
            Ok(())
        }
    })
}

pub(crate) fn checkpoint_pending_subagents(
    snapshot: &Value,
) -> Result<std::collections::BTreeMap<String, PendingSubAgent>, Error> {
    let root = Root::restore(snapshot)?;
    let mut messages = checkpoint_messages(snapshot)?;
    let steps = checkpoint_steps(snapshot)?;
    Ok(root
        .resume_calls(&steps, &mut messages)?
        .into_iter()
        .map(|(index, call)| (call.call_id.clone(), PendingSubAgent { index, call }))
        .collect())
}

pub(crate) fn checkpoint_messages(snapshot: &Value) -> Result<Messages, Error> {
    serde_json::from_value(snapshot["messages"].clone())
        .map_err(|_| Error::new("INVALID_STATE", "checkpoint has invalid Agent messages"))
}

pub(crate) fn checkpoint_steps(snapshot: &Value) -> Result<Vec<Step>, Error> {
    snapshot
        .get("steps")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| Error::new("INVALID_STATE", "checkpoint has invalid Tool steps"))
        .map(|steps| steps.unwrap_or_default())
}

pub(crate) fn checkpoint_query(snapshot: &Value) -> Result<Input, Error> {
    serde_json::from_value(snapshot["query"].clone()).map_err(|_| {
        Error::new(
            "INVALID_STATE",
            "checkpoint has invalid original Agent input",
        )
    })
}

pub(crate) fn is_waiting_snapshot(snapshot: &Value) -> bool {
    matches!(
        snapshot.get("status").and_then(Value::as_str),
        Some("running" | "waiting" | "paused")
    )
}

pub(crate) fn checkpoint_identity_matches(snapshot: &Value, metadata: &Map<String, Value>) -> bool {
    crate::runtime::agent::checkpoint::identity_matches(snapshot, metadata)
}
