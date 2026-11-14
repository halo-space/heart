//! Workflow history lifecycle: begin, resume, and close the short-memory record.

use super::*;

impl Engine {
    pub(super) async fn begin_history(
        &mut self,
        initial_messages: &Messages,
        cancellation: &crate::Cancellation,
    ) -> Result<(), Error> {
        let Some(history) = self.memory.clone() else {
            return Ok(());
        };
        if self.history_execute.is_some() {
            return Ok(());
        }
        let input = messages_to_agent_input(initial_messages);
        let run = if self.state.initial.is_some() {
            history
                .resume_for(input, crate::memory::short::Type::Workflow, cancellation)
                .await?
        } else {
            history
                .begin_for(input, crate::memory::short::Type::Workflow, cancellation)
                .await?
        };
        self.history_execute = Some(run);
        Ok(())
    }

    pub(super) async fn close_history(
        &mut self,
        result: &Result<Messages, Error>,
    ) -> Result<(), Error> {
        if self.history_closed {
            return Ok(());
        }
        let Some(run) = self.history_execute.clone() else {
            return Ok(());
        };
        let status = match result {
            Err(error) if error.code == "WAITING" => return Ok(()),
            Ok(_) => crate::memory::short::chat::Status::Completed,
            Err(error) if error.code == "CANCELLED" => crate::memory::short::chat::Status::Stopped,
            Err(_) => crate::memory::short::chat::Status::Failed,
        };
        run.settle(self.state.usage.clone());
        let graph = serde_json::to_value(self.definition())
            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
        let state = self.snapshot()?;
        let message = result.as_ref().ok().and_then(|messages| {
            messages
                .as_slice()
                .iter()
                .rev()
                .find(|message| message.role == Role::Assistant)
                .cloned()
        });
        run.close_workflow(status, message, Some(graph), Some(state))
            .await?;
        self.history_closed = true;
        Ok(())
    }
}
