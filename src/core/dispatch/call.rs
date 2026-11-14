//! Tool lifecycle execution for the internal Dispatch table.
//!
//! Tool assembly and Skill discovery remain in the parent module; this
//! module only advances a selected tool through validation and lifecycle
//! phases.

use super::*;

impl Dispatch {
    pub async fn execute(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<(Message, Evaluation), Error> {
        let prepared = self.prepared(cancellation).await?;
        let (call, message) = prepared.call(call, cancellation).await?;
        let evaluation = prepared.evaluate(&call, &message, cancellation).await?;
        Ok((message, evaluation))
    }

    pub(crate) async fn call(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<(Call, Message), Error> {
        self.advance_call(crate::core::lifecycle::Phase::Before, call, cancellation)
            .await
    }

    pub(crate) async fn prepare_call(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<Call, Error> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        crate::cancellation::cancellable(tool.prepare(call, cancellation.clone()), cancellation)
            .await
    }

    async fn advance_call(
        &self,
        phase: crate::core::lifecycle::Phase,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<(Call, Message), Error> {
        use crate::core::lifecycle::{Methods, Phase, advance};
        let (call, message) = advance(
            phase,
            Phase::Complete,
            call,
            None,
            Methods {
                before: |call| self.prepare_call(call, cancellation),
                call: |call| async {
                    self.invoke(call, cancellation)
                        .await
                        .map(|(_, message)| message)
                },
                after: |call, message| async {
                    self.finish(call, message, cancellation)
                        .await
                        .map(|(_, message)| message)
                },
                // Standalone tool calls have no Runtime owner.
                // They do not persist recovery facts or attempt stage retries.
                commit: |_: Phase, _: &Call, _: Option<&Message>| async { Ok(()) },
            },
            cancellation,
        )
        .await?;
        Ok((call, message.expect("completed Tool call has a result")))
    }

    // Native Agent Steps retain each successful boundary before entering the
    // next user hook. The public convenience lifecycle still runs all stages.
    pub(crate) async fn invoke(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> Result<(Call, Message), Error> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        crate::cancellation::cancellable(tool.call(call, cancellation.clone()), cancellation).await
    }

    pub(crate) async fn finish(
        &self,
        call: Call,
        message: Message,
        cancellation: &Cancellation,
    ) -> Result<(Call, Message), Error> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        crate::cancellation::cancellable(
            tool.after(call, message, cancellation.clone()),
            cancellation,
        )
        .await
    }

    /// Preflight before optional Agent permission checks; no Tool/MCP hooks.
    pub(crate) fn validate(&self, call: &Call) -> Result<(), Error> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        validate_call(&tool.definition(), call)
    }

    pub(crate) async fn evaluate(
        &self,
        call: &Call,
        message: &Message,
        cancellation: &Cancellation,
    ) -> Result<Evaluation, Error> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.definition().name == call.name)
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered"))?;
        crate::cancellation::cancellable(
            tool.evaluate(call.clone(), message.clone(), cancellation.clone()),
            cancellation,
        )
        .await
    }

    /// Count currently prepared Tools, including the shared Skill reader.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.tools.len()
    }
}
