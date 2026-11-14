use super::*;
use futures_util::stream::{self, Empty};

struct EmptyAgent;

impl Agent for EmptyAgent {
    type Stream = Empty<Result<Event, Error>>;

    async fn run(
        &self,
        _input: Input,
        _stream: bool,
        _cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        Ok(stream::empty())
    }

    async fn reply(&self, _input: Input, _cancellation: Cancellation) -> Result<(), Error> {
        Ok(())
    }
}

#[test]
fn delegation_definition_limits_model_targets_to_registered_members() {
    let subagents = vec![
        SubAgent::new("researcher", EmptyAgent)
            .unwrap()
            .description("researches facts"),
        SubAgent::new("reviewer", EmptyAgent)
            .unwrap()
            .description("reviews results"),
        SubAgent::new("coder", EmptyAgent)
            .unwrap()
            .description("writes code"),
    ];
    let definition = SubAgent::definition_named(&subagents, "agent_spawn", "Delegate");
    let names = &definition.parameters["properties"]["name"]["enum"];
    assert_eq!(
        names,
        &serde_json::json!(["researcher", "reviewer", "coder"])
    );
    definition.validate().unwrap();
}
