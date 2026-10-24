//! Run with a local QA CSV path; no Runtime, database or model is required.
use futures_executor::block_on;
use halo::agent::Cancellation;
use halo::agent::document::{self, Loader, Parser, Transformer};
use serde_json::json;

fn main() -> Result<(), document::Error> {
    block_on(async {
        let source = document::Source {
            uri: std::env::args().nth(1).ok_or_else(|| {
                document::Error::new("INVALID_ARGUMENTS", "provide a local QA CSV path")
            })?,
        };
        let cancellation = Cancellation::new();
        let data = document::LocalLoader.load(&source, &cancellation).await?;
        let parser = document::parser::new(
            "qa",
            json!({"format":"csv", "engine":"csv", "options":{"question":"question", "answer":"answer"}}),
        )?;
        let mut documents = parser.parse(&data, &cancellation).await?;
        for document in &mut documents {
            // An application can also supply its actual doc_id here.
            document
                .metadata
                .insert("source".into(), json!({"uri": source.uri}));
        }
        let documents = document::transformer::new()
            .remove_empty()
            .deduplicate()
            .transform(documents, &cancellation)
            .await?;
        let output = serde_json::to_string_pretty(&documents).map_err(|_| {
            document::Error::new("INVALID_RESPONSE", "Document cannot be serialized")
        })?;
        println!("{output}");
        Ok(())
    })
}
