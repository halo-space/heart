//! Live OpenAI Embeddings API acceptance; no canned vector responses.

use halo::agent::{
    model::embed::{self, Embed as _},
    provider::openai::OpenAI,
};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn openai_text_embedding_returns_real_vector_dimension_and_usage() {
    let provider = OpenAI::builder()
        .api_key(std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY"))
        .build()
        .expect("construct OpenAI Provider");
    let embed = provider
        .embed(embed::Config {
            model_name: "text-embedding-3-small".into(),
            options: Default::default(),
        })
        .expect("construct OpenAI Embed");
    let message = embed
        .embed(embed::Request {
            input: "A real OpenAI embedding acceptance request".into(),
            dimensions: Some(512),
            options: Default::default(),
        })
        .await
        .expect("request a real OpenAI embedding");

    let vector = message
        .values
        .get("vector")
        .and_then(serde_json::Value::as_array)
        .expect("Embed Message contains values.vector");
    assert_eq!(vector.len(), 512);
    assert!(vector.iter().all(serde_json::Value::is_number));
    assert_eq!(message.metadata["provider_name"], "openai");
    assert_eq!(message.metadata["model_name"], "text-embedding-3-small");
    assert_eq!(message.metadata["dimension"], 512);
    assert!(
        message
            .usage
            .as_ref()
            .and_then(|usage| usage.input)
            .is_some_and(|tokens| tokens > 0)
    );
    println!(
        "LIVE_OPENAI_EMBED_DIMENSION={} USAGE={:?}",
        vector.len(),
        message.usage
    );
}
