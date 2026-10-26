//! Semantic chunking acceptance using actual embedding Provider responses.

use halo::agent::{
    Cancellation,
    document::{self, Chunker as _, Document},
    model::embed::Config as EmbedConfig,
    provider::zhipu::{Config as ZhipuConfig, Zhipu},
};
use serde_json::json;
use std::collections::HashSet;

#[tokio::test]
#[ignore = "requires HEART_EMBED_API_KEY and optional HEART_EMBED_BASE_URL/HEART_EMBED_MODEL"]
async fn semantic_chunker_preserves_real_source_content_and_metadata() {
    let provider = Zhipu::new(ZhipuConfig {
        base_url: std::env::var("HEART_EMBED_BASE_URL")
            .unwrap_or_else(|_| "https://open.bigmodel.cn/api/paas/v4".into()),
        api_key: std::env::var("HEART_EMBED_API_KEY").expect("set HEART_EMBED_API_KEY"),
        ..ZhipuConfig::default()
    })
    .expect("construct real Zhipu Embed Provider");
    let embed = provider
        .embed(EmbedConfig {
            model_name: std::env::var("HEART_EMBED_MODEL").unwrap_or_else(|_| "embedding-3".into()),
            options: Default::default(),
        })
        .expect("construct real Embed component");
    let source = "露营前应查看天气和风力。强风可能影响帐篷安全。徒步时应携带足够饮水。";
    let chunker = document::chunker::semantic::new(
        json!({"max_tokens":12,"min_tokens":1,"threshold":0.72,"max_concurrency":2}),
        embed,
    )
    .expect("valid Semantic Chunker options");
    let mut next_id = 100_i64;
    let chunks = chunker
        .chunk(
            vec![Document {
                doc_id: Some(55),
                content: source.into(),
                metadata: serde_json::Map::from_iter([(
                    "source".into(),
                    json!({"uri":"live://camping-guide"}),
                )]),
                ..Default::default()
            }],
            &Cancellation::new(),
            &mut || {
                next_id += 1;
                next_id
            },
        )
        .await
        .expect("real Zhipu embeddings should split source text");

    assert!(!chunks.is_empty());
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<String>(),
        source
    );
    assert!(chunks.iter().all(|chunk| chunk.doc_id == Some(55)));
    assert!(chunks.iter().all(|chunk| {
        chunk.metadata.get("source") == Some(&json!({"uri":"live://camping-guide"}))
    }));
    let ids = chunks
        .iter()
        .map(|chunk| chunk.id.expect("each delivered chunk receives an ID"))
        .collect::<HashSet<_>>();
    assert_eq!(ids.len(), chunks.len());
    println!(
        "LIVE_SEMANTIC_CHUNKS={}",
        serde_json::to_string(&chunks).unwrap()
    );
}
