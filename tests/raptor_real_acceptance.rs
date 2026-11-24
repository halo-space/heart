//! Real RAPTOR acceptance using a live Chat provider and a live Embed provider.
//!
//! The test does not synthesize vectors or model messages. It verifies the
//! production RAPTOR path with the provider credentials loaded by the real
//! acceptance script.

use halo::agent::{
    Cancellation,
    document::{Document, raptor::Raptor},
    model::{self, chat},
    provider::{deepseek::DeepSeek, zhipu::Zhipu},
};
use serde_json::Value;
use std::collections::HashSet;

fn deepseek_chat() -> impl chat::Chat {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real DeepSeek Chat")
}

fn zhipu_embed() -> impl model::embed::Embed {
    let provider = Zhipu::builder()
        .base_url(
            std::env::var("HEART_EMBED_BASE_URL")
                .unwrap_or_else(|_| "https://open.bigmodel.cn/api/paas/v4".into()),
        )
        .api_key(std::env::var("HEART_EMBED_API_KEY").expect("set HEART_EMBED_API_KEY"))
        .build()
        .expect("construct real Zhipu provider");
    provider
        .embed(model::embed::Config {
            model_name: std::env::var("HEART_EMBED_MODEL").unwrap_or_else(|_| "embedding-3".into()),
            options: Default::default(),
        })
        .expect("construct real Zhipu Embed")
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY, HEART_EMBED_API_KEY and sends live model requests"]
async fn real_raptor_generates_summary_documents_and_relations() {
    let raptor = Raptor::new(
        serde_json::Map::from_iter([
            ("cluster_size".into(), Value::from(2)),
            ("levels".into(), Value::from(1)),
            ("threshold".into(), Value::from(0.0)),
        ]),
        deepseek_chat(),
        zhipu_embed(),
    )
    .expect("construct Raptor with real providers");

    let documents = vec![
        Document {
            id: Some(1001),
            content: "北京春季天气通常温和，白天适合城市步行和短途出行。".into(),
            ..Default::default()
        },
        Document {
            id: Some(1002),
            content: "北京春季出行应关注风力和降雨，户外活动需要准备饮水和防风衣物。".into(),
            ..Default::default()
        },
        Document {
            id: Some(1003),
            content: "北京公共交通覆盖范围较广，地铁适合连接主要景点和商业区域。".into(),
            ..Default::default()
        },
    ];
    let mut next_id = 2000_i64;
    let output = raptor
        .run(&documents, &Cancellation::new(), &mut || {
            next_id += 1;
            next_id
        })
        .await
        .expect("real providers should complete RAPTOR");

    assert!(!output.documents.is_empty(), "RAPTOR returned no summaries");
    assert!(output.documents.iter().all(|document| {
        document.id.is_some_and(|id| id > 2000)
            && !document.content.trim().is_empty()
            && document.metadata["raptor"]["method"] == "raptor"
    }));

    let summary_ids = output
        .documents
        .iter()
        .map(|document| document.id.expect("summary has an id"))
        .collect::<HashSet<_>>();
    assert_eq!(summary_ids.len(), output.documents.len());
    assert!(!output.relations.is_empty(), "RAPTOR returned no relations");
    assert!(output.relations.iter().all(|relation| {
        relation.kind == "summary"
            && documents
                .iter()
                .any(|document| document.id == Some(relation.from))
            && summary_ids.contains(&relation.to)
    }));

    println!(
        "LIVE_RAPTOR_OUTPUT={}",
        serde_json::to_string(&output).expect("serialize real RAPTOR output")
    );
}
