//! Shared helpers for storage integration tests.
//!
//! Uses `RAG_TEST_ES_URL` (default `http://localhost:9200`). When ES is
//! unreachable, [`skip_if_no_es`] returns true and tests early-return so
//! environments without ES still pass.

#![allow(dead_code)]

use elasticsearch::{
    Elasticsearch,
    http::transport::{SingleNodeConnectionPool, TransportBuilder},
    indices::{IndicesCreateParts, IndicesDeleteParts, IndicesRefreshParts},
};
use rag::index::storage::{Base, ElasticStorage, Error};
use serde_json::{Value, json};

pub fn es_url() -> String {
    std::env::var("RAG_TEST_ES_URL").unwrap_or_else(|_| "http://localhost:9200".into())
}

pub fn raw_client(url: &str) -> Elasticsearch {
    let parsed = url::Url::parse(url).expect("parse url");
    let pool = SingleNodeConnectionPool::new(parsed);
    let transport = TransportBuilder::new(pool)
        .build()
        .expect("build transport");
    Elasticsearch::new(transport)
}

pub async fn skip_if_no_es(store: &ElasticStorage) -> bool {
    match store
        .search(
            "__rag_probe_nonexistent__",
            json!({ "query": { "match_all": {} } }),
        )
        .await
    {
        Err(Error::Transport(_)) => true,
        // Any reachable ES will respond with index_not_found_exception or
        // a 404 Es body, which is fine — the cluster is up.
        _ => false,
    }
}

pub fn unique_index(prefix: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{prefix}_{nanos}")
}

pub async fn create_index(client: &Elasticsearch, index: &str, mapping: Value) {
    let resp = client
        .indices()
        .create(IndicesCreateParts::Index(index))
        .body(mapping)
        .send()
        .await
        .expect("create index");
    let status = resp.status_code().as_u16();
    let body = resp.text().await.unwrap_or_default();
    assert!(
        (200..300).contains(&status),
        "create index {index} failed: {status} {body}"
    );
}

pub async fn delete_index(client: &Elasticsearch, index: &str) {
    let _ = client
        .indices()
        .delete(IndicesDeleteParts::Index(&[index]))
        .send()
        .await;
}

pub async fn refresh(client: &Elasticsearch, index: &str) {
    let _ = client
        .indices()
        .refresh(IndicesRefreshParts::Index(&[index]))
        .send()
        .await;
}
