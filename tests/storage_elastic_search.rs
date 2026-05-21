//! Integration tests: ElasticStorage search / multi_search.

mod storage_common;

use rag::index::pagewiki::PageWiki;
use rag::index::storage::{Base, ElasticStorage, mapping::pagewiki_mapping};
use serde_json::json;
use storage_common::*;

fn make_page(id: &str, content: &str, vec: Vec<f32>, tag: &str) -> PageWiki {
    PageWiki {
        id: Some(id.into()),
        doc_id: Some("d".into()),
        version: Some("v1".into()),
        content: content.into(),
        embedding: Some(vec),
        tags: vec![tag.into()],
        ..Default::default()
    }
}

async fn seed(store: &ElasticStorage, raw: &elasticsearch::Elasticsearch, index: &str) {
    let pages = vec![
        make_page("a", "hello world", vec![1.0, 0.0, 0.0, 0.0], "x"),
        make_page("b", "rust language", vec![0.0, 1.0, 0.0, 0.0], "y"),
        make_page("c", "search engine", vec![0.0, 0.0, 1.0, 0.0], "x"),
    ];
    store.bulk_create(index, pages).await.unwrap();
    refresh(raw, index).await;
}

#[tokio::test]
async fn search_match_all_returns_three() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        eprintln!("ES unreachable; skipping");
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_search_all");
    create_index(&raw, &index, pagewiki_mapping(4)).await;
    seed(&store, &raw, &index).await;

    let body = store
        .search(&index, json!({ "query": { "match_all": {} } }))
        .await
        .unwrap();
    let total = body["hits"]["total"]["value"].as_u64().unwrap();
    assert_eq!(total, 3);

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn search_term_filter() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_search_term");
    create_index(&raw, &index, pagewiki_mapping(4)).await;
    seed(&store, &raw, &index).await;

    let body = store
        .search(&index, json!({ "query": { "term": { "tags": "x" } } }))
        .await
        .unwrap();
    let total = body["hits"]["total"]["value"].as_u64().unwrap();
    assert_eq!(total, 2);

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn search_knn_top1() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_search_knn");
    create_index(&raw, &index, pagewiki_mapping(4)).await;
    seed(&store, &raw, &index).await;

    let body = store
        .search(
            &index,
            json!({
                "size": 1,
                "knn": {
                    "field": "embedding",
                    "query_vector": [0.0, 1.0, 0.0, 0.0],
                    "k": 1,
                    "num_candidates": 10
                }
            }),
        )
        .await
        .unwrap();
    let hits = body["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["_id"], "b");

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn multi_search_two_queries_preserve_order() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_msearch");
    create_index(&raw, &index, pagewiki_mapping(4)).await;
    seed(&store, &raw, &index).await;

    let responses = store
        .multi_search(
            &index,
            vec![
                json!({ "query": { "term": { "tags": "x" } } }),
                json!({ "query": { "term": { "tags": "y" } } }),
            ],
        )
        .await
        .unwrap();
    assert_eq!(responses.len(), 2);
    assert_eq!(responses[0]["hits"]["total"]["value"].as_u64().unwrap(), 2);
    assert_eq!(responses[1]["hits"]["total"]["value"].as_u64().unwrap(), 1);

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn multi_search_empty_returns_empty_without_request() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let out = store.multi_search("__noop__", vec![]).await.unwrap();
    assert!(out.is_empty());
}
