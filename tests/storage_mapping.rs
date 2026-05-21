//! Integration tests: PUT pagewiki_mapping, GET mapping back, verify
//! dynamic=false rejects unknown fields.

mod storage_common;

use elasticsearch::indices::IndicesGetMappingParts;
use rag::index::pagewiki::PageWiki;
use rag::index::storage::{Base, ElasticStorage, Error, mapping::pagewiki_mapping};
use serde_json::{Value, json};
use storage_common::*;

#[tokio::test]
async fn mapping_round_trip() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        eprintln!("ES unreachable; skipping");
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_mapping");
    create_index(&raw, &index, pagewiki_mapping(1024)).await;

    let resp = raw
        .indices()
        .get_mapping(IndicesGetMappingParts::Index(&[&index]))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let props = &body[&index]["mappings"]["properties"];
    assert_eq!(props["id"]["type"], json!("keyword"));
    assert_eq!(props["content"]["type"], json!("text"));
    assert_eq!(props["content_tokens"]["analyzer"], json!("whitespace"));
    assert_eq!(props["embedding"]["type"], json!("dense_vector"));
    assert_eq!(props["embedding"]["dims"], json!(1024));
    assert_eq!(props["embedding"]["similarity"], json!("cosine"));
    assert_eq!(body[&index]["mappings"]["dynamic"], json!("false"));

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn dynamic_false_ignores_unknown_field() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_mapping_strict");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    // ES `dynamic: "false"` semantics: unknown fields are stored in _source
    // but NOT indexed/searchable. Verify that.
    use elasticsearch::IndexParts;
    let resp = raw
        .index(IndexParts::IndexId(&index, "weird"))
        .body(json!({ "id": "weird", "totally_unknown_field": 42 }))
        .send()
        .await
        .unwrap();
    assert!(resp.status_code().is_success());
    refresh(&raw, &index).await;

    // Confirm the unknown field is not searchable (term query → 0 hits).
    let body = store
        .search(
            &index,
            json!({ "query": { "term": { "totally_unknown_field": 42 } } }),
        )
        .await
        .unwrap();
    assert_eq!(body["hits"]["total"]["value"].as_u64().unwrap(), 0);

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn valid_pagewiki_writes_through_mapping() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_mapping_ok");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    let page = PageWiki {
        id: Some("ok".into()),
        doc_id: Some("d".into()),
        version: Some("v1".into()),
        content: "hi".into(),
        embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
        ..Default::default()
    };
    store.create(&index, page).await.unwrap();
    refresh(&raw, &index).await;

    let got = store.get(&index, "ok").await.unwrap();
    assert_eq!(got.content, "hi");

    // Sanity: get on a non-existent doc within the well-formed index still
    // returns NotFound (not IndexNotFound).
    match store.get(&index, "ghost").await.unwrap_err() {
        Error::NotFound { .. } => {}
        e => panic!("expected NotFound, got {e:?}"),
    }

    delete_index(&raw, &index).await;
}
