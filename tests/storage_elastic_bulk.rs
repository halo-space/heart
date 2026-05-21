//! Integration tests: ElasticStorage bulk operations.

mod storage_common;

use rag::index::pagewiki::PageWiki;
use rag::index::storage::{Base, BulkAction, ElasticStorage, Error, mapping::pagewiki_mapping};
use storage_common::*;

fn make_page(id: &str) -> PageWiki {
    PageWiki {
        id: Some(id.into()),
        doc_id: Some("doc_x".into()),
        version: Some("v1".into()),
        content: format!("c-{id}"),
        embedding: Some(vec![0.0, 0.1, 0.2, 0.3]),
        ..Default::default()
    }
}

#[tokio::test]
async fn bulk_create_update_delete_round_trip() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        eprintln!("ES unreachable; skipping");
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_bulk");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    let pages: Vec<PageWiki> = (0..3).map(|i| make_page(&format!("b{i}"))).collect();
    store.bulk_create(&index, pages.clone()).await.unwrap();
    refresh(&raw, &index).await;

    for p in &pages {
        let got = store.get(&index, p.id.as_ref().unwrap()).await.unwrap();
        assert_eq!(got.content, p.content);
    }

    // bulk_update: change content on the same 3 ids
    let updates: Vec<(String, PageWiki)> = pages
        .iter()
        .map(|p| {
            let mut np = p.clone();
            np.content = format!("u-{}", p.id.as_ref().unwrap());
            (p.id.clone().unwrap(), np)
        })
        .collect();
    store.bulk_update(&index, updates).await.unwrap();
    refresh(&raw, &index).await;

    for p in &pages {
        let got = store.get(&index, p.id.as_ref().unwrap()).await.unwrap();
        assert!(got.content.starts_with("u-"));
    }

    // bulk_delete
    let ids: Vec<String> = pages.iter().map(|p| p.id.clone().unwrap()).collect();
    store.bulk_delete(&index, ids).await.unwrap();
    refresh(&raw, &index).await;

    for p in &pages {
        match store.get(&index, p.id.as_ref().unwrap()).await {
            Err(Error::NotFound { .. }) => {}
            other => panic!("expected NotFound after bulk_delete, got {other:?}"),
        }
    }

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn bulk_create_partial_failure_on_duplicate_id() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_bulk_dup");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    let p = make_page("dup");
    store.bulk_create(&index, vec![p.clone()]).await.unwrap();
    refresh(&raw, &index).await;

    // Second bulk_create with same id + a new id → first conflicts, second ok.
    let err = store
        .bulk_create(&index, vec![p, make_page("fresh")])
        .await
        .unwrap_err();
    match err {
        Error::BulkPartialFailure { failures } => {
            assert_eq!(failures.len(), 1);
            assert_eq!(failures[0].action, BulkAction::Create);
            assert_eq!(failures[0].id, "dup");
            assert_eq!(failures[0].status, 409);
        }
        e => panic!("expected BulkPartialFailure, got {e:?}"),
    }

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn bulk_empty_inputs_return_ok_without_request() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    // No index needed; empty vecs short-circuit before any HTTP call.
    store.bulk_create("__noop__", vec![]).await.unwrap();
    store.bulk_update("__noop__", vec![]).await.unwrap();
    store.bulk_delete("__noop__", vec![]).await.unwrap();
}
