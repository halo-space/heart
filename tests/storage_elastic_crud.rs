//! Integration tests: ElasticStorage CRUD against a local ES instance.
//!
//! Set `RAG_TEST_ES_URL` (default `http://localhost:9200`).

mod storage_common;

use rag::index::pagewiki::PageWiki;
use rag::index::storage::{Base, ElasticStorage, Error, mapping::pagewiki_mapping};
use storage_common::*;

fn make_page(id: &str, content: &str) -> PageWiki {
    PageWiki {
        id: Some(id.into()),
        doc_id: Some("doc_x".into()),
        version: Some("v1".into()),
        content: content.into(),
        embedding: Some(vec![0.0, 0.1, 0.2, 0.3]),
        ..Default::default()
    }
}

#[tokio::test]
async fn crud_round_trip() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        eprintln!("ES unreachable; skipping crud_round_trip");
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_crud");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    let page = make_page("p1", "hello");
    store.create(&index, page.clone()).await.unwrap();
    refresh(&raw, &index).await;

    let got = store.get(&index, "p1").await.unwrap();
    assert_eq!(got.id.as_deref(), Some("p1"));
    assert_eq!(got.content, "hello");

    let mut updated = page.clone();
    updated.content = "world".into();
    store.update(&index, "p1", updated).await.unwrap();
    refresh(&raw, &index).await;

    let got2 = store.get(&index, "p1").await.unwrap();
    assert_eq!(got2.content, "world");

    store.delete(&index, "p1").await.unwrap();

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn create_conflict_when_id_exists() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_conflict");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    let page = make_page("dup", "a");
    store.create(&index, page.clone()).await.unwrap();
    let err = store.create(&index, page).await.unwrap_err();
    assert!(matches!(err, Error::Conflict { .. }), "got {err:?}");

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn get_missing_returns_not_found() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_nf");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    match store.get(&index, "nope").await.unwrap_err() {
        Error::NotFound { .. } => {}
        e => panic!("expected NotFound, got {e:?}"),
    }

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn get_against_missing_index_returns_index_not_found() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let index = unique_index("rag_it_no_index");
    match store.get(&index, "x").await.unwrap_err() {
        Error::IndexNotFound(_) => {}
        e => panic!("expected IndexNotFound, got {e:?}"),
    }
}

#[tokio::test]
async fn update_missing_returns_not_found() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_upd_nf");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    let page = make_page("ghost", "x");
    match store.update(&index, "ghost", page).await.unwrap_err() {
        Error::NotFound { .. } => {}
        e => panic!("expected NotFound, got {e:?}"),
    }

    delete_index(&raw, &index).await;
}

#[tokio::test]
async fn delete_missing_returns_not_found() {
    let url = es_url();
    let store = ElasticStorage::from_url(&url).unwrap();
    if skip_if_no_es(&store).await {
        return;
    }
    let raw = raw_client(&url);
    let index = unique_index("rag_it_del_nf");
    create_index(&raw, &index, pagewiki_mapping(4)).await;

    match store.delete(&index, "ghost").await.unwrap_err() {
        Error::NotFound { .. } => {}
        e => panic!("expected NotFound, got {e:?}"),
    }

    delete_index(&raw, &index).await;
}
