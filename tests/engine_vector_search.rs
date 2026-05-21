//! engine.vector_search e2e + NoVectorBackend 兜底。

mod common;

use serde_json::{Map, json};

use rag::index::engine::{Context, Engine, FieldsConfig};

use common::mock_storage::{MockStorage, make_es_response};

#[tokio::test]
async fn vector_search_uses_vector_backend() {
    let storage = Box::new(MockStorage::new(make_es_response(vec![])));
    let vector = Box::new(MockStorage::new(make_es_response(vec![(
        "v1".into(),
        "vec".into(),
        0.8,
    )])));
    let storage_calls = std::sync::Arc::clone(&storage.call_count);
    let vector_calls = std::sync::Arc::clone(&vector.call_count);
    let engine = Engine::new(storage, Some(vector), None, 100, 0.0, 60, Map::new()).unwrap();

    let r = engine
        .vector_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            Map::new(),
            100,
        )
        .await
        .unwrap();

    assert_eq!(r.total, 1);
    assert_eq!(*vector_calls.lock().unwrap(), 1);
    assert_eq!(*storage_calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn vector_search_no_backend_returns_no_vector_backend() {
    let storage = Box::new(MockStorage::new(make_es_response(vec![])));
    let storage_calls = std::sync::Arc::clone(&storage.call_count);
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    let err = engine
        .vector_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            Map::new(),
            100,
        )
        .await
        .unwrap_err();

    assert!(matches!(err, rag::index::engine::Error::NoVectorBackend));
    // text storage MUST NOT be called.
    assert_eq!(*storage_calls.lock().unwrap(), 0);
}
