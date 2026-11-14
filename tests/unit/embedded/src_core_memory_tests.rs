use super::*;
use futures_executor::block_on;
use serde_json::json;

fn memory_request(id: i64, content: &str) -> Request {
    Request {
        id,
        r#type: Type::Fact,
        user_id: 7,
        agent_id: Some(9),
        session_id: None,
        from: 0,
        to: 10,
        content: content.into(),
        metadata: None,
        vector: None,
    }
}

fn options() -> serde_json::Map<String, Value> {
    serde_json::from_value(json!({
        "filter": {"user_id": 7, "agent_id": 9, "session_id": null, "type": "fact"}
    }))
    .unwrap()
}

#[test]
fn writes_updates_and_deletes_with_version_checks() {
    let memory = InMemory::new();
    let cancellation = Cancellation::new();
    let item = block_on(memory.write(memory_request(1, "北京天气"), &cancellation)).unwrap();
    assert_eq!(item.version, 1);
    assert_eq!(
        block_on(memory.write(memory_request(2, "另一条"), &cancellation))
            .unwrap_err()
            .code,
        "CONFLICT"
    );

    let mut changed = memory_request(1, "北京天气晴");
    changed.metadata = Some(serde_json::Map::from_iter([(
        String::from("source"),
        json!("chat"),
    )]));
    let updated = block_on(memory.update(changed, item.version, &cancellation)).unwrap();
    assert_eq!(updated.version, 2);
    assert_eq!(
        block_on(memory.update(memory_request(1, "过期"), 1, &cancellation))
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    block_on(memory.delete(1, &cancellation)).unwrap();
    assert!(
        block_on(memory.get(7, Some(9), None, Type::Fact, &cancellation))
            .unwrap()
            .is_none()
    );
}

#[test]
fn keyword_requires_exact_scope_and_supports_pagination() {
    let memory = InMemory::new();
    let cancellation = Cancellation::new();
    block_on(memory.write(memory_request(1, "北京天气晴"), &cancellation)).unwrap();
    let mut different = memory_request(2, "上海天气");
    different.r#type = Type::Preference;
    block_on(memory.write(different, &cancellation)).unwrap();
    let result = block_on(memory.keyword(
        KeywordRequest {
            query: "天气".into(),
            options: options(),
        },
        &cancellation,
    ))
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id, 1);
}

#[test]
fn vector_capability_is_explicit_and_dimension_is_fixed() {
    let cancellation = Cancellation::new();
    let memory = InMemory::new();
    let request = VectorRequest {
        vector: vec![1.0],
        options: options(),
    };
    assert_eq!(
        block_on(memory.vector(request, &cancellation))
            .unwrap_err()
            .code,
        "UNSUPPORTED"
    );

    let memory = InMemory::with_vector_dimension(2).unwrap();
    let mut item = memory_request(3, "北京天气");
    item.vector = Some(vec![1.0, 0.0]);
    block_on(memory.write(item, &cancellation)).unwrap();
    let result = block_on(memory.vector(
        VectorRequest {
            vector: vec![1.0, 0.0],
            options: options(),
        },
        &cancellation,
    ))
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id, 3);
}

#[test]
fn search_requires_every_scope_field_including_explicit_nulls() {
    let memory = InMemory::with_vector_dimension(2).unwrap();
    let cancellation = Cancellation::new();
    for field in ["user_id", "agent_id", "session_id", "type"] {
        let mut options = options();
        options["filter"].as_object_mut().unwrap().remove(field);
        let error = block_on(memory.keyword(
            KeywordRequest {
                query: "天气".into(),
                options: options.clone(),
            },
            &cancellation,
        ))
        .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
        assert_eq!(error.details, Some(format!("filter.{field}")));
        assert_eq!(
            block_on(memory.vector(
                VectorRequest {
                    vector: vec![1.0, 0.0],
                    options: options.clone(),
                },
                &cancellation
            ))
            .unwrap_err()
            .code,
            "INVALID_ARGUMENTS"
        );
        assert_eq!(
            block_on(memory.hybrid(
                HybridRequest {
                    query: "天气".into(),
                    vector: vec![1.0, 0.0],
                    options,
                },
                &cancellation
            ))
            .unwrap_err()
            .code,
            "INVALID_ARGUMENTS"
        );
    }
    let options = serde_json::from_value(json!({
        "filter": {"user_id": 7, "agent_id": null, "session_id": null, "type": "fact"}
    }))
    .unwrap();
    assert!(
        block_on(memory.keyword(
            KeywordRequest {
                query: "天气".into(),
                options
            },
            &cancellation
        ))
        .unwrap()
        .is_empty()
    );
}

#[test]
fn cancellation_does_not_mutate_memory() {
    let memory = InMemory::new();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        block_on(memory.write(memory_request(1, "ignored"), &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}

fn malformed_paginations() -> Vec<Value> {
    vec![
        Value::Null,
        json!("20"),
        json!([]),
        json!(true),
        json!({"offset": -1}),
        json!({"offset": "1"}),
        json!({"offset": 1.5}),
        json!({"offset": null}),
        json!({"limit": -1}),
        json!({"limit": "20"}),
        json!({"limit": true}),
        json!({"limit": null}),
        json!({"limit": 0}),
    ]
}

#[test]
fn keyword_rejects_malformed_pagination_instead_of_defaulting() {
    let memory = InMemory::new();
    let cancellation = Cancellation::new();
    for pagination in malformed_paginations() {
        let mut options = options();
        options.insert("pagination".into(), pagination);
        let error = block_on(memory.keyword(
            KeywordRequest {
                query: "天气".into(),
                options,
            },
            &cancellation,
        ))
        .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
    }
}

#[test]
fn vector_rejects_malformed_pagination_instead_of_defaulting() {
    let memory = InMemory::with_vector_dimension(2).unwrap();
    let cancellation = Cancellation::new();
    for pagination in malformed_paginations() {
        let mut options = options();
        options.insert("pagination".into(), pagination);
        let error = block_on(memory.vector(
            VectorRequest {
                vector: vec![1.0, 0.0],
                options,
            },
            &cancellation,
        ))
        .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
    }
}

#[test]
fn hybrid_rejects_malformed_pagination_instead_of_defaulting() {
    let memory = InMemory::with_vector_dimension(2).unwrap();
    let cancellation = Cancellation::new();
    for pagination in malformed_paginations() {
        let mut options = options();
        options.insert("pagination".into(), pagination);
        let error = block_on(memory.hybrid(
            HybridRequest {
                query: "天气".into(),
                vector: vec![1.0, 0.0],
                options,
            },
            &cancellation,
        ))
        .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
    }
}

#[test]
fn pagination_defaults_only_apply_to_absent_fields() {
    assert_eq!(pagination(&options()).unwrap(), (0, 20));
    for (value, expected) in [
        (json!({}), (0, 20)),
        (json!({"offset": 3}), (3, 20)),
        (json!({"limit": 5}), (0, 5)),
        (json!({"offset": 0, "limit": 5}), (0, 5)),
    ] {
        let mut options = options();
        options.insert("pagination".into(), value);
        assert_eq!(pagination(&options).unwrap(), expected);
    }
}

#[test]
fn keyword_rejects_wrong_sort_types() {
    let memory = InMemory::new();
    let cancellation = Cancellation::new();
    for sort in [
        json!({"field": null}),
        json!({"field": 7}),
        json!({"order": false}),
        json!({"order": ["desc"]}),
    ] {
        let mut options = options();
        options.insert("sort".into(), sort);
        let error = block_on(memory.keyword(
            KeywordRequest {
                query: "天气".into(),
                options,
            },
            &cancellation,
        ))
        .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
    }
}

#[test]
fn exhausted_version_leaves_entire_record_and_vector_unchanged() {
    let memory = InMemory::with_vector_dimension(2).unwrap();
    let cancellation = Cancellation::new();
    let mut original = memory_request(1, "original");
    original.vector = Some(vec![1.0, 0.0]);
    block_on(memory.write(original, &cancellation)).unwrap();
    let before = {
        let mut records = memory.records.blocking_write();
        let stored = records.get_mut(&1).unwrap();
        stored.item.version = u64::MAX;
        stored.clone()
    };
    let mut replacement = memory_request(1, "replacement");
    replacement.from = 10;
    replacement.to = 20;
    replacement.vector = Some(vec![0.0, 1.0]);
    replacement.metadata = Some(serde_json::Map::from_iter([(
        "changed".into(),
        json!(true),
    )]));
    let error = block_on(memory.update(replacement, u64::MAX, &cancellation)).unwrap_err();
    assert_eq!(error.code, "BACKEND");
    let records = memory.records.blocking_read();
    assert_eq!(records[&1].item, before.item);
    assert_eq!(records[&1].vector, before.vector);
}

#[test]
fn cosine_remains_finite_for_extreme_finite_vectors() {
    for magnitude in [f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
        let forward = [magnitude, magnitude];
        let opposite = [-magnitude, -magnitude];
        let orthogonal = [magnitude, -magnitude];
        for (other, expected) in [(forward, 1.0), (opposite, -1.0), (orthogonal, 0.0)] {
            let score = cosine(&forward, &other);
            assert!(score.is_finite());
            assert!((score - expected).abs() < 1e-12);
        }
    }
    assert_eq!(cosine(&[0.0, 0.0], &[f32::MAX, f32::MAX]), 0.0);
}
