use futures_executor::block_on;
use halo_agents::{
    Cancellation,
    document::{self, Chunker as _, Document, Loader as _, Parser as _, Transformer as _},
};
use serde_json::{Value, json};

fn document(content: &str) -> Document {
    Document {
        content: content.into(),
        ..Default::default()
    }
}

fn split(method: &str, options: Value, documents: Vec<Document>) -> Vec<Document> {
    let chunker = document::chunker::new(method, options).unwrap();
    let mut id = 1000;
    block_on(chunker.chunk(documents, &Cancellation::new(), &mut || {
        id += 1;
        id
    }))
    .unwrap()
}

fn texts(chunks: &[Document]) -> Vec<&str> {
    chunks.iter().map(|chunk| chunk.content.as_str()).collect()
}

#[test]
fn fixed_counts_unicode_scalars_not_bytes_or_tokens() {
    let chunks = split(
        "fixed",
        json!({"max_chars":2}),
        vec![document("北京🙂a\r\n")],
    );
    assert_eq!(texts(&chunks), ["北京", "🙂a", "\r\n"]);
    assert_eq!(
        chunks.iter().map(|c| c.id.unwrap()).collect::<Vec<_>>(),
        [1001, 1002, 1003]
    );
}

#[test]
fn sliding_preserves_overlap_and_does_not_create_a_redundant_tail() {
    let chunks = split(
        "sliding",
        json!({"max_chars":4,"overlap":2}),
        vec![document("abcdef")],
    );
    assert_eq!(texts(&chunks), ["abcd", "cdef"]);
    let chunks = split(
        "sliding",
        json!({"max_chars":4,"overlap":2}),
        vec![document("abcdefg")],
    );
    assert_eq!(texts(&chunks), ["abcd", "cdef", "efg"]);
}

#[test]
fn sliding_with_zero_overlap_equals_fixed() {
    let inputs = vec![document("abc🙂def")];
    assert_eq!(
        split("fixed", json!({"max_chars":3}), inputs.clone()),
        split("sliding", json!({"max_chars":3}), inputs)
    );
}

#[test]
fn delimiter_preserves_crlf_and_merges_small_units() {
    let input = "ab\r\ncd\r\ne";
    let chunks = split(
        "delimiter",
        json!({"max_chars":8,"delimiter":"\r\n"}),
        vec![document(input)],
    );
    assert_eq!(texts(&chunks), ["ab\r\ncd\r\n", "e"]);
    assert_eq!(texts(&chunks).concat(), input);
}

#[test]
fn long_delimited_units_are_split_without_losing_repeated_separators() {
    let input = "甲乙丙丁戊。。，己庚辛壬。";
    let chunks = split(
        "delimiter",
        json!({"max_chars":3,"delimiter":"。"}),
        vec![document(input)],
    );
    assert_eq!(texts(&chunks).concat(), input);
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk.content.chars().count() <= 3)
    );
}

#[test]
fn fixed_and_delimiter_round_trip_small_unicode_corpora_and_sizes() {
    for input in [
        "",
        "abc",
        "甲。乙。丙",
        "🙂e\u{301}🇨🇳",
        "\r\n\r\n",
        "  \t  ",
        "。。。",
    ] {
        for size in 1..=12 {
            for method in ["fixed", "delimiter"] {
                let mut options = json!({"max_chars":size});
                if method == "delimiter" {
                    options["delimiter"] = json!("。");
                }
                let chunks = split(method, options, vec![document(input)]);
                assert_eq!(texts(&chunks).concat(), input);
                assert!(
                    chunks
                        .iter()
                        .all(|chunk| (1..=size).contains(&chunk.content.chars().count()))
                );
            }
        }
    }
}

#[test]
fn configuration_errors_are_not_silently_defaulted() {
    for (method, options) in [
        ("fixed", json!({})),
        ("fixed", json!({"max_chars":0})),
        ("fixed", json!({"max_chars":-1})),
        ("fixed", json!({"max_chars":1.5})),
        ("fixed", json!({"max_chars":2,"max_tokens":20})),
        ("fixed", json!({"max_chars":2,"overlap":1})),
        ("fixed", json!({"max_chars":2,"delimiter":"."})),
        ("sliding", json!({"max_chars":2,"overlap":2})),
        ("sliding", json!({"max_chars":2,"overlap":3})),
        ("sliding", json!({"max_chars":2,"overlap":-1})),
        ("sliding", json!({"max_chars":2,"delimiter":"."})),
        ("delimiter", json!({"max_chars":2})),
        ("delimiter", json!({"max_chars":2,"delimiter":""})),
        ("delimiter", json!({"max_chars":2,"delimiter":false})),
        (
            "delimiter",
            json!({"max_chars":2,"delimiter":".","overlap":0}),
        ),
        ("fixed", Value::Null),
        ("fixed", json!({"max_chars":2,"overlap":null})),
        ("sliding", json!({"max_chars":2,"overlap":null})),
        ("delimiter", json!({"max_chars":2,"delimiter":null})),
    ] {
        assert_eq!(
            document::chunker::new(method, options).err().unwrap().code,
            "INVALID_ARGUMENTS"
        );
    }
}

#[test]
fn semantic_and_unknown_methods_do_not_pretend_to_be_fixed_splitters() {
    for method in ["semantic", "laws", "qa", "FIXED", ""] {
        assert_eq!(
            document::chunker::new(method, json!({"max_chars":2}))
                .err()
                .unwrap()
                .code,
            "UNSUPPORTED"
        );
    }
}

#[test]
fn different_documents_remain_separate_and_keep_application_provenance() {
    let mut a = document("a");
    a.doc_id = Some(10);
    a.id = Some(11);
    a.metadata.insert("source".into(), json!({"uri":"a.txt"}));
    let mut b = document("b");
    b.doc_id = Some(20);
    let chunks = split(
        "delimiter",
        json!({"max_chars":100,"delimiter":"\n"}),
        vec![a, b],
    );
    assert_eq!(texts(&chunks), ["a", "b"]);
    assert_eq!(chunks[0].doc_id, Some(10));
    assert_eq!(chunks[1].doc_id, Some(20));
    assert_ne!(chunks[0].id, Some(11));
    assert_eq!(chunks[0].metadata["source"], json!({"uri":"a.txt"}));
}

#[test]
fn recutting_allocates_new_ids_instead_of_hashing_or_reusing_input_ids() {
    let chunker = document::chunker::new("fixed", json!({"max_chars":100})).unwrap();
    let mut id = 100;
    let mut next_id = || {
        id += 1;
        id
    };
    let input = vec![document("unchanged")];
    let first = block_on(chunker.chunk(input.clone(), &Cancellation::new(), &mut next_id)).unwrap();
    let second =
        block_on(chunker.chunk(first.clone(), &Cancellation::new(), &mut next_id)).unwrap();
    assert_eq!(texts(&first), texts(&second));
    assert_ne!(first[0].id, second[0].id);
}

#[test]
fn duplicate_generated_ids_return_error_without_partial_collection_or_retry() {
    let chunker = document::chunker::new("fixed", json!({"max_chars":1})).unwrap();
    let mut calls = 0;
    let result = block_on(
        chunker.chunk(vec![document("abc")], &Cancellation::new(), &mut || {
            calls += 1;
            100
        }),
    );
    assert_eq!(result.unwrap_err().code, "INVALID_ARGUMENTS");
    assert_eq!(calls, 2);
}

#[test]
fn generated_ids_cannot_reuse_an_existing_input_identity() {
    let chunker = document::chunker::new("fixed", json!({"max_chars":2})).unwrap();
    let mut input = document("abc");
    input.id = Some(100);
    let error =
        block_on(chunker.chunk(vec![input], &Cancellation::new(), &mut || 100)).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}

#[test]
fn empty_input_does_not_allocate_ids_but_whitespace_is_not_silently_trimmed() {
    let chunker = document::chunker::new("fixed", json!({"max_chars":2})).unwrap();
    assert!(
        block_on(
            chunker.chunk(vec![document("")], &Cancellation::new(), &mut || panic!(
                "no ID needed"
            ))
        )
        .unwrap()
        .is_empty()
    );
    let chunks = split("fixed", json!({"max_chars":2}), vec![document("  \t ")]);
    assert_eq!(texts(&chunks).concat(), "  \t ");
}

#[test]
fn cancelled_calls_do_not_allocate_or_expose_chunks() {
    let chunker = document::chunker::new("fixed", json!({"max_chars":2})).unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let error = block_on(
        chunker.chunk(vec![document("abc")], &cancellation, &mut || {
            panic!("cancelled")
        }),
    )
    .unwrap_err();
    assert_eq!(error.code, "CANCELLED");
}

#[test]
fn cancellation_during_id_assignment_also_prevents_successful_delivery() {
    let chunker = document::chunker::new("fixed", json!({"max_chars":2})).unwrap();
    let cancellation = Cancellation::new();
    let error = block_on(chunker.chunk(vec![document("ab")], &cancellation, &mut || {
        cancellation.cancel();
        100
    }))
    .unwrap_err();
    assert_eq!(error.code, "CANCELLED");
}

#[test]
fn single_paragraph_character_positions_are_adjusted_without_resetting_the_origin() {
    let mut input = document("甲🙂乙丙");
    input.metadata = json!({"source":{"uri":"x.txt"},"method":"text","vector":[0.1],"location":{
        "page":{"from":3,"to":3},"paragraph":{"from":2,"to":2},"char":{"from":11,"to":14},
        "x":{"from":0.1,"to":0.9},"y":{"from":0.2,"to":0.8}
    }})
    .as_object()
    .unwrap()
    .clone();
    let chunks = split("sliding", json!({"max_chars":3,"overlap":1}), vec![input]);
    assert_eq!(
        chunks[0].metadata["location"]["char"],
        json!({"from":11,"to":13})
    );
    assert_eq!(
        chunks[1].metadata["location"]["char"],
        json!({"from":13,"to":14})
    );
    for chunk in chunks {
        assert_eq!(chunk.metadata["location"]["page"], json!({"from":3,"to":3}));
        assert_eq!(chunk.metadata["method"], "text");
        assert!(chunk.metadata.get("vector").is_none());
        assert!(chunk.metadata["location"].get("x").is_none());
        assert!(chunk.metadata.get("source").is_some());
    }
}

#[test]
fn whole_text_chunk_keeps_existing_vector_and_location() {
    let mut input = document("ab");
    input.metadata = json!({"vector":[0.1],"location":{"page":{"from":1,"to":2}}})
        .as_object()
        .unwrap()
        .clone();
    let chunks = split("fixed", json!({"max_chars":2}), vec![input.clone()]);
    assert_eq!(chunks[0].metadata, input.metadata);
}

#[test]
fn sliced_multi_page_ranges_are_not_interpolated_or_copied_to_every_chunk() {
    let mut input = document("abcd");
    input.metadata = json!({"location":{"page":{"from":1,"to":2},"paragraph":{"from":1,"to":3},"char":{"from":1,"to":4}},"source":{"uri":"x.pdf"}}).as_object().unwrap().clone();
    let chunks = split("fixed", json!({"max_chars":2}), vec![input]);
    assert!(
        chunks
            .iter()
            .all(|chunk| !chunk.metadata.contains_key("location")
                && chunk.metadata.contains_key("source"))
    );
}

#[test]
fn partial_text_can_keep_known_page_without_inventing_paragraphs() {
    let mut input = document("abcd");
    input.metadata = json!({"location":{"page":{"from":3,"to":3},"paragraph":{"from":1,"to":2},"char":{"from":1,"to":4}}}).as_object().unwrap().clone();
    let chunks = split("fixed", json!({"max_chars":2}), vec![input]);
    assert!(
        chunks
            .iter()
            .all(|c| c.metadata["location"] == json!({"page":{"from":3,"to":3}}))
    );
}

#[test]
fn mismatched_or_invalid_character_ranges_are_not_claimed_as_valid() {
    for location in [
        json!({"paragraph":{"from":1,"to":1},"char":{"from":1,"to":10}}),
        json!({"paragraph":{"from":0,"to":0}}),
    ] {
        let mut input = document("abcd");
        input.metadata.insert("location".into(), location);
        let chunks = split("fixed", json!({"max_chars":2}), vec![input]);
        assert!(chunks.iter().all(|c| {
            c.metadata
                .get("location")
                .is_none_or(|l| l.get("char").is_none())
        }));
    }
}

#[test]
fn actual_local_file_pipeline_can_continue_through_chunking_without_runtime() {
    let cancellation = Cancellation::new();
    let source = document::Source {
        uri: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qa.csv").into(),
    };
    let bytes = block_on(document::LocalLoader.load(&source, &cancellation)).unwrap();
    let parser = document::parser::new("qa", json!({"format":"csv","engine":"csv"})).unwrap();
    let mut documents = block_on(parser.parse(&bytes, &cancellation)).unwrap();
    for document in &mut documents {
        document.doc_id = Some(900);
    }
    let documents = block_on(
        document::transformer::new()
            .remove_empty()
            .transform(documents, &cancellation),
    )
    .unwrap();
    let original: String = documents.iter().map(|d| d.content.as_str()).collect();
    let chunks = split(
        "delimiter",
        json!({"max_chars":12,"delimiter":"\n"}),
        documents,
    );
    assert_eq!(texts(&chunks).concat(), original);
    assert!(
        chunks
            .iter()
            .all(|c| c.doc_id == Some(900) && c.id.is_some())
    );
}
