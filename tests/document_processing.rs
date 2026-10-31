use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use futures_executor::block_on;
use halo::agent::Cancellation;
use halo::agent::document::{self, Document, Loader as _, Parser as _, Transformer as _};
use serde_json::{Map, json};

fn parser(
    method: &str,
    format: &str,
    engine: &str,
    options: serde_json::Value,
) -> document::parser::Parser {
    document::parser::new(
        method,
        json!({"format":format,"engine":engine,"options":options}),
    )
    .unwrap()
}
fn document(content: &str) -> Document {
    Document {
        content: content.into(),
        ..Default::default()
    }
}

#[test]
fn local_file_to_parser_to_transformer_runs_without_runtime_or_storage() {
    let cancellation = Cancellation::new();
    let source = document::Source {
        uri: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qa.csv").into(),
    };
    let data = block_on(document::LocalLoader.load(&source, &cancellation)).unwrap();
    let mut parsed =
        block_on(parser("qa", "csv", "csv", json!({})).parse(&data, &cancellation)).unwrap();
    assert_eq!(parsed.len(), 2);
    for document in &mut parsed {
        assert!(document.id.is_none() && document.doc_id.is_none());
        document.doc_id = Some(42);
        document
            .metadata
            .insert("source".into(), json!({"uri": source.uri}));
    }
    let result = block_on(
        document::transformer::new()
            .remove_empty()
            .deduplicate()
            .transform(parsed, &cancellation),
    )
    .unwrap();
    assert_eq!(result.len(), 2);
    assert!(
        result[0].metadata["qa"]["answer"]
            .as_str()
            .unwrap()
            .contains('\n')
    );
    assert_eq!(result[0].doc_id, Some(42));
    assert_eq!(result[0].metadata["source"]["uri"], source.uri);
}

#[test]
fn parser_construction_rejects_malformed_and_unsupported_configurations() {
    for config in [
        json!(null),
        json!({}),
        json!({"format":3,"engine":"text"}),
        json!({"format":"text","engine":"text","options":null}),
        json!({"format":"text","engine":"text","options":{"trim":"true"}}),
        json!({"format":"text","engine":"text","options":{"unknown":true}}),
    ] {
        assert_eq!(
            document::parser::new("general", config).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }
    for (method, format, engine) in [
        ("laws", "text", "text"),
        ("general", "pdf", "text"),
        ("general", "text", "ocr"),
        ("qa", "text", "text"),
        ("table", "markdown", "markdown"),
    ] {
        assert_eq!(
            document::parser::new(method, json!({"format":format,"engine":engine}))
                .unwrap_err()
                .code,
            "UNSUPPORTED"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn liteparse_reads_a_real_pdf_through_the_document_parser_contract() {
    let data = include_bytes!("fixtures/liteparse.pdf");
    let parser = document::parser::new("general", json!({"format":"pdf"})).unwrap();
    let pages = parser.parse(data, &Cancellation::new()).await.unwrap();
    assert_eq!(pages.len(), 1);
    assert!(pages[0].content.contains("Heart LiteParse integration"));
    assert_eq!(pages[0].metadata["parser"], "liteparse");
    assert_eq!(
        pages[0].metadata["location"]["page"],
        json!({"from":1,"to":1})
    );
    assert!(pages[0].id.is_none() && pages[0].doc_id.is_none());

    let one = document::parser::new("one", json!({"format":"pdf"})).unwrap();
    let documents = one.parse(data, &Cancellation::new()).await.unwrap();
    assert_eq!(documents.len(), 1);
    assert!(documents[0].content.contains("Heart LiteParse integration"));
    assert!(documents[0].metadata.get("location").is_none());
}

#[test]
fn liteparse_options_are_validated_before_parsing() {
    for options in [
        json!({"ocr":"yes"}),
        json!({"max_pages":0}),
        json!({"target_pages":""}),
        json!({"output":"json"}),
    ] {
        assert_eq!(
            document::parser::new("general", json!({"format":"pdf","options":options}))
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
}

#[test]
fn one_preserves_original_text_and_leaves_identity_to_the_application() {
    let input = "  北京\r\n\r\n露营🙂  ";
    let parsed = block_on(
        parser("one", "text", "text", json!({})).parse(input.as_bytes(), &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].content, input);
    assert!(parsed[0].id.is_none() && parsed[0].doc_id.is_none());
    assert!(parsed[0].metadata.get("source").is_none());
    assert!(parsed[0].metadata.get("location").is_none());
    assert_eq!(parsed[0].metadata["method"], "text");
}

#[test]
fn plain_parser_recognizes_original_paragraphs_without_length_chunking() {
    let parsed = block_on(parser("general", "text", "text", json!({})).parse(
        "第一段\r\n续行\r\n\r\n第二段".as_bytes(),
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].content, "第一段\r\n续行\r\n");
    assert_eq!(parsed[1].content, "第二段");
    assert_eq!(
        parsed[0].metadata["location"],
        json!({"paragraph":{"from":1,"to":1}})
    );
    assert_eq!(parsed[1].metadata["location"]["paragraph"]["from"], 2);
    assert!(parsed[0].metadata["location"].get("char").is_none());
}

#[test]
fn parser_normalization_is_explicit_and_does_not_publish_false_char_offsets() {
    let parsed = block_on(
        parser(
            "one",
            "text",
            "text",
            json!({"trim":true,"normalize_newlines":true}),
        )
        .parse("  北京\r\n露营 \r\n".as_bytes(), &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(parsed[0].content, "北京\n露营");
    assert!(parsed[0].metadata.get("location").is_none());
}

#[test]
fn utf8_bom_is_decoded_but_invalid_encoding_is_not_lossily_replaced() {
    let parsed = block_on(
        parser("one", "text", "text", json!({})).parse(b"\xef\xbb\xbfhello", &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(parsed[0].content, "hello");
    assert_eq!(
        block_on(
            parser("one", "text", "text", json!({})).parse(&[0xff, 0xfe], &Cancellation::new())
        )
        .unwrap_err()
        .code,
        "PARSE_ERROR"
    );
}

#[test]
fn markdown_recognizes_blocks_lists_code_and_tables_without_rendered_location() {
    let input = "# 标题\n\n正文 **粗体**\n\n- 项一\n- 项二\n\n```python\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n";
    let parsed = block_on(
        parser("general", "markdown", "markdown", json!({}))
            .parse(input.as_bytes(), &Cancellation::new()),
    )
    .unwrap();
    let roles: Vec<_> = parsed
        .iter()
        .map(|document| document.metadata["layout"]["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, vec!["heading", "paragraph", "list", "code", "table"]);
    assert!(parsed[3].content.contains("```python"));
    assert!(parsed[1].content.contains("**粗体**"));
    assert!(
        parsed
            .iter()
            .all(|document| document.metadata.get("location").is_none())
    );
}

#[test]
fn qa_csv_uses_configured_columns_and_supports_quoted_multiline_values() {
    let input = "q,a,other\r\n\"天气,好吗\",\"第一行\r\n第二行\",ignored\r\n";
    let parsed = block_on(
        parser("qa", "csv", "csv", json!({"question":"q","answer":"a"}))
            .parse(input.as_bytes(), &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].metadata["qa"]["question"], "天气,好吗");
    assert_eq!(parsed[0].metadata["qa"]["answer"], "第一行\r\n第二行");
    assert!(parsed[0].content.contains("第一行\r\n第二行"));
    assert!(parsed[0].metadata.get("location").is_none());
}

#[test]
fn qa_csv_preserves_zero_false_and_empty_answers_as_actual_fields() {
    let parsed = block_on(parser("qa", "csv", "csv", json!({})).parse(
        b"question,answer\nq1,0\nq2,false\nq3,\n",
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0].metadata["qa"]["answer"], "0");
    assert_eq!(parsed[1].metadata["qa"]["answer"], "false");
    assert_eq!(parsed[2].metadata["qa"]["answer"], "");
}

#[test]
fn csv_table_retains_columns_quotes_and_configurable_delimiters() {
    let parsed = block_on(
        parser("table", "csv", "csv", json!({"delimiter":"\t"})).parse(
            b"name\tvalue\n\"a \"\"quote\"\"\"\t0\n",
            &Cancellation::new(),
        ),
    )
    .unwrap();
    assert_eq!(
        parsed[0].metadata["table"]["fields"],
        json!({"name":"a \"quote\"","value":"0"})
    );
    assert_eq!(parsed[0].metadata["layout"]["role"], "row");
}

#[test]
fn csv_rejects_ambiguous_headers_missing_columns_and_incomplete_late_rows() {
    for input in [
        "question,answer\nok,yes\nonly_one_field\n",
        "question,question\nx,y\n",
        "wrong,answer\nx,y\n",
        "question,answer\nx,\"unfinished",
        "question,answer\nx,\"closed\"junk",
    ] {
        assert_eq!(
            block_on(
                parser("qa", "csv", "csv", json!({})).parse(input.as_bytes(), &Cancellation::new())
            )
            .unwrap_err()
            .code,
            "PARSE_ERROR"
        );
    }
}

#[test]
fn csv_options_cannot_panic_by_indexing_empty_or_multibyte_delimiters() {
    for delimiter in ["", "。", "\n", "\r", "\"", "\0", "ab"] {
        assert_eq!(
            document::parser::new(
                "table",
                json!({"format":"csv","engine":"csv","options":{"delimiter":delimiter}})
            )
            .unwrap_err()
            .code,
            "INVALID_ARGUMENTS"
        );
    }
    assert_eq!(
        document::parser::new(
            "qa",
            json!({"format":"json","engine":"json","options":{"question":"x","answer":"x"}})
        )
        .unwrap_err()
        .code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn general_json_selects_text_and_does_not_reinterpret_application_ids() {
    let input = json!([{"id":123,"doc_id":456,"body":"北京"},{"body":"露营"}]).to_string();
    let parsed = block_on(
        parser("general", "json", "json", json!({"content":"body"}))
            .parse(input.as_bytes(), &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(parsed[0].content, "北京");
    assert_eq!(parsed[1].content, "露营");
    assert!(
        parsed
            .iter()
            .all(|document| document.id.is_none() && document.doc_id.is_none())
    );
}

#[test]
fn json_qa_and_table_preserve_real_scalar_and_nested_field_values() {
    let parsed = block_on(parser("qa", "json", "json", json!({})).parse(
        br#"[{"question":"zero?","answer":0},{"question":"enabled?","answer":false}]"#,
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(parsed[0].metadata["qa"]["answer"], "0");
    assert_eq!(parsed[1].metadata["qa"]["answer"], "false");
    let parsed = block_on(
        parser("table", "json", "json", json!({}))
            .parse(br#"[{"a":false,"b":{"x":1}}]"#, &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(
        parsed[0].metadata["table"]["fields"],
        json!({"a":false,"b":{"x":1}})
    );
}

#[test]
fn malformed_json_or_late_row_fails_the_whole_call() {
    for input in [
        "{broken",
        "[\"valid\",123]",
        "[{\"content\":\"valid\"},{\"content\":null}]",
    ] {
        assert_eq!(
            block_on(
                parser("general", "json", "json", json!({}))
                    .parse(input.as_bytes(), &Cancellation::new())
            )
            .unwrap_err()
            .code,
            "PARSE_ERROR"
        );
    }
    assert_eq!(
        block_on(parser("one", "json", "json", json!({})).parse(b"{broken", &Cancellation::new()))
            .unwrap_err()
            .code,
        "PARSE_ERROR"
    );
}

#[test]
fn empty_parser_results_are_explicit_and_configurable_for_whole_inputs() {
    assert!(
        block_on(parser("one", "text", "text", json!({})).parse(b" ", &Cancellation::new()))
            .unwrap()
            .is_empty()
    );
    let parsed = block_on(
        parser("one", "text", "text", json!({"remove_empty":false}))
            .parse(b" ", &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].content, " ");
    assert!(
        block_on(parser("general", "json", "json", json!({})).parse(b"[]", &Cancellation::new()))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn cancelled_parser_returns_no_success_even_for_empty_or_invalid_input() {
    let cancellation = Cancellation::new();
    cancellation.cancel();
    for input in [b"valid".as_slice(), &[0xff], b""] {
        assert_eq!(
            block_on(parser("general", "text", "text", json!({})).parse(input, &cancellation))
                .unwrap_err()
                .code,
            "CANCELLED"
        );
    }
}

#[test]
fn metadata_merge_preserves_existing_identity_and_source_facts() {
    let mut original = document("北京");
    original.id = Some(1);
    original.doc_id = Some(10);
    original.metadata = serde_json::from_value(json!({"source":{"uri":"a.txt"},"location":{"paragraph":{"from":1,"to":1}},"vector":[0.1],"custom":"old"})).unwrap();
    let result = block_on(
        document::transformer::new()
            .metadata(Map::from_iter([("custom".into(), json!("new"))]))
            .transform(vec![original.clone()], &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(result[0].id, original.id);
    assert_eq!(result[0].metadata["source"], original.metadata["source"]);
    assert_eq!(
        result[0].metadata["location"],
        original.metadata["location"]
    );
    assert_eq!(result[0].metadata["vector"], original.metadata["vector"]);
    assert_eq!(result[0].metadata["custom"], "new");
}

#[test]
fn fluent_transformations_obey_configuration_order_and_stable_output_order() {
    let transform = document::transformer::new()
        .map(|mut document| {
            document.content = document.content.trim().into();
            Ok(document)
        })
        .remove_empty()
        .deduplicate()
        .filter(|document| document.content != "skip");
    let result = block_on(transform.transform(
        vec![
            document(" a "),
            document(""),
            document("a"),
            document("skip"),
            document("b"),
        ],
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(
        result
            .iter()
            .map(|document| document.content.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    let transform = document::transformer::new()
        .deduplicate()
        .map(|mut document| {
            document.content = document.content.trim().into();
            Ok(document)
        });
    assert_eq!(
        block_on(transform.transform(vec![document(" a "), document("a")], &Cancellation::new()))
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn duplicate_text_from_different_sources_is_not_silently_merged() {
    let mut a = document("same");
    a.doc_id = Some(1);
    let mut b = document("same");
    b.doc_id = Some(2);
    let result = block_on(
        document::transformer::new()
            .deduplicate()
            .transform(vec![a.clone(), a, b], &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].doc_id, Some(1));
    assert_eq!(result[1].doc_id, Some(2));
}

#[test]
fn duplicate_metadata_is_order_independent_without_cloning_an_entire_corpus() {
    let mut a = document("same");
    a.metadata.insert("a".into(), json!(1));
    a.metadata.insert("b".into(), json!(2));
    let mut b = document("same");
    b.metadata.insert("b".into(), json!(2));
    b.metadata.insert("a".into(), json!(1));
    let result = block_on(
        document::transformer::new()
            .deduplicate()
            .transform(vec![a, b], &Cancellation::new()),
    )
    .unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn content_changes_drop_stale_location_vector_and_method_but_keep_source() {
    let mut original = document("before");
    original.metadata=serde_json::from_value(json!({"location":{"paragraph":{"from":1,"to":1}},"method":"text","vector":[0.1],"source":"a.txt","business":1})).unwrap();
    let transform = document::transformer::new().map(|mut document| {
        document.content = "after".into();
        Ok(document)
    });
    let result = block_on(transform.transform(vec![original], &Cancellation::new())).unwrap();
    for key in ["location", "vector", "method"] {
        assert!(result[0].metadata.get(key).is_none());
    }
    assert_eq!(result[0].metadata["source"], "a.txt");
    assert_eq!(result[0].metadata["business"], 1);
}

#[test]
fn explicit_updated_provenance_is_not_deleted_by_custom_transformation() {
    let mut original = document("before");
    original.metadata =
        serde_json::from_value(json!({"location":{"paragraph":{"from":1,"to":1}},"vector":[0.1]}))
            .unwrap();
    let transform = document::transformer::new().map(|mut document| {
        document.content = "after".into();
        document.metadata.insert("vector".into(), json!([0.2]));
        document
            .metadata
            .insert("location".into(), json!({"paragraph":{"from":2,"to":2}}));
        Ok(document)
    });
    let result = block_on(transform.transform(vec![original], &Cancellation::new())).unwrap();
    assert_eq!(result[0].metadata["vector"], json!([0.2]));
    assert_eq!(result[0].metadata["location"]["paragraph"]["from"], 2);
}

#[test]
fn custom_transform_error_is_returned_once_without_partial_result_or_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let transform = document::transformer::new().map(move |document| {
        counter.fetch_add(1, Ordering::SeqCst);
        if document.content == "bad" {
            Err(document::Error::new("CUSTOM_ERROR", "bad input"))
        } else {
            Ok(document)
        }
    });
    assert_eq!(
        block_on(transform.transform(
            vec![document("ok"), document("bad"), document("never")],
            &Cancellation::new()
        ))
        .unwrap_err()
        .code,
        "CUSTOM_ERROR"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn cancellation_from_hook_stops_later_documents_without_returning_partial_success() {
    let cancellation = Cancellation::new();
    let trigger = cancellation.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let transform = document::transformer::new().map(move |document| {
        counter.fetch_add(1, Ordering::SeqCst);
        trigger.cancel();
        Ok(document)
    });
    assert_eq!(
        block_on(transform.transform(vec![document("a"), document("b")], &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        block_on(document::transformer::new().transform(Vec::new(), &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}
