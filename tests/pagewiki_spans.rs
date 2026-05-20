//! 集成测试：pagewiki::spans 的字符坐标反算。

use rag::index::pagewiki::{Error, Evidence, resolve_spans};
use serde_json::Map;

fn ev(start_text: &str, end_text: &str, start_line: usize, end_line: usize) -> Evidence {
    Evidence {
        start_text: start_text.into(),
        end_text: end_text.into(),
        start_line,
        end_line,
        extra: Map::new(),
    }
}

fn ev_with_include_end(
    start_text: &str,
    end_text: &str,
    start_line: usize,
    end_line: usize,
) -> Evidence {
    let mut e = ev(start_text, end_text, start_line, end_line);
    e.extra
        .insert("include_end_text".into(), serde_json::Value::Bool(true));
    e
}

#[test]
fn resolve_spans_multiple_evidences() {
    let text = "alpha beta gamma\ndelta epsilon zeta\n";
    let evs = vec![ev("alpha", "gamma", 1, 1), ev("delta", "zeta", 2, 2)];
    let spans = resolve_spans(text, &evs).unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].start, 0);
    assert_eq!(spans[0].end, 11); // "alpha beta " 11 字符（不含 gamma）
    assert_eq!(spans[1].start, 17); // delta 在第二行起点
}

#[test]
fn resolve_spans_cross_line_hit() {
    let text = "line one here\nline two here\nline three here\n";
    let span = resolve_spans(text, &[ev("one", "three", 1, 3)]).unwrap();
    assert_eq!(span[0].start, 5);
    // "three" 起点在第 3 行偏移 5 → 14 + 14 + 5 = 33。
    assert_eq!(span[0].end, 33);
    assert!(
        span[0]
            .original_text
            .starts_with("one here\nline two here\nline ")
    );
}

#[test]
fn include_end_text_true_extends_span() {
    let text = "hello world\nfoo bar baz\n";
    let span = resolve_spans(text, &[ev_with_include_end("world", "baz", 1, 2)]).unwrap();
    assert_eq!(span[0].start, 6);
    assert_eq!(span[0].end, 23);
    assert_eq!(span[0].original_text, "world\nfoo bar baz");
}

#[test]
fn include_end_text_false_excludes_end() {
    let text = "hello world\nfoo bar baz\n";
    let span = resolve_spans(text, &[ev("world", "baz", 1, 2)]).unwrap();
    assert_eq!(span[0].start, 6);
    assert_eq!(span[0].end, 20);
    assert_eq!(span[0].original_text, "world\nfoo bar ");
}

#[test]
fn resolve_spans_fails_on_missing_anchor() {
    let text = "abc\n";
    let err = resolve_spans(text, &[ev("missing", "abc", 1, 1)]).unwrap_err();
    assert!(matches!(err, Error::SpanResolve(_)));
}

#[test]
fn resolve_spans_propagates_extra() {
    let text = "foo bar\n";
    let mut e = ev("foo", "bar", 1, 1);
    e.extra.insert("score".into(), serde_json::json!(0.9));
    let spans = resolve_spans(text, &[e]).unwrap();
    assert_eq!(spans[0].extra.get("score"), Some(&serde_json::json!(0.9)));
}
