//! `pagewiki::spans` —— LLM Evidence → Span 的字符坐标反算助手。
//!
//! 详见 `openspec/changes/rag-page-wiki/design.md` 第 5 节。
//!
//! 全部为同步纯函数。`Span.start / Span.end` 一律为**字符下标**
//! （Unicode scalar value index，`text.chars().count()` 口径），
//! 与 `docs/feature-design/05-wiki-page.md` 第 2 节保持一致。

use crate::index::pagewiki::types::{Error, Evidence, Span};

/// 行号窗口：在 `[line_hint - WINDOW, line_hint + WINDOW]` 行内优先搜索锚点，
/// 未命中再退化为全文搜索。
pub(crate) const NEAR_LINE_WINDOW: usize = 5;

/// 把若干 [`Evidence`] 批量反算为 [`Span`]。
///
/// 任一 evidence 反算失败立即返回 [`Error::SpanResolve`]。
pub fn resolve_spans(text: &str, evidence_list: &[Evidence]) -> Result<Vec<Span>, Error> {
    evidence_list
        .iter()
        .map(|ev| resolve_span(text, ev))
        .collect()
}

/// 把单个 [`Evidence`] 反算为 [`Span`]。
///
/// 行为：
/// 1. 通过 [`find_text_near_line`] 定位 `start_text`、`end_text` 各自的字符起点；
/// 2. 若 `evidence.extra.include_end_text == true`，则 `end` 推进 `end_text` 自身长度；
/// 3. `original_text` 由 `text[start..end]`（字符口径）截取生成。
pub fn resolve_span(text: &str, evidence: &Evidence) -> Result<Span, Error> {
    let start = find_text_near_line(text, &evidence.start_text, evidence.start_line)?;
    let end_anchor = find_text_near_line(text, &evidence.end_text, evidence.end_line)?;
    let include_end = evidence
        .extra
        .get("include_end_text")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let end = if include_end {
        end_anchor + evidence.end_text.chars().count()
    } else {
        end_anchor
    };
    if end < start {
        return Err(Error::SpanResolve(format!(
            "end ({end}) precedes start ({start})"
        )));
    }
    let original_text = char_slice(text, start, end)?;
    Ok(Span {
        start,
        end,
        original_text,
        extra: evidence.extra.clone(),
    })
}

/// 在 `line_hint` 附近 [`NEAR_LINE_WINDOW`] 行内查找 `target`，未命中退化为全文搜索。
///
/// 返回 `target` 首字符在 `text` 中的**字符下标**。
pub fn find_text_near_line(text: &str, target: &str, line_hint: usize) -> Result<usize, Error> {
    if target.is_empty() {
        return Err(Error::SpanResolve("target text is empty".into()));
    }
    let total_chars = text.chars().count();
    let line_starts = collect_line_starts(text);

    // 优先在窗口内搜索。
    if !line_starts.is_empty() {
        let line_count = line_starts.len();
        let hint_idx = line_hint.saturating_sub(1).min(line_count - 1);
        let lo_idx = hint_idx.saturating_sub(NEAR_LINE_WINDOW);
        let hi_idx_excl = (hint_idx + NEAR_LINE_WINDOW + 1).min(line_count);
        let win_start = line_starts[lo_idx];
        let win_end = if hi_idx_excl < line_count {
            line_starts[hi_idx_excl]
        } else {
            total_chars
        };
        if let Some(found) = char_find(text, target, win_start, win_end) {
            return Ok(found);
        }
    }

    // 退化为全文搜索。
    if let Some(found) = char_find(text, target, 0, total_chars) {
        return Ok(found);
    }

    Err(Error::SpanResolve(format!(
        "target {target:?} not found near line {line_hint}"
    )))
}

/// 在 `text` 的 `[start_char, end_char)` 字符区间内查找 `needle`，返回**字符下标**。
fn char_find(text: &str, needle: &str, start_char: usize, end_char: usize) -> Option<usize> {
    if start_char >= end_char {
        return None;
    }
    let byte_start = char_index_to_byte(text, start_char)?;
    let byte_end = char_index_to_byte(text, end_char)?;
    let slice = &text[byte_start..byte_end];
    let local_byte = slice.find(needle)?;
    let global_byte = byte_start + local_byte;
    Some(byte_index_to_char(text, global_byte))
}

/// 截取 `text[start_char..end_char]`（字符口径），返回新字符串。
fn char_slice(text: &str, start_char: usize, end_char: usize) -> Result<String, Error> {
    let total = text.chars().count();
    if start_char > end_char || end_char > total {
        return Err(Error::SpanResolve(format!(
            "char range out of bounds: [{start_char}, {end_char}) total={total}"
        )));
    }
    let byte_start = char_index_to_byte(text, start_char)
        .ok_or_else(|| Error::SpanResolve(format!("char index out of bounds: {start_char}")))?;
    let byte_end = char_index_to_byte(text, end_char)
        .ok_or_else(|| Error::SpanResolve(format!("char index out of bounds: {end_char}")))?;
    Ok(text[byte_start..byte_end].to_string())
}

/// 字符下标 → 字节偏移（允许 `char_idx == chars().count()`，返回 `text.len()`）。
fn char_index_to_byte(text: &str, char_idx: usize) -> Option<usize> {
    if char_idx == 0 {
        return Some(0);
    }
    let mut count = 0usize;
    for (b, _) in text.char_indices() {
        if count == char_idx {
            return Some(b);
        }
        count += 1;
    }
    if count == char_idx {
        Some(text.len())
    } else {
        None
    }
}

/// 字节偏移 → 字符下标。
fn byte_index_to_char(text: &str, byte_idx: usize) -> usize {
    let mut count = 0usize;
    for (b, _) in text.char_indices() {
        if b >= byte_idx {
            return count;
        }
        count += 1;
    }
    count
}

/// 收集每一行的起始字符下标（1-based 对齐 `evidence.start_line`）。
///
/// `text` 为空 → 返回空 vec。否则首行总是从 0 开始。
fn collect_line_starts(text: &str) -> Vec<usize> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut starts = vec![0usize];
    let mut count = 0usize;
    for ch in text.chars() {
        count += 1;
        if ch == '\n' {
            starts.push(count);
        }
    }
    let total = text.chars().count();
    // 末尾换行会产生一个等于 total 的虚拟起点（其后无内容），去掉。
    if let Some(&last) = starts.last()
        && last >= total
        && starts.len() > 1
    {
        starts.pop();
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn resolve_span_basic_ascii() {
        let text = "hello world\nfoo bar baz\n";
        let span = resolve_span(text, &ev("world", "baz", 1, 2)).unwrap();
        assert_eq!(span.start, 6);
        assert_eq!(span.end, 20);
        assert_eq!(span.original_text, "world\nfoo bar ");
    }

    #[test]
    fn resolve_span_with_include_end_text() {
        let text = "hello world\nfoo bar baz\n";
        let mut e = ev("world", "baz", 1, 2);
        e.extra
            .insert("include_end_text".into(), serde_json::Value::Bool(true));
        let span = resolve_span(text, &e).unwrap();
        assert_eq!(span.start, 6);
        assert_eq!(span.end, 23);
        assert_eq!(span.original_text, "world\nfoo bar baz");
    }

    #[test]
    fn resolve_span_chinese_uses_char_index() {
        let text = "你好世界\n再见世界\n";
        let span = resolve_span(text, &ev("世界", "再见", 1, 2)).unwrap();
        assert_eq!(span.start, 2);
        assert_eq!(span.end, 5);
        assert_eq!(span.original_text, "世界\n");
    }

    #[test]
    fn find_text_falls_back_to_full_scan() {
        let text = "line1\nline2\nline3\nline4\nline5\nline6\ntargethere\nline8\n";
        let pos = find_text_near_line(text, "targethere", 1).unwrap();
        // "targethere" 出现在第 7 行起点。
        // 前 6 行每行 "lineN" 5 字符 + 换行 1 = 6 字符，共 36 字符。
        assert_eq!(pos, 36);
    }

    #[test]
    fn find_text_returns_err_when_missing() {
        let text = "abc\ndef\n";
        let err = find_text_near_line(text, "zzz", 1).unwrap_err();
        assert!(matches!(err, Error::SpanResolve(_)));
    }

    #[test]
    fn empty_target_is_rejected() {
        let err = find_text_near_line("abc", "", 1).unwrap_err();
        assert!(matches!(err, Error::SpanResolve(_)));
    }

    #[test]
    fn resolve_spans_collects_all() {
        let text = "alpha beta gamma\n";
        let evs = vec![ev("alpha", "beta", 1, 1), ev("beta", "gamma", 1, 1)];
        let spans = resolve_spans(text, &evs).unwrap();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[1].start, 6);
    }

    #[test]
    fn near_line_window_prefers_local_match() {
        // "foo" 在第 1 行和第 7 行都出现；line_hint=7 时应命中第 7 行。
        let text = "foo first\nA\nB\nC\nD\nE\nfoo seventh\n";
        let pos = find_text_near_line(text, "foo", 7).unwrap();
        // 前 6 行字符数: "foo first\n"(10) + "A\n"(2)*5 = 20
        assert_eq!(pos, 20);
    }
}
