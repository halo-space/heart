//! `Qa`：将 JSONL 文档解析为问答型 PageWiki。
//!
//! 详见 `openspec/changes/rag-page-wiki/design.md` 第 6 节。

use crate::index::pagewiki::base::Base;
use crate::index::pagewiki::types::{Error, PageWiki, Span};
use std::future::Future;
use std::pin::Pin;

/// 单行 JSONL 的反序列化结构。
#[derive(Debug, serde::Deserialize)]
struct QaLine {
    question: String,
    answer: String,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
}

/// 问答型切分器（JSONL 输入）。
#[derive(Debug, Clone, Default)]
pub struct Qa;

impl Qa {
    /// 构造默认 [`Qa`]。
    pub fn new() -> Self {
        Self
    }
}

impl Base for Qa {
    fn cut<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PageWiki>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let mut out = Vec::new();
            let mut char_cursor = 0usize;
            for (idx, line) in text.split('\n').enumerate() {
                let line_char_len = line.chars().count();
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    // 空行跳过，但 char_cursor 仍要前进。
                    char_cursor += line_char_len + 1; // +1 = '\n'
                    continue;
                }
                let parsed: QaLine = serde_json::from_str(trimmed).map_err(|e| Error::QaParse {
                    line: idx + 1,
                    reason: e.to_string(),
                })?;

                let content = parsed.answer;
                let span = Span {
                    start: char_cursor,
                    end: char_cursor + line_char_len,
                    original_text: line.to_string(),
                    extra: Default::default(),
                };
                out.push(PageWiki {
                    content,
                    questions: vec![parsed.question],
                    keywords: parsed.keywords,
                    tags: parsed.tags,
                    spans: vec![span],
                    ..Default::default()
                });
                char_cursor += line_char_len + 1;
            }
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn two_valid_lines() {
        let jsonl = r#"{"question":"q1","answer":"a1","keywords":["k"]}
{"question":"q2","answer":"a2","tags":["t"]}"#;
        let pages = Qa::new().cut(jsonl).await.unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].questions, vec!["q1".to_string()]);
        assert_eq!(pages[0].content, "a1");
        assert_eq!(pages[0].keywords, vec!["k".to_string()]);
        assert_eq!(pages[1].tags, vec!["t".to_string()]);
        assert!(!pages[0].spans.is_empty());
        assert!(pages[0].spans[0].end > pages[0].spans[0].start);
    }

    #[tokio::test]
    async fn malformed_second_line() {
        let jsonl = "{\"question\":\"q\",\"answer\":\"a\"}\nnot-json";
        let err = Qa::new().cut(jsonl).await.unwrap_err();
        match err {
            Error::QaParse { line, .. } => assert_eq!(line, 2),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn blank_lines_skipped() {
        let jsonl = "\n{\"question\":\"q\",\"answer\":\"a\"}\n\n";
        let pages = Qa::new().cut(jsonl).await.unwrap();
        assert_eq!(pages.len(), 1);
    }
}
