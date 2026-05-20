//! `Delimiter`：按字面分隔符切分文本。
//!
//! 详见 `openspec/changes/rag-page-wiki/design.md` 第 6 节。

use std::future::Future;
use std::pin::Pin;
use crate::index::pagewiki::base::Base;
use crate::index::pagewiki::types::{Error, PageWiki, Span};

/// 按字面分隔符切分器。
#[derive(Debug, Clone)]
pub struct Delimiter {
    /// 分隔符字符串（非空）。
    pub delimiter: String,
}

impl Delimiter {
    /// 构造 [`Delimiter`]。空分隔符返回 [`Error::InvalidInput`]。
    pub fn new(delimiter: impl Into<String>) -> Result<Self, Error> {
        let delimiter = delimiter.into();
        if delimiter.is_empty() {
            return Err(Error::InvalidInput("delimiter must not be empty".into()));
        }
        Ok(Self { delimiter })
    }
}

impl Base for Delimiter {
    fn cut<'a>(&'a self, text: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<PageWiki>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let delim_char_len = self.delimiter.chars().count();
            let mut out = Vec::new();
            let mut char_cursor = 0usize;
            let mut byte_cursor = 0usize;

            loop {
                // 在剩余文本中按字节查找分隔符。
                let rel = text[byte_cursor..].find(self.delimiter.as_str());
                let (seg_byte_end, next_byte_start) = match rel {
                    Some(off) => (byte_cursor + off, byte_cursor + off + self.delimiter.len()),
                    None => (text.len(), text.len() + 1),
                };
                let seg = &text[byte_cursor..seg_byte_end];
                let seg_char_len = seg.chars().count();

                // 跳过空 chunk。
                if !seg.is_empty() {
                    let content = seg.to_string();
                    let span = Span {
                        start: char_cursor,
                        end: char_cursor + seg_char_len,
                        original_text: content.clone(),
                        extra: Default::default(),
                    };
                    out.push(PageWiki {
                        content,
                        spans: vec![span],
                        ..Default::default()
                    });
                }

                char_cursor += seg_char_len + delim_char_len;

                if next_byte_start > text.len() {
                    break;
                }
                byte_cursor = next_byte_start;
            }
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn split_on_triple_dash() {
        let d = Delimiter::new("---").unwrap();
        let pages = d.cut("a---b------c").await.unwrap();
        let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
        // "a---b------c": a | b | (空) | c
        assert_eq!(contents, vec!["a", "b", "c"]);
        assert!(pages[0].spans[0].start < pages[1].spans[0].start);
        assert!(pages[1].spans[0].start < pages[2].spans[0].start);
    }

    #[tokio::test]
    async fn trailing_delimiter_no_empty_chunk() {
        let d = Delimiter::new("\n").unwrap();
        let pages = d.cut("a\nb\n").await.unwrap();
        let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
        assert_eq!(contents, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn chinese_delimiter() {
        let d = Delimiter::new("。").unwrap();
        let pages = d.cut("句子一。句子二。").await.unwrap();
        let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
        assert_eq!(contents, vec!["句子一", "句子二"]);
        // 字符下标：句子一(0..3)，句子二(4..7)
        assert_eq!(pages[0].spans[0].start, 0);
        assert_eq!(pages[0].spans[0].end, 3);
        assert_eq!(pages[1].spans[0].start, 4);
        assert_eq!(pages[1].spans[0].end, 7);
    }

    #[test]
    fn empty_delimiter_rejected() {
        let err = Delimiter::new("").unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn empty_text_yields_no_chunks() {
        let d = Delimiter::new("---").unwrap();
        assert!(d.cut("").await.unwrap().is_empty());
    }
}
