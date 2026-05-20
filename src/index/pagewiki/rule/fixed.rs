//! 固定长度切分实现。

use std::future::Future;
use std::pin::Pin;

use crate::index::pagewiki::base::Base;
use crate::index::pagewiki::types::{Error, PageWiki, Span};

/// 按固定字符数切分文本。
pub struct Fixed {
    size: usize,
}

impl Fixed {
    /// 构造固定长度切分器，`size` 为每块最大字符数。
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Base for Fixed {
    fn cut<'a>(&'a self, text: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<PageWiki>, Error>> + Send + 'a>> {
        Box::pin(async move {
            if text.is_empty() {
                return Ok(vec![]);
            }
            let chars: Vec<char> = text.chars().collect();
            let mut pages = Vec::new();
            let mut start = 0usize;

            while start < chars.len() {
                let end = (start + self.size).min(chars.len());
                let chunk: String = chars[start..end].iter().collect();

                pages.push(PageWiki {
                    content: chunk.clone(),
                    spans: vec![Span {
                        start,
                        end,
                        original_text: chunk,
                        extra: Default::default(),
                    }],
                    ..Default::default()
                });

                start = end;
            }
            Ok(pages)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn 空文本返回空切片() {
        let f = Fixed::new(10);
        assert!(f.cut("").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn 短文本不切分() {
        let f = Fixed::new(100);
        let pages = f.cut("hello").await.unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].content, "hello");
    }

    #[tokio::test]
    async fn 超长文本切分() {
        let f = Fixed::new(3);
        let pages = f.cut("abcdef").await.unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].content, "abc");
        assert_eq!(pages[1].content, "def");
    }
}
