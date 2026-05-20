//! `SlidingWindow`：带 overlap 的滑动窗口切分。
//!
//! 详见 `openspec/changes/rag-page-wiki/design.md` 第 6 节。

use crate::index::pagewiki::base::Base;
use crate::index::pagewiki::types::{Error, PageWiki, Span};
use std::future::Future;
use std::pin::Pin;

/// 滑动窗口切分器。窗口大小 `size`、重叠 `overlap`，均以字符计数。
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    /// 窗口字符长度。
    pub size: usize,
    /// 相邻窗口的重叠字符数，需小于 `size`。
    pub overlap: usize,
}

impl SlidingWindow {
    /// 构造 [`SlidingWindow`]。`size == 0` 或 `overlap >= size` 时返回
    /// [`Error::InvalidInput`]。
    pub fn new(size: usize, overlap: usize) -> Result<Self, Error> {
        if size == 0 {
            return Err(Error::InvalidInput("size must be > 0".into()));
        }
        if overlap >= size {
            return Err(Error::InvalidInput("overlap must be < size".into()));
        }
        Ok(Self { size, overlap })
    }
}

impl Base for SlidingWindow {
    fn cut<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PageWiki>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let chars: Vec<char> = text.chars().collect();
            let mut out = Vec::new();
            if chars.is_empty() {
                return Ok(out);
            }
            let step = self.size - self.overlap;
            let mut start = 0usize;
            loop {
                let end = (start + self.size).min(chars.len());
                let content: String = chars[start..end].iter().collect();
                let span = Span {
                    start,
                    end,
                    original_text: content.clone(),
                    extra: Default::default(),
                };
                out.push(PageWiki {
                    content,
                    spans: vec![span],
                    ..Default::default()
                });
                if end >= chars.len() {
                    break;
                }
                start += step;
            }
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn size4_overlap1_over_abcdefghi() {
        let sw = SlidingWindow::new(4, 1).unwrap();
        let pages = sw.cut("abcdefghi").await.unwrap();
        // step = 3, 起点 0,3,6 → "abcd", "defg", "ghi"
        let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
        assert_eq!(contents, vec!["abcd", "defg", "ghi"]);
        assert_eq!(pages[1].spans[0].start, 3);
        assert_eq!(pages[1].spans[0].end, 7);
    }

    #[tokio::test]
    async fn tail_shorter_than_size_emitted() {
        let sw = SlidingWindow::new(5, 2).unwrap();
        let pages = sw.cut("abcdefg").await.unwrap();
        // step = 3, 起点 0,3 → "abcde", "defg"
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[1].content, "defg");
    }

    #[tokio::test]
    async fn empty_text_yields_no_pages() {
        let sw = SlidingWindow::new(4, 1).unwrap();
        assert!(sw.cut("").await.unwrap().is_empty());
    }

    #[test]
    fn zero_size_rejected() {
        assert!(matches!(
            SlidingWindow::new(0, 0).unwrap_err(),
            Error::InvalidInput(_)
        ));
    }

    #[test]
    fn equal_overlap_rejected() {
        assert!(matches!(
            SlidingWindow::new(4, 4).unwrap_err(),
            Error::InvalidInput(_)
        ));
    }
}
