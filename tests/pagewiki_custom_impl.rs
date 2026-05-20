//! 集成测试：自定义 `Base` 实现示例。
//!
//! 验证外部 crate 可以实现 `pagewiki::Base` trait，并正确返回 [`PageWiki`]。

use std::future::Future;
use std::pin::Pin;
use rag::index::pagewiki::{Base, Error, PageWiki, Span};

/// 按空格切分的简单切分器（演示自定义实现）。
struct WordSplitter;

impl Base for WordSplitter {
    fn cut<'a>(&'a self, text: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<PageWiki>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let mut out = Vec::new();
            let mut char_cursor = 0usize;
            for word in text.split(' ') {
                if word.is_empty() {
                    char_cursor += 1;
                    continue;
                }
                let word_char_len = word.chars().count();
                let span = Span {
                    start: char_cursor,
                    end: char_cursor + word_char_len,
                    original_text: word.to_string(),
                    extra: Default::default(),
                };
                out.push(PageWiki {
                    content: word.to_string(),
                    spans: vec![span],
                    ..Default::default()
                });
                char_cursor += word_char_len + 1; // +1 = 空格
            }
            Ok(out)
        })
    }
}

#[tokio::test]
async fn custom_word_splitter_basic() {
    let pages = WordSplitter.cut("hello world foo").await.unwrap();
    assert_eq!(pages.len(), 3);
    assert_eq!(pages[0].content, "hello");
    assert_eq!(pages[1].content, "world");
    assert_eq!(pages[2].content, "foo");
    // 字符坐标正确
    assert_eq!(pages[0].spans[0].start, 0);
    assert_eq!(pages[0].spans[0].end, 5);
    assert_eq!(pages[1].spans[0].start, 6);
    assert_eq!(pages[2].spans[0].start, 12);
}

#[tokio::test]
async fn custom_word_splitter_chinese() {
    let pages = WordSplitter.cut("你好 世界").await.unwrap();
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].content, "你好");
    assert_eq!(pages[1].content, "世界");
    // 中文字符 char index：你好 = 0..2，世界 = 3..5
    assert_eq!(pages[0].spans[0].start, 0);
    assert_eq!(pages[0].spans[0].end, 2);
    assert_eq!(pages[1].spans[0].start, 3);
    assert_eq!(pages[1].spans[0].end, 5);
}

#[tokio::test]
async fn custom_impl_option_fields_are_none() {
    let pages = WordSplitter.cut("abc").await.unwrap();
    assert_eq!(pages.len(), 1);
    assert!(pages[0].id.is_none());
    assert!(pages[0].doc_id.is_none());
    assert!(pages[0].version.is_none());
    assert!(pages[0].scenario.is_none());
    assert!(pages[0].idx.is_none());
    assert!(pages[0].embedding.is_none());
}
