//! 集成测试：Fixed / SlidingWindow / Delimiter 在中英文混排上的字符级正确性。

use rag::index::pagewiki::{Base, Delimiter, Fixed, SlidingWindow};

// ── Fixed ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fixed_ascii_three_chunks() {
    let pages = Fixed::new(3).cut("abcdefg").await.unwrap();
    let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
    assert_eq!(contents, vec!["abc", "def", "g"]);
    // 字符下标
    assert_eq!(pages[0].spans[0].start, 0);
    assert_eq!(pages[0].spans[0].end, 3);
    assert_eq!(pages[1].spans[0].start, 3);
    assert_eq!(pages[2].spans[0].start, 6);
}

#[tokio::test]
async fn fixed_chinese_char_index() {
    let pages = Fixed::new(2).cut("你好世界你好").await.unwrap();
    let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
    assert_eq!(contents, vec!["你好", "世界", "你好"]);
    // 每块恰好 2 个 Unicode 字符
    assert_eq!(pages[1].spans[0].start, 2);
    assert_eq!(pages[1].spans[0].end, 4);
}

#[tokio::test]
async fn fixed_mixed_cjk_ascii() {
    // "你好world" = 7 个字符，size=3 → ["你好w", "orl", "d"]
    let pages = Fixed::new(3).cut("你好world").await.unwrap();
    let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
    assert_eq!(contents, vec!["你好w", "orl", "d"]);
    assert_eq!(pages[0].spans[0].start, 0);
    assert_eq!(pages[0].spans[0].end, 3);
}

#[tokio::test]
async fn fixed_option_fields_are_none() {
    let pages = Fixed::new(4).cut("abcdef").await.unwrap();
    for p in &pages {
        assert!(p.id.is_none());
        assert!(p.doc_id.is_none());
        assert!(p.version.is_none());
        assert!(p.scenario.is_none());
        assert!(p.idx.is_none());
        assert!(p.embedding.is_none());
    }
}

// ── SlidingWindow ──────────────────────────────────────────────────────────

#[tokio::test]
async fn sliding_window_ascii_step3() {
    let sw = SlidingWindow::new(5, 2).unwrap();
    let pages = sw.cut("abcdefghij").await.unwrap();
    let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
    assert_eq!(contents, vec!["abcde", "defgh", "ghij"]);
    assert_eq!(pages[1].spans[0].start, 3);
    assert_eq!(pages[1].spans[0].end, 8);
}

#[tokio::test]
async fn sliding_window_chinese_char_level() {
    // "零一二三四五六七八九" 10 字，size=3 overlap=1 step=2
    let sw = SlidingWindow::new(3, 1).unwrap();
    let pages = sw.cut("零一二三四五六七八九").await.unwrap();
    let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
    // 起点 0,2,4,6,8
    assert_eq!(contents[0], "零一二");
    assert_eq!(contents[1], "二三四");
    assert_eq!(contents[4], "八九");
}

// ── Delimiter ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn delimiter_triple_dash() {
    let d = Delimiter::new("---").unwrap();
    let pages = d.cut("a---b------c").await.unwrap();
    let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
    assert_eq!(contents, vec!["a", "b", "c"]);
    assert!(pages[0].spans[0].start < pages[1].spans[0].start);
    assert!(pages[1].spans[0].start < pages[2].spans[0].start);
}

#[tokio::test]
async fn delimiter_chinese() {
    let d = Delimiter::new("。").unwrap();
    let pages = d.cut("句子一。句子二。").await.unwrap();
    let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
    assert_eq!(contents, vec!["句子一", "句子二"]);
    assert_eq!(pages[0].spans[0].start, 0);
    assert_eq!(pages[0].spans[0].end, 3);
    assert_eq!(pages[1].spans[0].start, 4);
}

#[tokio::test]
async fn delimiter_mixed_char_index() {
    // "你好|world|!" delimiter="|"
    let d = Delimiter::new("|").unwrap();
    let pages = d.cut("你好|world|!").await.unwrap();
    let contents: Vec<_> = pages.iter().map(|p| p.content.as_str()).collect();
    assert_eq!(contents, vec!["你好", "world", "!"]);
    // "你好" = 2 chars, start=0 end=2
    assert_eq!(pages[0].spans[0].end, 2);
    // "world" start=3 (2 chars + 1 delimiter)
    assert_eq!(pages[1].spans[0].start, 3);
}
