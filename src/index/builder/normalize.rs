//! 文本规范化。

use std::sync::OnceLock;
use regex::Regex;

static TAG_RE: OnceLock<Regex> = OnceLock::new();

fn tag_re() -> &'static Regex {
    TAG_RE.get_or_init(|| Regex::new(r"<[^>]*>").expect("正则表达式有效"))
}

/// 对原始文本做索引前规范化。
///
/// 处理顺序：去 BOM → 统一换行符 → 删除 HTML/XML 标签 →
/// 解码 HTML 实体（`&amp;` 最后处理）→ 去首尾空白。
pub fn normalize_text(text: &str) -> String {
    let input_len = text.len();

    // 1. 去掉 UTF-8 BOM
    let s = text.strip_prefix('\u{FEFF}').unwrap_or(text);

    // 2. 统一换行符：\r\n → \n，再把单独的 \r → \n
    let s = s.replace("\r\n", "\n").replace('\r', "\n");

    // 3. 删除 HTML/XML 标签
    let s = tag_re().replace_all(&s, "").into_owned();

    // 4. 解码 HTML 实体（&amp; 最后处理，避免二次替换）
    let s = s
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&apos;", "'")
        .replace("&quot;", "\"")
        .replace("&amp;", "&");

    // 5. 去掉首尾空白
    let s = s.trim().to_string();

    let output_len = s.len();
    tracing::trace!(input_len, output_len, "index.normalize.done");

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 去掉_bom() {
        assert_eq!(normalize_text("\u{FEFF}hello"), "hello");
    }

    #[test]
    fn 统一换行符() {
        assert_eq!(normalize_text("a\r\nb\rc"), "a\nb\nc");
    }

    #[test]
    fn 删除_html_标签() {
        assert_eq!(normalize_text("<p>hello <b>world</b></p>"), "hello world");
    }

    #[test]
    fn 解码_html_实体() {
        assert_eq!(
            normalize_text("&nbsp;Tom &amp; Jerry &lt;3&gt;&apos;s"),
            "Tom & Jerry <3>'s"
        );
    }

    #[test]
    fn 保留空行() {
        assert_eq!(normalize_text("a\n\n\nb"), "a\n\n\nb");
    }
}
