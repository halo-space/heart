//! 文本规范化。

use ferrous_opencc::{OpenCC, config::BuiltinConfig};
use regex::Regex;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

static TAG_RE: OnceLock<Regex> = OnceLock::new();
static ZERO_WIDTH_RE: OnceLock<Regex> = OnceLock::new();
static T2S: OnceLock<OpenCC> = OnceLock::new();

fn tag_re() -> &'static Regex {
    TAG_RE.get_or_init(|| Regex::new(r"<[^>]*>").expect("正则表达式有效"))
}

fn zero_width_re() -> &'static Regex {
    ZERO_WIDTH_RE.get_or_init(|| {
        // 零宽空格、零宽非连接符、零宽连接符、软连字符、零宽不换行空格(BOM)
        Regex::new(r"[\u{200B}\u{200C}\u{200D}\u{00AD}\u{FEFF}]").expect("正则表达式有效")
    })
}

fn t2s() -> &'static OpenCC {
    T2S.get_or_init(|| OpenCC::from_config(BuiltinConfig::T2s).expect("加载繁简转换配置失败"))
}

/// 对原始文本做索引前规范化。
///
/// 处理顺序：
/// 1. 去 UTF-8 BOM
/// 2. 统一换行符（`\r\n` / `\r` → `\n`）
/// 3. 删除零宽字符及软连字符
/// 4. 删除 HTML/XML 标签
/// 5. 解码 HTML 实体（`&amp;` 最后处理，避免二次替换）
/// 6. NFKC 规范化（全角→半角、Unicode 兼容等价分解后合成）
/// 7. Unicode 小写（`to_lowercase`）
/// 8. 繁体中文→简体中文
/// 9. 去首尾空白
pub fn normalize_text(text: &str) -> String {
    let input_len = text.len();

    // 1. 去掉 UTF-8 BOM（\u{FEFF} 已在零宽正则中处理，这里提前 strip_prefix 更高效）
    let s = text.strip_prefix('\u{FEFF}').unwrap_or(text);

    // 2. 统一换行符：\r\n → \n，再把单独的 \r → \n
    let s = s.replace("\r\n", "\n").replace('\r', "\n");

    // 3. 删除零宽字符及软连字符
    let s = zero_width_re().replace_all(&s, "").into_owned();

    // 4. 删除 HTML/XML 标签
    let s = tag_re().replace_all(&s, "").into_owned();

    // 5. 解码 HTML 实体（&amp; 最后处理，避免二次替换）
    let s = s
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&apos;", "'")
        .replace("&quot;", "\"")
        .replace("&amp;", "&");

    // 6. NFKC 规范化：全角字符→半角、Unicode 兼容等价分解后合成
    let s: String = s.nfkc().collect();

    // 7. Unicode 小写
    let s = s.to_lowercase();

    // 8. 繁体→简体
    let s = t2s().convert(&s);

    // 9. 去掉首尾空白
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
            "tom & jerry <3>'s"
        );
    }

    #[test]
    fn 保留空行() {
        assert_eq!(normalize_text("a\n\n\nb"), "a\n\n\nb");
    }

    #[test]
    fn 零宽字符删除() {
        // 零宽空格 U+200B、零宽非连接符 U+200C
        assert_eq!(
            normalize_text("hello\u{200B}world\u{200C}!"),
            "helloworld!"
        );
    }

    #[test]
    fn 全角转半角() {
        // NFKC：全角字母数字 → 半角
        assert_eq!(normalize_text("ＡＢＣ１２３"), "abc123");
    }

    #[test]
    fn 大写转小写() {
        assert_eq!(normalize_text("Hello WORLD"), "hello world");
    }

    #[test]
    fn 繁体转简体() {
        assert_eq!(normalize_text("開放中文轉換"), "开放中文转换");
    }

    #[test]
    fn 综合() {
        // 全角 + 繁体 + HTML 标签
        assert_eq!(normalize_text("<b>開放</b>Ｈｅｌｌｏ"), "开放hello");
    }
}
