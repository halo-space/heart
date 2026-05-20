//! 通用时间戳工具。

/// 返回当前 UTC 毫秒级时间戳字符串。
pub fn current_millis_string() -> String {
    chrono::Utc::now().timestamp_millis().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn millis_string_非空且为数字() {
        let s = current_millis_string();
        assert!(!s.is_empty());
        assert!(s.parse::<i64>().is_ok(), "非数字: {s}");
    }
}
