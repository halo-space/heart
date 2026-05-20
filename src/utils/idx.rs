//! 通用 ID 生成工具。

/// 生成 UUID v4 字符串。
pub fn new_uuid_v4() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    #[test]
    fn uuid_v4_格式正确() {
        let re = Regex::new(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$").unwrap();
        let id = new_uuid_v4();
        assert!(re.is_match(&id), "uuid 格式错误: {id}");
    }
}
