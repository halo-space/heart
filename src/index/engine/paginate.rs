//! `engine::paginate`：1-based 分页切片；越界返回空 Vec。

use super::types::Hit;

/// 1-based 分页：`start = (page_num - 1) * page_size`。
///
/// 越界（`page_num == 0` 或 `start >= top_hits.len()`）返回 `Vec::new()`，**不**报错、**不** panic。
pub fn paginate_hits(top_hits: Vec<Hit>, page_num: usize, page_size: usize) -> Vec<Hit> {
    if page_num == 0 || page_size == 0 {
        return Vec::new();
    }
    let start = (page_num - 1) * page_size;
    if start >= top_hits.len() {
        return Vec::new();
    }
    let end = (start + page_size).min(top_hits.len());
    top_hits[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::pagewiki::PageWiki;
    use std::collections::HashMap;

    fn h(id: &str) -> Hit {
        Hit {
            pagewiki: PageWiki {
                id: Some(id.into()),
                ..Default::default()
            },
            score: 0.0,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[test]
    fn paginates_within_bounds() {
        let v = (0..25).map(|i| h(&format!("h{i}"))).collect::<Vec<_>>();
        let out = paginate_hits(v, 2, 10);
        assert_eq!(out.len(), 10);
        assert_eq!(out[0].pagewiki.id.as_deref(), Some("h10"));
        assert_eq!(out[9].pagewiki.id.as_deref(), Some("h19"));
    }

    #[test]
    fn last_page_partial_returns_remainder() {
        let v = (0..15).map(|i| h(&format!("h{i}"))).collect::<Vec<_>>();
        let out = paginate_hits(v, 2, 10);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn out_of_range_returns_empty() {
        let v = (0..5).map(|i| h(&format!("h{i}"))).collect::<Vec<_>>();
        let out = paginate_hits(v, 10, 10);
        assert!(out.is_empty());
    }

    #[test]
    fn page_num_zero_returns_empty() {
        let v = (0..5).map(|i| h(&format!("h{i}"))).collect::<Vec<_>>();
        let out = paginate_hits(v, 0, 10);
        assert!(out.is_empty());
    }

    #[test]
    fn page_size_zero_returns_empty() {
        let v = (0..5).map(|i| h(&format!("h{i}"))).collect::<Vec<_>>();
        let out = paginate_hits(v, 1, 0);
        assert!(out.is_empty());
    }
}
