//! `engine::filter`：score 阈值过滤 + top 截断。

use super::types::Hit;

/// 按 score 倒序排序、按阈值过滤、最后取前 `top` 条。
///
/// 行为（与 spec 严格一致）：
///
/// 1. 先按 `score` 倒序排序（防御性：rerank / fusion 后可能已倒序，再排一次保证）。
/// 2. `disable_score_threshold == false` → 仅保留 `score >= score_threshold`；
///    `true` → 跳过阈值过滤。
/// 3. 取前 `top` 条返回；`hits.len() < top` 时返回全部。
pub fn filter_hits(
    mut hits: Vec<Hit>,
    score_threshold: f32,
    disable_score_threshold: bool,
    top: usize,
) -> Vec<Hit> {
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if !disable_score_threshold {
        hits.retain(|h| h.score >= score_threshold);
    }
    if hits.len() > top {
        hits.truncate(top);
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::pagewiki::PageWiki;
    use std::collections::HashMap;

    fn hit(id: &str, score: f32) -> Hit {
        Hit {
            pagewiki: PageWiki {
                id: Some(id.into()),
                ..Default::default()
            },
            score,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[test]
    fn threshold_filters_below_cutoff() {
        let hits = (0..10)
            .map(|i| hit(&format!("h{i}"), 0.1 * (i as f32 + 1.0)))
            .collect::<Vec<_>>();
        let out = filter_hits(hits, 0.5, false, 100);
        assert!(out.iter().all(|h| h.score >= 0.5));
    }

    #[test]
    fn top_truncates_after_threshold() {
        let hits = (0..10)
            .map(|i| hit(&format!("h{i}"), 0.5 + 0.05 * i as f32))
            .collect::<Vec<_>>();
        let out = filter_hits(hits, 0.5, false, 3);
        assert_eq!(out.len(), 3);
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn disable_threshold_skips_filter() {
        let hits = (0..10)
            .map(|i| hit(&format!("h{i}"), 0.1 * i as f32))
            .collect::<Vec<_>>();
        let out = filter_hits(hits, 0.9, true, 100);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn fewer_than_top_returns_all() {
        let hits = (0..3)
            .map(|i| hit(&format!("h{i}"), 0.6 + 0.1 * i as f32))
            .collect::<Vec<_>>();
        let out = filter_hits(hits, 0.5, false, 100);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn sort_is_defensive() {
        let mut h = vec![hit("a", 0.3), hit("b", 0.9), hit("c", 0.6)];
        // 故意打乱
        h.reverse();
        let out = filter_hits(h, 0.0, false, 10);
        assert_eq!(out[0].score, 0.9);
        assert_eq!(out[1].score, 0.6);
        assert_eq!(out[2].score, 0.3);
    }
}
