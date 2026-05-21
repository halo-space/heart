//! `engine::fusion`：Reciprocal Rank Fusion (RRF) 融合两路单路 hits。
//!
//! 单纯计算函数；无 IO、无 async；可独立单测。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::types::Hit;

/// 文本路 / 向量路融合权重；缺省 1.0 / 1.0。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionWeights {
    pub text: f32,
    pub vector: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            text: 1.0,
            vector: 1.0,
        }
    }
}

/// 把两路单路 hits 用 RRF 融合为一路。
///
/// 算法（与 spec 一致）：
///
/// 1. 按 `pagewiki.id` 唯一化两路（同 id 的 hit 文本路 / 向量路只占一份；以文本路 hit 为基础合并）。
/// 2. 文本路 hits 按入参顺序赋 `text_rank`（1-based），同时 `scores.text = 原 score`。
/// 3. 向量路同理 `vector_rank` / `scores.vector`。
/// 4. `rrf = weights.text/(rrf_k + text_rank) + weights.vector/(rrf_k + vector_rank)`；
///    缺失路贡献 0。
/// 5. 写 `scores.rrf = rrf` / `scores.stage = rrf` / `hit.score = rrf`；按 `score` 倒序返回。
///
/// `pagewiki.id == None` 视为程序约束错误：Builder 写入前一律分配 UUID v4，运行期到这里
/// 不应该再出现 `None`，所以用 `expect` 直接 panic。
pub fn fuse_by_rrf(
    text_hits: Vec<Hit>,
    vector_hits: Vec<Hit>,
    weights: &FusionWeights,
    rrf_k: u32,
) -> Vec<Hit> {
    let rrf_k_f = rrf_k as f32;

    // text 侧索引
    let mut by_id: HashMap<String, Hit> =
        HashMap::with_capacity(text_hits.len() + vector_hits.len());
    let mut order: Vec<String> = Vec::with_capacity(text_hits.len() + vector_hits.len());

    for (idx, mut hit) in text_hits.into_iter().enumerate() {
        let id = hit
            .pagewiki
            .id
            .clone()
            .expect("pagewiki.id must be Some at engine layer (Builder ensures UUID v4)");
        let rank = (idx + 1) as f32;
        let text_score = hit.score;
        hit.scores.insert("text".into(), text_score);
        hit.scores.insert("text_rank".into(), rank);
        by_id.insert(id.clone(), hit);
        order.push(id);
    }

    for (idx, hit) in vector_hits.into_iter().enumerate() {
        let id = hit
            .pagewiki
            .id
            .clone()
            .expect("pagewiki.id must be Some at engine layer (Builder ensures UUID v4)");
        let rank = (idx + 1) as f32;
        let vector_score = hit.score;
        if let Some(existing) = by_id.get_mut(&id) {
            existing.scores.insert("vector".into(), vector_score);
            existing.scores.insert("vector_rank".into(), rank);
        } else {
            let mut h = hit;
            h.scores.insert("vector".into(), vector_score);
            h.scores.insert("vector_rank".into(), rank);
            by_id.insert(id.clone(), h);
            order.push(id);
        }
    }

    let mut out: Vec<Hit> = Vec::with_capacity(order.len());
    for id in order {
        let mut h = by_id.remove(&id).expect("entry inserted above");
        let text_rank = h.scores.get("text_rank").copied();
        let vector_rank = h.scores.get("vector_rank").copied();
        let mut rrf = 0.0_f32;
        if let Some(r) = text_rank {
            rrf += weights.text / (rrf_k_f + r);
        }
        if let Some(r) = vector_rank {
            rrf += weights.vector / (rrf_k_f + r);
        }
        h.scores.insert("rrf".into(), rrf);
        h.scores.insert("stage".into(), rrf);
        h.score = rrf;
        out.push(h);
    }

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::pagewiki::PageWiki;

    fn hit(id: &str, score: f32) -> Hit {
        Hit {
            pagewiki: PageWiki {
                id: Some(id.to_string()),
                content: "x".into(),
                ..Default::default()
            },
            score,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[test]
    fn both_paths_full_overlap_three_each() {
        let t = vec![hit("a", 1.0), hit("b", 0.9), hit("c", 0.8)];
        let v = vec![hit("a", 0.95), hit("b", 0.92), hit("c", 0.7)];
        let rrf_k = 60;
        let out = fuse_by_rrf(t, v, &FusionWeights::default(), rrf_k);
        assert_eq!(out.len(), 3);
        for h in &out {
            assert!(h.scores.contains_key("rrf"));
            assert!(h.scores.contains_key("stage"));
            assert!(h.scores.contains_key("text"));
            assert!(h.scores.contains_key("vector"));
            assert!(h.scores.contains_key("text_rank"));
            assert!(h.scores.contains_key("vector_rank"));
            // 完全重合：text_rank == vector_rank（按入参顺序赋 rank）
            let tr = h.scores["text_rank"];
            let vr = h.scores["vector_rank"];
            let expected = 1.0 / (rrf_k as f32 + tr) + 1.0 / (rrf_k as f32 + vr);
            assert!((h.scores["rrf"] - expected).abs() < 1e-6);
            assert!((h.score - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn missing_route_contributes_zero() {
        let t = vec![hit("a", 1.0)];
        let v: Vec<Hit> = vec![];
        let out = fuse_by_rrf(t, v, &FusionWeights::default(), 60);
        assert_eq!(out.len(), 1);
        let h = &out[0];
        assert!(!h.scores.contains_key("vector"));
        assert!(!h.scores.contains_key("vector_rank"));
        let expected = 1.0 / (60.0 + 1.0);
        assert!((h.scores["rrf"] - expected).abs() < 1e-6);
    }

    #[test]
    fn output_sorted_desc_by_score() {
        let t = vec![hit("a", 1.0), hit("b", 0.9)];
        let v = vec![hit("c", 0.8)];
        let out = fuse_by_rrf(t, v, &FusionWeights::default(), 60);
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn disjoint_paths_dedup_correctly() {
        let t = vec![hit("a", 1.0)];
        let v = vec![hit("b", 0.5)];
        let out = fuse_by_rrf(t, v, &FusionWeights::default(), 60);
        let ids: Vec<&str> = out
            .iter()
            .map(|h| h.pagewiki.id.as_deref().unwrap())
            .collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn weights_apply() {
        let t = vec![hit("a", 1.0)];
        let v = vec![hit("a", 1.0)];
        let w = FusionWeights {
            text: 2.0,
            vector: 0.5,
        };
        let out = fuse_by_rrf(t, v, &w, 60);
        let expected = 2.0 / 61.0 + 0.5 / 61.0;
        assert!((out[0].scores["rrf"] - expected).abs() < 1e-6);
    }
}
