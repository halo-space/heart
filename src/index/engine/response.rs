//! `engine::response`：把分页后的 hits 组装成 [`Response`]，附带 `doc_aggs`。

use super::types::{DocAgg, Hit, Response};

/// 把分页后的 hits 与 filter 后的总数组装成 [`Response`]。
///
/// `total` 由调用方传入（spec：等于 `filter` 后 `paginate` 前的 hits 总数）。
/// `doc_aggs` 按 `page_hits[*].pagewiki.doc_id`（缺省取空串）group_by count，保持首次出现顺序。
pub fn build_response(total: usize, page_hits: Vec<Hit>) -> Response {
    let mut order: Vec<String> = Vec::new();
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for h in &page_hits {
        let doc_id = h.pagewiki.doc_id.clone().unwrap_or_default();
        if !counts.contains_key(&doc_id) {
            order.push(doc_id.clone());
        }
        *counts.entry(doc_id).or_insert(0) += 1;
    }
    let doc_aggs = order
        .into_iter()
        .map(|doc_id| {
            let count = counts.get(&doc_id).copied().unwrap_or(0);
            DocAgg { doc_id, count }
        })
        .collect();
    Response {
        total,
        hits: page_hits,
        doc_aggs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::pagewiki::PageWiki;
    use std::collections::HashMap;

    fn h(doc: Option<&str>) -> Hit {
        Hit {
            pagewiki: PageWiki {
                doc_id: doc.map(String::from),
                ..Default::default()
            },
            score: 0.0,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[test]
    fn doc_aggs_count_in_first_seen_order() {
        let hits = vec![
            h(Some("A")),
            h(Some("A")),
            h(Some("B")),
            h(Some("A")),
            h(Some("C")),
        ];
        let r = build_response(5, hits);
        assert_eq!(
            r.doc_aggs,
            vec![
                DocAgg {
                    doc_id: "A".into(),
                    count: 3
                },
                DocAgg {
                    doc_id: "B".into(),
                    count: 1
                },
                DocAgg {
                    doc_id: "C".into(),
                    count: 1
                }
            ]
        );
    }

    #[test]
    fn doc_id_none_aggregated_under_empty_string() {
        let hits = vec![h(None), h(Some("A"))];
        let r = build_response(2, hits);
        assert_eq!(r.doc_aggs.len(), 2);
        assert_eq!(r.doc_aggs[0].doc_id, "");
        assert_eq!(r.doc_aggs[0].count, 1);
        assert_eq!(r.doc_aggs[1].doc_id, "A");
    }

    #[test]
    fn total_is_independent_of_page_hits_len() {
        let r = build_response(80, vec![h(Some("A"))]);
        assert_eq!(r.total, 80);
        assert_eq!(r.hits.len(), 1);
    }
}
