//! 内存队列 [`Inline`]。

use std::collections::VecDeque;

use crate::index::source::base::Base;
use crate::index::source::types::{Error, Item, Scenario};

/// 内存 FIFO 队列。
///
/// 非持久化，主要用于测试和 demo。
#[derive(Default)]
pub struct Inline {
    items: VecDeque<Item>,
}

impl Inline {
    /// 新建一个空队列。
    pub fn new() -> Self {
        Self {
            items: VecDeque::new(),
        }
    }

    /// 把单个 [`Item`] 追加到队尾。
    pub async fn push(&mut self, item: Item) {
        self.items.push_back(item);
    }

    /// 批量把多个 [`Item`] 追加到队尾。
    pub async fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = Item> + Send,
    {
        self.items.extend(items);
    }

    /// 当前队列中已缓冲的 Item 数（主要用于测试断言）。
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// 队列是否为空。
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Base for Inline {
    async fn read(&mut self, batch_size: usize, scenario: Scenario) -> Result<Vec<Item>, Error> {
        let _span = tracing::debug_span!("source.inline.read", batch_size, ?scenario).entered();

        let take = batch_size.min(self.items.len());
        if take == 0 {
            tracing::debug!("source.read.eos");
            return Ok(Vec::new());
        }

        // 先把"将要被消费"的前 take 个 Item 全部校验一遍，校验通过后再 pop；
        // 如此一来任何一个 Item 失败都不会把队列消费掉一半。
        for item in self.items.iter().take(take) {
            if let Err(e) = self.validate_doc_id(&item.metadata).await {
                tracing::warn!(error = %e, "source.read.validate_doc_id failed");
                return Err(e);
            }
        }

        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            let mut item = self.items.pop_front().expect("just validated len");
            item.scenario = scenario;
            out.push(item);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, Value, json};

    fn item(doc_id: &str, text: &str) -> Item {
        let mut md = Map::new();
        md.insert("doc_id".into(), json!(doc_id));
        Item {
            text: text.into(),
            scenario: Scenario::Qa,
            metadata: md,
        }
    }

    fn item_without_doc_id(text: &str) -> Item {
        Item {
            text: text.into(),
            scenario: Scenario::Qa,
            metadata: Map::new(),
        }
    }

    #[tokio::test]
    async fn push_then_read_preserves_order_and_overwrites_scenario() {
        let mut s = Inline::new();
        s.push(item("a", "AAA")).await;
        s.push(item("b", "BBB")).await;

        let got = s.read(10, Scenario::General).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].text, "AAA");
        assert_eq!(got[0].scenario, Scenario::General);
        assert_eq!(got[1].text, "BBB");
        assert_eq!(got[1].scenario, Scenario::General);
    }

    #[tokio::test]
    async fn multi_read_slices_then_returns_empty() {
        let mut s = Inline::new();
        for i in 0..5 {
            s.push(item(&format!("doc_{i}"), &format!("t{i}"))).await;
        }
        let a = s.read(2, Scenario::General).await.unwrap();
        let b = s.read(2, Scenario::General).await.unwrap();
        let c = s.read(2, Scenario::General).await.unwrap();
        let d = s.read(2, Scenario::General).await.unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(c.len(), 1);
        assert!(d.is_empty());
    }

    #[tokio::test]
    async fn batch_size_zero_is_noop() {
        let mut s = Inline::new();
        s.push(item("a", "x")).await;
        let got = s.read(0, Scenario::General).await.unwrap();
        assert!(got.is_empty());
        assert_eq!(s.len(), 1, "batch_size=0 时不应消费任何 Item");
    }

    #[tokio::test]
    async fn empty_queue_returns_empty_vec() {
        let mut s = Inline::new();
        let got = s.read(5, Scenario::General).await.unwrap();
        assert!(got.is_empty());
    }

    #[tokio::test]
    async fn second_item_missing_doc_id_fails_whole_batch_and_keeps_queue() {
        let mut s = Inline::new();
        s.push(item("a", "AAA")).await;
        s.push(item_without_doc_id("BBB")).await;
        s.push(item("c", "CCC")).await;

        let err = s.read(3, Scenario::General).await.unwrap_err();
        assert!(matches!(err, Error::MissingDocId));
        // 队列原封不动，没有被 pop 掉任何 Item。
        assert_eq!(s.len(), 3);
    }

    #[tokio::test]
    async fn extend_accepts_many_items() {
        let mut s = Inline::new();
        s.extend(vec![item("a", "x"), item("b", "y")]).await;
        assert_eq!(s.len(), 2);
        let got = s.read(10, Scenario::Manual).await.unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|i| i.scenario == Scenario::Manual));
    }

    #[tokio::test]
    async fn non_string_doc_id_fails() {
        let mut s = Inline::new();
        let mut bad = item("a", "x");
        bad.metadata.insert("doc_id".into(), Value::from(42i64));
        s.push(bad).await;
        let err = s.read(1, Scenario::General).await.unwrap_err();
        assert!(matches!(err, Error::MissingDocId));
    }
}
