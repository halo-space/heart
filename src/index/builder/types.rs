//! 构建层核心类型：[`Error`] 与 [`Builder`]。

use crate::index::builder::embed::Embedder;
use crate::index::builder::tokenize::Tokenizer;
use crate::index::pagewiki;
use crate::index::source;
use std::collections::HashMap;

/// 构建层错误类型。
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("no pagewiki impl registered for scenario {0:?}")]
    MissingScenario(source::Scenario),

    #[error("metadata.doc_id missing or unusable")]
    MissingDocId,

    #[error("forbidden field in metadata_promote_fields: {0}")]
    InvalidPromoteField(String),

    #[error("promote deserialize failed for field `{field}`: {reason}")]
    PromoteDeserialize { field: String, reason: String },

    #[error("source error: {0}")]
    Source(#[from] source::Error),

    #[error("pagewiki error: {0}")]
    PageWiki(#[from] pagewiki::Error),

    #[error("tokenize error: {0}")]
    Tokenize(String),

    #[error("embed error: {0}")]
    Embed(String),
}

/// 构建层主结构体。
///
/// 通过依赖注入接收切分实现、分词器、嵌入服务，把 `source::Item`
/// 跑通完整流水线，输出可入库的 `Vec<pagewiki::PageWiki>`。
pub struct Builder {
    pub(crate) pagewikis: HashMap<source::Scenario, Box<dyn pagewiki::Base>>,
    pub(crate) metadata_promote_fields: Vec<String>,
    pub(crate) tokenizer: Box<dyn Tokenizer>,
    pub(crate) embedder: Option<Box<dyn Embedder>>,
}

impl std::fmt::Debug for Builder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Builder")
            .field(
                "pagewikis",
                &format!("<{} scenarios>", self.pagewikis.len()),
            )
            .field("metadata_promote_fields", &self.metadata_promote_fields)
            .field("tokenizer", &"<dyn Tokenizer>")
            .field(
                "embedder",
                &if self.embedder.is_some() {
                    "Some(<dyn Embedder>)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}
