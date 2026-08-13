use serde::{Deserialize, Serialize};

pub mod document;
pub mod wiki;

use document::Document;
use wiki::Wiki;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Error {
    InvalidInput,
    NotFound,
    Storage,
    Conflict,
    Timeout,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub wiki: Wiki,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Wikis {
    pub total: u64,
    pub hits: Vec<Hit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentHit {
    pub document: Document,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Documents {
    pub total: u64,
    pub hits: Vec<DocumentHit>,
}
