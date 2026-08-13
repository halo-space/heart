use futures_util::future::BoxFuture;
use schema::knowledge::{
    Documents, Error, Wikis,
    document::{self as schema_document, Read, Update},
    wiki::Wiki,
};
use serde_json::Value;

use crate::Cancellation;

pub trait Knowledge: Send + Sync {
    fn document(&self) -> &dyn Document;

    fn vector(&self) -> &dyn Vector;

    fn hybrid(&self) -> &dyn Hybrid;
}

pub trait Document: Send + Sync {
    fn create<'a>(
        &'a self,
        document: schema_document::Document,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<schema_document::Document, Error>>;

    fn read(
        &self,
        input: Read,
        cancellation: Cancellation,
    ) -> BoxFuture<'_, Result<schema_document::Document, Error>>;

    fn update<'a>(
        &'a self,
        input: Update,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<schema_document::Document, Error>>;

    fn delete(&self, id: u64, cancellation: Cancellation) -> BoxFuture<'_, Result<(), Error>>;

    fn search(
        &self,
        dsl: Value,
        cancellation: Cancellation,
    ) -> BoxFuture<'_, Result<Documents, Error>>;
}

pub trait Vector: Send + Sync {
    fn create<'a>(
        &'a self,
        wiki: Wiki,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Wiki, Error>>;

    fn batch_create<'a>(
        &'a self,
        wikis: Vec<Wiki>,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Vec<Wiki>, Error>>;

    fn read(&self, id: u64, cancellation: Cancellation) -> BoxFuture<'_, Result<Wiki, Error>>;

    fn update<'a>(
        &'a self,
        wiki: Wiki,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Wiki, Error>>;

    fn delete(&self, id: u64, cancellation: Cancellation) -> BoxFuture<'_, Result<(), Error>>;

    fn search(&self, dsl: Value, cancellation: Cancellation)
    -> BoxFuture<'_, Result<Wikis, Error>>;
}

pub trait Hybrid: Send + Sync {
    fn search(&self, dsl: Value, cancellation: Cancellation)
    -> BoxFuture<'_, Result<Wikis, Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knowledge_contract_is_object_safe() {
        fn accepts_knowledge(_: &dyn Knowledge) {}
        fn accepts_document(_: &dyn Document) {}
        fn accepts_vector(_: &dyn Vector) {}
        fn accepts_hybrid(_: &dyn Hybrid) {}

        let _ = accepts_knowledge;
        let _ = accepts_document;
        let _ = accepts_vector;
        let _ = accepts_hybrid;
    }
}
