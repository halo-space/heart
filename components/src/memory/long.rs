use futures_util::future::BoxFuture;
use schema::{
    Metadata,
    memory::Error,
    session::profile::{Profile, Type},
};

use crate::{Cancellation, model};

pub trait Long: Send + Sync {
    fn profiles(&self) -> &dyn Profiles;
}

pub trait Profiles: Send + Sync {
    fn create<'a>(
        &'a self,
        profile: Profile,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Profile, Error>>;

    fn read(
        &self,
        tenant_id: u64,
        agent_id: u64,
        user_id: u64,
        profile_type: Type,
        cancellation: Cancellation,
    ) -> BoxFuture<'_, Result<Profile, Error>>;

    fn update<'a>(
        &'a self,
        profile: Profile,
        version: u64,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Profile, Error>>;

    fn delete(
        &self,
        profile_id: u64,
        cancellation: Cancellation,
    ) -> BoxFuture<'_, Result<(), Error>>;

    #[allow(clippy::too_many_arguments)]
    fn compress<'a>(
        &'a self,
        tenant_id: u64,
        agent_id: u64,
        user_id: u64,
        profile_type: Type,
        session_id: u64,
        model: &'a dyn model::Chat,
        metadata: &'a Metadata,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Profile, Error>>;
}
