//! Offline Session conditional update; no model, Runtime or database service.
use halo::agent::{Cancellation, Error};
use halo::{
    core,
    memory::short::{Type, session},
    model::token::Usage,
};
use serde_json::json;
use session::Store as _;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let sessions = core::memory::short::session::InMemory::new();
    let cancellation = Cancellation::new();
    let original = sessions
        .create(
            session::Session {
                id: 101,
                tenant_id: 1,
                user_id: 2,
                r#type: Type::Agent,
                title: Some("天气与露营".into()),
                summary: None,
                usage: None,
                status: session::Status::Active,
                metadata: Default::default(),
                created_time: 0,
                updated_time: 0,
            },
            &cancellation,
        )
        .await?;

    let mut latest = original.clone();
    latest.summary = Some("用户想在天气晴朗时去露营。".into());
    latest.metadata.insert("from".into(), json!(201));
    latest.metadata.insert("to".into(), json!(205));
    latest.usage = Some(Usage {
        input: Some(80),
        output: Some(20),
        total: Some(100),
        ..Default::default()
    });
    let committed = sessions.update(latest, &cancellation).await?;

    // A late task still holds the previous snapshot. It must not overwrite
    // the newer summary. No retry or recompression happens inside the store.
    let mut stale = original;
    stale.title = Some("晚到的旧标题".into());
    let conflict = sessions.update(stale, &cancellation).await.unwrap_err();
    assert_eq!(conflict.code, "CONFLICT");
    assert_eq!(sessions.read(101, &cancellation).await?, committed);
    println!(
        "stale update: {}; current summary: {:?}",
        conflict.code, committed.summary
    );
    Ok(())
}
