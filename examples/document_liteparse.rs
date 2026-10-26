//! Parse a real local PDF, Office document, or image with LiteParse.
use halo::agent::{
    Cancellation,
    document::{self, Loader, Parser},
};
use serde_json::json;

fn main() -> Result<(), document::Error> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| document::Error::new("RUNTIME_ERROR", error.to_string()))?
        .block_on(run())
}

async fn run() -> Result<(), document::Error> {
    let path = std::env::args().nth(1).ok_or_else(|| {
        document::Error::new("INVALID_ARGUMENTS", "provide a local document path")
    })?;
    let format = std::path::Path::new(&path)
        .extension()
        .and_then(|extension| extension.to_str())
        .ok_or_else(|| document::Error::new("INVALID_ARGUMENTS", "file extension is required"))?
        .to_ascii_lowercase();
    let cancellation = Cancellation::new();
    let data = document::LocalLoader
        .load(&document::Source { uri: path.clone() }, &cancellation)
        .await?;
    let parser = document::parser::new("general", json!({"format": format}))?;
    let mut documents = parser.parse(&data, &cancellation).await?;
    for document in &mut documents {
        document
            .metadata
            .insert("source".into(), json!({"uri": path}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&documents)
            .map_err(|error| { document::Error::new("INVALID_RESPONSE", error.to_string()) })?
    );
    Ok(())
}
