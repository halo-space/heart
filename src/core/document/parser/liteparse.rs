use liteparse::{LiteParse, LiteParseConfig, OutputFormat, types::PdfInput};
use serde_json::{Map, Value, json};

use super::{Method, invalid};
use crate::{
    Cancellation,
    document::{Document, Error},
};

pub(super) fn validate(options: &Map<String, Value>) -> Result<(), Error> {
    if options.get("ocr").is_some_and(|value| !value.is_boolean()) {
        return Err(invalid("ocr", "ocr must be boolean"));
    }
    if options.get("max_pages").is_some_and(|value| {
        value
            .as_u64()
            .is_none_or(|pages| pages == 0 || usize::try_from(pages).is_err())
    }) {
        return Err(invalid("max_pages", "max_pages must be a positive integer"));
    }
    for key in ["target_pages", "ocr_language", "ocr_server_url"] {
        if options
            .get(key)
            .is_some_and(|value| value.as_str().is_none_or(|text| text.trim().is_empty()))
        {
            return Err(invalid(key, "option must be nonempty text"));
        }
    }
    if options
        .get("output")
        .is_some_and(|value| !matches!(value.as_str(), Some("markdown" | "text")))
    {
        return Err(invalid("output", "output must be markdown or text"));
    }
    let server = options.get("ocr_server_url");
    if options.get("ocr").and_then(Value::as_bool) == Some(false) && server.is_some() {
        return Err(invalid("ocr", "ocr_server_url requires OCR to be enabled"));
    }
    if options.get("ocr").and_then(Value::as_bool) == Some(true)
        && server.is_none()
        && !cfg!(feature = "ocr")
    {
        return Err(Error::new(
            "UNSUPPORTED",
            "OCR requires the ocr feature or ocr_server_url",
        ));
    }
    Ok(())
}

pub(super) async fn parse(
    data: &[u8],
    extension: &str,
    method: Method,
    options: &Map<String, Value>,
    cancellation: &Cancellation,
) -> Result<Vec<Document>, Error> {
    super::check_cancelled(cancellation)?;
    let output_format = match options.get("output").and_then(Value::as_str) {
        Some("text") => OutputFormat::Text,
        _ => OutputFormat::Markdown,
    };
    let config = LiteParseConfig {
        output_format,
        ocr_enabled: options
            .get("ocr")
            .and_then(Value::as_bool)
            .unwrap_or(cfg!(feature = "ocr") || options.contains_key("ocr_server_url")),
        ocr_server_url: options
            .get("ocr_server_url")
            .and_then(Value::as_str)
            .map(str::to_owned),
        ocr_language: options
            .get("ocr_language")
            .and_then(Value::as_str)
            .unwrap_or("eng")
            .to_owned(),
        max_pages: options
            .get("max_pages")
            .and_then(Value::as_u64)
            .map(|pages| pages as usize)
            .unwrap_or(1000),
        target_pages: options
            .get("target_pages")
            .and_then(Value::as_str)
            .map(str::to_owned),
        quiet: true,
        ..LiteParseConfig::default()
    };
    let parser = LiteParse::new(config);
    let result = if extension == "pdf" {
        parser.parse_input(PdfInput::Bytes(data.to_vec())).await
    } else {
        // LiteParse selects Office and image conversion from the path suffix.
        // Keep the owned temporary file alive until its async parse completes.
        let file = tempfile::Builder::new()
            .prefix("heart-document-")
            .suffix(&format!(".{extension}"))
            .tempfile()
            .map_err(|error| Error::new("IO_ERROR", error.to_string()).with_details("data"))?;
        std::fs::write(file.path(), data)
            .map_err(|error| Error::new("IO_ERROR", error.to_string()).with_details("data"))?;
        let path = file.path().to_str().ok_or_else(|| {
            Error::new("INVALID_ARGUMENTS", "temporary path is not UTF-8").with_details("data")
        })?;
        parser.parse(path).await
    }
    .map_err(|error| Error::new("PARSE_ERROR", error.to_string()).with_details("data"))?;
    super::check_cancelled(cancellation)?;

    let clean = |text: &str| {
        let text = if options
            .get("trim")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            text.trim()
        } else {
            text
        };
        if options
            .get("normalize_newlines")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            text.replace("\r\n", "\n").replace('\r', "\n")
        } else {
            text.to_owned()
        }
    };
    let mut documents = if matches!(method, Method::One) {
        vec![Document {
            content: clean(&result.text),
            metadata: Map::from_iter([("parser".into(), json!("liteparse"))]),
            ..Document::default()
        }]
    } else {
        result
            .pages
            .into_iter()
            .map(|page| {
                let content = if options.get("output").and_then(Value::as_str) == Some("text")
                    || page.markdown.is_empty()
                {
                    &page.text
                } else {
                    &page.markdown
                };
                let page_number = page.page_number as u64;
                Document {
                    content: clean(content),
                    metadata: Map::from_iter([
                        ("parser".into(), json!("liteparse")),
                        (
                            "location".into(),
                            json!({"page":{"from":page_number,"to":page_number}}),
                        ),
                    ]),
                    ..Document::default()
                }
            })
            .collect()
    };
    if options
        .get("remove_empty")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        documents.retain(|document| !document.content.trim().is_empty());
    }
    Ok(documents)
}
