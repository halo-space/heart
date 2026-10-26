# Document parsing

[Usage guide](USAGE.en.md) · [中文](DOCUMENT.zh-CN.md)

Heart's `document::Loader` reads raw bytes, and `document::Parser` returns a collection of `Document` values. Each Parser implementation declares its supported format identifiers through the read-only associated constant `FORMATS`. Callers can inspect it to select a Parser, but cannot change the formats declared by an implementation. The caller selects the Loader and may add `doc_id` and `metadata.source`. The Parser does not allocate IDs, create Checkpoints, or introduce a special Workflow Node.

## Default implementation

- `text`, `markdown`, `csv`, and `json` keep their existing Heart parsers.
- `pdf`, `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp`, `png`, `jpg`/`jpeg`, and `tif`/`tiff` use [LiteParse](https://github.com/run-llama/liteparse) by default. Their `engine` can be omitted or explicitly set to `"liteparse"`.
- `general` returns one `Document` per page with the source page number in `metadata.location.page.from/to`; `one` returns one combined `Document`. Both leave `id` and `doc_id` for the application to assign.
- Markdown output is the default. The default build reads existing text layers without OCR. `options` accepts `output: "text"`, `ocr`, `ocr_language`, `ocr_server_url`, `max_pages`, `target_pages`, and the existing `trim`, `normalize_newlines`, and `remove_empty` settings.

LiteParse converts Office/OpenDocument files to PDF and needs LibreOffice installed in the execution environment. PDF input is parsed directly from bytes. Office and image inputs use a temporary file that is removed after parsing, preserving the suffix LiteParse needs for format detection. LiteParse downloads its native PDFium dependency during the first build unless it is provided as described upstream. Scanned PDFs and image-only inputs require OCR: enable the Cargo feature `ocr` for LiteParse's Tesseract support, or set `options.ocr_server_url`. The Tesseract feature downloads and builds native dependencies at compile time; other languages also need their language data. `cargo install liteparse` installs the standalone `lit` CLI. Heart library users do not need that CLI.

## Runnable example

```bash
cargo run --example document_liteparse -- ./document.pdf
cargo run --example document_liteparse -- ./slides.pptx
```

Or use the common API in an application:

```rust
use halo::agent::{Cancellation, document::{self, Loader, Parser}};
use serde_json::json;

let cancellation = Cancellation::new();
let source = document::Source { uri: "./document.pdf".into() };
let bytes = document::LocalLoader.load(&source, &cancellation).await?;
let parser = document::parser::new("general", json!({
    "format": "pdf",
    "options": {"max_pages": 1000}
}))?;
let mut pages = parser.parse(&bytes, &cancellation).await?;
for page in &mut pages {
    page.metadata.insert("source".into(), json!({"uri": source.uri}));
}
```

The Parser returns a complete collection or an error, never a partial success collection. Only page-level `general` results carry page positions; the combined `one` result does not claim a single-page location. Cancellation is checked before and after the call; an individual LiteParse parse already in progress is not forcibly interrupted.

To add another OCR engine or custom parsing logic, implement the existing `document::Parser` contract directly. Declare the read-only `FORMATS` constant; the parser receives source bytes as `&[u8]` plus `&Cancellation` and returns `Result<Vec<Document>, document::Error>`. The implementation owns format detection, OCR, and Document construction; the framework does not introduce a separate OCR input/output contract or require extending LiteParse.
