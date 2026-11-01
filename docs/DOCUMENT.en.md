# Document parsing

[Usage guide](USAGE.en.md) · [中文](DOCUMENT.zh-CN.md)

Heart's `document::Loader` reads raw bytes, and `document::Parser` returns a collection of `Document` values. Each Parser implementation declares its supported input identifiers through the read-only associated constant `EXTS`. Callers can inspect it to select a Parser, but cannot change the input capability declared by an implementation. The caller selects the Loader and may add `doc_id` and `metadata.source`. The Parser does not allocate IDs, create Checkpoints, or introduce a special Workflow Node.

## Default implementation

- `text`, `markdown`, `csv`, and `json` keep their existing Heart parsers.
- `laws` supports text, Markdown, and document bytes readable by LiteParse. It preserves explicit chapter, section, and article boundaries while returning ordinary `Document` values.
- `paper`, `book`, and `manual` support text, Markdown, and document bytes readable by LiteParse. They preserve explicit headings and section boundaries.
- `presentation` supports PDF, PPT, and PPTX, returns one value per page, and marks `metadata.layout.role = "slide"`.
- `email` supports standard EML by extracting common headers, the body, and remaining header metadata; MSG/OLE2 returns `UNSUPPORTED`.
- `tag` supports CSV/JSON tag examples and writes the source text plus `metadata.tag.tags`; it does not infer tags or perform similarity matching.
- `resume` supports text, Markdown, and document bytes readable by LiteParse, preserving resume sections and page locations without model-based field inference.
- `picture` supports PNG/JPG/TIFF/GIF/BMP/WEBP/SVG image bytes through configured LiteParse OCR; image results do not invent page locations.
- `audio` requires a separate ASR/audio-model contract and currently returns `UNSUPPORTED`; raw audio bytes are never presented as text.
- `pdf`, `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp`, `png`, `jpg`/`jpeg`, and `tif`/`tiff` use [LiteParse](https://github.com/run-llama/liteparse) by default. Their `engine` can be omitted or explicitly set to `"liteparse"`.
- `general` returns one `Document` per page with the source page number in `metadata.location.page.from/to`; `one` returns one combined `Document`. Both leave `id` and `doc_id` for the application to assign.
- Markdown output is the default. The default build reads existing text layers without OCR. `options` accepts `output: "text"`, `ocr`, `ocr_language`, `ocr_server_url`, `max_pages`, `target_pages`, and the existing `trim`, `normalize_newlines`, and `remove_empty` settings.

## Output metadata contract

The Parser writes only metadata that the current parse can prove; it does not fill fields by guessing:

- `metadata.parser` identifies the implementation that actually ran, such as `"liteparse"`.
- `metadata.location` is present only when the parser can locate the content reliably. Page and paragraph ranges use `from` / `to`; unknown locations are omitted.
- `metadata.layout` is present only when the parser recognized a real layout role, such as `heading`, `paragraph`, `list`, `code`, `table`, or `row`.
- `metadata.method` is present only when the content provenance is known and is either `text` or `ocr`. Setting `ocr: true` alone does not mark every result as OCR; omit the field when the parser cannot identify provenance for each result.
- Structured fields such as `metadata.table` and `metadata.media` are written only when the current parse actually produced that data.
- `one` combines content into one `Document` and does not invent a single-page location, layout role, or OCR provenance.

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

To add another OCR engine or custom parsing logic, implement the existing `document::Parser` contract directly. Declare the read-only `EXTS` constant; the parser receives source bytes as `&[u8]` plus `&Cancellation` and returns `Result<Vec<Document>, document::Error>`. The implementation owns format detection, OCR, and Document construction; the framework does not introduce a separate OCR input/output contract or require extending LiteParse.
