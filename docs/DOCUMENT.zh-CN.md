# Document 解析

[返回使用手册](USAGE.zh-CN.md) · [English](DOCUMENT.en.md)

Heart 的 `document::Loader` 读取原始字节，`document::Parser` 将字节转成统一的 `Document` 集合。每个 Parser 实现通过只读关联常量 `FORMATS` 声明支持的格式标识；调用方可据此选择 Parser，不能在运行时修改实现声明的格式。调用方选择 Loader，并在需要时补充 `doc_id` 和 `metadata.source`。Parser 不创建 ID、Checkpoint 或专用 Workflow Node。

## 默认实现

- `text`、`markdown`、`csv`、`json` 继续使用 Heart 的现有解析实现。
- `pdf`、`docx`、`xlsx`、`pptx`、`odt`、`ods`、`odp`、`png`、`jpg`/`jpeg`、`tif`/`tiff` 默认使用 [LiteParse](https://github.com/run-llama/liteparse)；这些格式可省略 `engine`，显式写时为 `"liteparse"`。
- `general` 每页返回一个 `Document`，`metadata.location.page.from/to` 都是原始页码；`one` 返回合并后的一个 `Document`。两种方法都保留 `id`、`doc_id` 为空，供应用分配。
- 默认输出 Markdown；默认构建解析原有文本层，不开启 OCR。可在 `options` 中设置 `output: "text"`、`ocr`、`ocr_language`、`ocr_server_url`、`max_pages`、`target_pages`，以及已有的 `trim`、`normalize_newlines`、`remove_empty`。

Office/OpenDocument 由 LiteParse 转为 PDF，需要运行环境安装 LibreOffice。PDF 可直接从内存字节解析；Office 和图片输入会写入自动清理的临时文件，以保留格式后缀供 LiteParse 识别。LiteParse 的 PDFium 依赖会在首次构建时从上游下载原生库，也可以按其上游说明预先提供。扫描 PDF 和纯图片需要 OCR：可启用 Cargo 特性 `ocr` 来构建 LiteParse 的 Tesseract 支持，或在 `options` 中提供 `ocr_server_url`。Tesseract 特性在构建时下载并编译原生依赖，其他语种还需要相应语言数据。`cargo install liteparse` 安装的是独立的 `lit` CLI；使用 Heart 库时无需单独安装 CLI。

## 运行示例

项目自带完整入口：

```bash
cargo run --example document_liteparse -- ./document.pdf
cargo run --example document_liteparse -- ./slides.pptx
```

也可以在应用中直接调用统一契约：

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

Parser 完整返回或返回错误，不输出部分成功集合。页级位置只在 `general` 的单页 `Document` 上写入；`one` 的合并文本不伪造单页位置。取消在调用前后检查；LiteParse 内部正在进行的单次解析不会被强制中止。

需要接入其他 OCR 或自定义解析逻辑时，直接实现现有的 `document::Parser` 契约：声明只读 `FORMATS`，接收原始 `&[u8]` 和 `&Cancellation`，返回 `Result<Vec<Document>, document::Error>`。实现自行完成格式识别、OCR 和 `Document` 组织；框架不另设 OCR 专用输入输出契约，也不要求扩展 LiteParse。
