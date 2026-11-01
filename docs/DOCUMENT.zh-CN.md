# Document 解析

[返回使用手册](USAGE.zh-CN.md) · [English](DOCUMENT.en.md)

Heart 的 `document::Loader` 读取原始字节，`document::Parser` 将字节转成统一的 `Document` 集合。每个 Parser 实现通过只读关联常量 `EXTS` 声明支持的输入标识；调用方可据此选择 Parser，不能在运行时修改实现声明的输入能力。调用方选择 Loader，并在需要时补充 `doc_id` 和 `metadata.source`。Parser 不创建 ID、Checkpoint 或专用 Workflow Node。

## 默认实现

- `text`、`markdown`、`csv`、`json` 继续使用 Heart 的现有解析实现。
- `laws` 支持文本、Markdown 和 LiteParse 可读取的文档字节；按明确的章、节、条标题保留法条结构，输出仍是 `Document` 集合。
- `paper`、`book`、`manual` 支持文本、Markdown 和 LiteParse 可读取的文档字节；按明确的标题或章节标记保留 section 结构。
- `presentation` 支持 PDF、PPT 和 PPTX，按页面输出并标记 `metadata.layout.role = "slide"`。
- `email` 支持标准 EML，提取常用头部、正文和其他头部元数据；MSG/OLE2 仍返回 `UNSUPPORTED`。
- `tag` 支持 CSV/JSON 标签示例，输出原文和 `metadata.tag.tags`；不执行标签推理或相似度匹配。
- `resume` 支持文本、Markdown 和 LiteParse 可读取的文档字节，按简历章节保留原文和页码；不调用模型推断个人字段。
- `picture` 支持 PNG/JPG/TIFF/GIF/BMP/WEBP/SVG 图片字节，使用 LiteParse 已配置的 OCR 能力；图片结果不伪造页码。
- `audio` 需要独立 ASR/音频模型契约，当前返回 `UNSUPPORTED`，不会把音频字节伪造成文本。
- `pdf`、`docx`、`xlsx`、`pptx`、`odt`、`ods`、`odp`、`png`、`jpg`/`jpeg`、`tif`/`tiff` 默认使用 [LiteParse](https://github.com/run-llama/liteparse)；这些格式可省略 `engine`，显式写时为 `"liteparse"`。
- `general` 每页返回一个 `Document`，`metadata.location.page.from/to` 都是原始页码；`one` 返回合并后的一个 `Document`。两种方法都保留 `id`、`doc_id` 为空，供应用分配。
- 默认输出 Markdown；默认构建解析原有文本层，不开启 OCR。可在 `options` 中设置 `output: "text"`、`ocr`、`ocr_language`、`ocr_server_url`、`max_pages`、`target_pages`，以及已有的 `trim`、`normalize_newlines`、`remove_empty`。

## 输出元数据约定

Parser 只写入能够由本次解析可靠证明的元数据，不为了字段齐全猜测来源：

- `metadata.parser` 表示实际使用的解析实现，例如 LiteParse 输出 `"liteparse"`。
- `metadata.location` 只保存解析器能够准确定位的范围，页码和段落统一使用 `from` / `to`；无法定位时省略。
- `metadata.layout` 只在解析器识别出实际布局角色时写入，例如 `heading`、`paragraph`、`list`、`code`、`table` 或 `row`。
- `metadata.method` 只在内容来源已经明确时写入 `text` 或 `ocr`。仅配置 `ocr: true` 不自动把所有结果标成 OCR；解析器没有逐页来源信息时应省略该字段。
- `metadata.table`、`metadata.media` 等结构化字段只在本次解析确实产生对应数据时写入。
- `one` 将内容合并为一个 `Document`，不伪造单页位置、布局角色或 OCR 来源。

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

需要接入其他 OCR 或自定义解析逻辑时，直接实现现有的 `document::Parser` 契约：声明只读 `EXTS`，接收原始 `&[u8]` 和 `&Cancellation`，返回 `Result<Vec<Document>, document::Error>`。实现自行完成格式识别、OCR 和 `Document` 组织；框架不另设 OCR 专用输入输出契约，也不要求扩展 LiteParse。
