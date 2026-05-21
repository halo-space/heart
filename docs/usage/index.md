# Index 功能使用说明

## 1. 功能定位

`index` 模块负责把原始文档转换成可检索的 chunk 数据并写入后端索引。它包含三个子层：

```text
Builder（构建层）：
  文本提取 → 格式解析 → 规范化 → 切分 → tokenize → embedding → 写入 Store。

Normalizer（规范化）：
  对原始文本做索引前清洗，保证后端索引数据格式统一。

Tokenizer（分词）：
  把 content / keywords / questions 切成用于检索的 token 序列。
```

## 2. 文本规范化

`normalize_text()` 是 Builder 在切分之前对原始文本做的预处理。

### 2.1 处理链路

```text
1. 去掉 UTF-8 BOM（\u{FEFF}）
2. 统一换行符：\r\n / \r → \n
3. 删除零宽字符及软连字符
   （U+200B 零宽空格、U+200C 零宽非连接符、U+200D 零宽连接符、U+00AD 软连字符、U+FEFF 零宽不换行空格）
4. 删除 HTML/XML 标签（<p>、<b>、<img ...> 等）
5. 解码常见 HTML 实体（&nbsp; &lt; &gt; &apos; &quot; &amp;）
6. NFKC Unicode 规范化（全角字母数字 → 半角，兼容等价分解 + 规范合成）
7. 英文转小写（to_lowercase，支持 Unicode）
8. 繁体中文 → 简体中文（ferrous-opencc，纯 Rust 实现，t2s 规则集）
9. 去掉首尾空白
```

规范化只在**入库阶段**运行，目的是让索引数据的表示形式统一：全角和半角、大写和小写、繁体和简体在索引里会落到同一个 token。

查询侧（QueryEngine）做了同等处理（见 Query 处理链路），所以查询词和索引词的规范化方式是对齐的。

### 2.2 直接调用

通常不需要手动调用，`DefaultBuilder` 在 `build()` / `index()` 内部自动调用。如果需要单独使用：

```rust
use rag::index::builder::normalize::normalize_text;

let cleaned = normalize_text("  ＨＥＬＬＯworld\u{200B}！<b>測試</b>  ");
// → "helloworld！测试"
```

### 2.3 依赖说明

```text
unicode-normalization（0.1.25）：
  纯 Rust，NFKC 规范化，处理全角→半角及 Unicode 等价分解。

ferrous-opencc（0.4.0，feature = t2s-conversion）：
  纯 Rust 的 OpenCC 实现（Apache-2.0），无 C/C++ 依赖，跨平台部署无障碍。
  只启用 t2s-conversion feature，不加载反方向字典。
```

## 3. 构建并写入 Chunks

### 3.1 初始化 Builder

```rust
use std::sync::Arc;
use rag::{DefaultBuilder, Elastic};

let store = Arc::new(Elastic::new("http://127.0.0.1:9200")?);

let builder = DefaultBuilder::new(Some(store.clone()), None, "chunks")
    .with_chunking(ChunkerKind::Fixed, 800, 100, None)
    .with_keyword_top(3);
```

参数说明：

```text
store：
  可选。传 None 时只构建数据对象，不写入后端。

embedder：
  可选（第二个参数）。传 None 时 chunk.embedding = None。

index_name：
  必填。chunk 写入的后端索引名。
```

### 3.2 支持的内容格式

```text
ContentFormat::Text：
  普通纯文本。

ContentFormat::Qa：
  QA 对格式（question\tanswer 或类似结构）。
```

### 3.3 支持的切分策略

```text
ChunkerKind::Fixed（chunk_size, chunk_overlap）：
  固定大小切分。chunk_overlap = 0 时不重叠，> 0 时滑动窗口。

ChunkerKind::Delimiter：
  按指定分隔符聚合切分，适合结构化段落文本。

ChunkerKind::Semantic：
  第一版按换行近似切分，后续可替换为真正语义切分实现。
```

### 3.4 写入示例

```rust
use rag::index::{BuildInput, ChunkerKind, ContentFormat};
use serde_json::json;

let output = builder
    .index(BuildInput {
        content: "原始文档内容，可以很长，会自动切成多个 chunk。".to_owned(),
        title: "密码重置手册".to_owned(),
        kind: "manual".to_owned(),
        format: ContentFormat::Text,
        tenant_id: None,
        user_id: None,
        knowledge_base_id: Some("kb_1".to_owned()),
        metadata: json!({ "source": "upload" }),
        tags: vec!["account".to_owned()],
        chunker: None,       // 使用 Builder 初始化时的默认值
        chunk_size: None,
        chunk_overlap: None,
        delimiter: None,
        keywords: vec!["密码".to_owned(), "重置".to_owned()],
        questions: vec!["怎么重置密码".to_owned()],
    })
    .await?;

println!("document id: {}", output.document.id);
println!("chunk count: {}", output.chunks.len());
```

### 3.5 只构建不写入

```rust
let output = builder.build(input).await?;
// output.chunks 是构建好的 chunk 列表，尚未写入后端
```

## 4. Chunk 字段说明

```text
id：
  chunk 唯一 ID（UUID v4）。

doc_id：
  所属 document 的 ID。

knowledge_base_id：
  知识库 ID，来自 BuildInput。

title：
  来自 document 标题。

content：
  规范化后的 chunk 正文。

content_tokens：
  content 分词结果，用于文本检索。

keywords：
  关键词列表（BuildInput.keywords 或关键词抽取器生成）。

keyword_tokens：
  keywords 分词结果。

questions：
  预设问题列表（BuildInput.questions）。

question_tokens：
  questions 分词结果。

tags：
  标签，来自 BuildInput.tags。

embedding：
  向量（如果传入了 embedder）。
```

关键边界：

```text
content_tokens 是入库产物，写入索引后参与检索，不是用户 query 临时生成的字段。
它可以来自规则 tokenizer、ES analyzer、业务分词服务，或私有化大模型在语义切割时同步生成。
框架不规定生成方式。
```

## 5. 自定义关键词抽取

默认情况下，`DefaultBuilder` 使用 `BuildInput.keywords` 作为关键词。如果想自动生成：

```rust
pub trait KeywordExtractor: Send + Sync + std::fmt::Debug {
    fn extract_keywords<'a>(
        &'a self,
        input: KeywordExtractionInput<'a>,
    ) -> BoxFuture<'a, Result<Vec<String>>>;
}
```

接入方式：

```rust
let builder = DefaultBuilder::with_keyword_extractor(
    Some(store.clone()),
    Some(keyword_extractor),
    Some(embedder),
    "chunks",
)
.with_keyword_top(3);
```

优先级规则：

```text
BuildInput.keywords 非空：
  保留调用方传入的关键词，不调用抽取器。

BuildInput.keywords 为空 且 extractor 存在：
  用 chunk.content 自动抽取，再生成 keyword_tokens。

BuildInput.keywords 为空 且 extractor 不存在：
  keywords / keyword_tokens 保持为空。
```

## 6. 接入 Embedding

```rust
pub trait Embedder: Send + Sync + std::fmt::Debug {
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;
}
```

传入 embedder：

```rust
let builder = DefaultBuilder::new(Some(store.clone()), Some(embedder), "chunks");
```

不传时 `chunk.embedding = None`，不影响文本检索，但向量检索和本地向量 rerank 会降级。

## 7. 直接用 Store 写入自定义数据

如果业务有自己的字段，不需要经过 DefaultBuilder：

```rust
use rag::store::Item;
use serde_json::json;

store
    .insert(
        "custom_chunks",
        Item {
            id: "chunk_1".to_owned(),
            source: json!({
                "id": "chunk_1",
                "doc_id": "doc_1",
                "content": "自定义 chunk 内容",
                "content_tokens": "自定义 chunk 内容",
                "domain_field": "业务字段"
            }),
        },
    )
    .await?;
```

## 8. 注意事项

```text
规范化是单向的（入库时降维），不可逆。
  全角转半角、繁体转简体后，原始形态丢失；如果需要保留原文展示，应在规范化前单独存储 raw content。

chunk_size 的单位是字符数（char），不是字节数。
  中文 1 字 = 1 char，ASCII 1 字符 = 1 char。

keywords / questions 字段来自 BuildInput，不是自动从 content 提取的。
  如果业务需要自动生成，需要接入 KeywordExtractor 或在入库前预处理。

mapping 由业务侧维护。
  DefaultBuilder 不会自动创建 ES index 或 mapping。建议在首次入库前调用 store.create_schema()。

不要把 document 原文和文件元信息放进 DefaultChunk。
  DefaultChunk 是检索单元，document 原文应由业务侧选择独立索引或其他存储。
```
