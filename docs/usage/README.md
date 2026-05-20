# rag 项目使用说明

## 1. 项目定位

`rag` 是一个 Rust RAG 检索库，核心目标是把数据访问、索引构建、查询检索三层拆清楚：

```text
Store:
  数据访问层。负责创建 schema、CRUD、执行调用方传入的后端查询体。

Builder:
  构建层。负责文本提取、格式解析、切分、tokens、embedding、写入 Store。

QueryEngine:
  查询层。负责 query 处理、调用方构造的检索条件、rerank、过滤、分页、返回。
```

当前第一版后端实现是 Elasticsearch：

```text
Elastic:
  Store 的 Elasticsearch 实现。
```

重要边界：

```text
Store 不构造业务查询 DSL。
Store 不做 rerank。
QueryEngine 不构造 ES query_string / multi_match / knn / bool filter。
DefaultDocument / DefaultChunk 只是默认 Builder 的输出形状，不是强制业务 schema。
```

## 2. 环境要求

```text
Rust:
  1.95

Edition:
  Rust 2024

Elasticsearch:
  推荐本地先用单节点开发环境。
```

常用命令：

```bash
cargo fmt
cargo test
cargo check
openspec validate rag-search-core
```

## 3. 模块入口

常用公开入口：

```rust
use rag::{
    DefaultQueryEngine,
    DefaultBuilder,
    Elastic,
    QueryEngine,
    Builder,
    Store,
};

use rag::index::{BuildInput, ChunkerKind, ContentFormat};
use rag::store::Item;
```

默认模型在 `index` 模块下：

```rust
use rag::index::{DefaultChunk, DefaultDocument};
```

注意：

```text
rag::index::DefaultDocument / rag::index::DefaultChunk:
  默认构建模型。

rag::store::Item:
  Store 真正写入和读取的通用数据容器。

rag::query::Hit:
  QueryEngine 返回的通用命中结果，source 保留后端原始字段。
```

## 4. 初始化 Store

单节点：

```rust
use std::sync::Arc;

use rag::{Elastic, Store};

let store: Arc<dyn Store> = Arc::new(Elastic::new("http://127.0.0.1:9200")?);
```

多节点：

```rust
let store: Arc<dyn Store> =
    Arc::new(Elastic::new("http://es1:9200,http://es2:9200")?);
```

说明：

```text
Elastic::new() 只初始化 ES 连接。
index 名称、mapping、查询 body 都由调用方传入。
```

## 5. 创建索引 Schema

`Store::create_schema()` 只负责把调用方传入的 schema 交给后端执行。

示例：

```rust
use serde_json::json;

store
    .create_schema(
        "chunks",
        json!({
            "mappings": {
                "properties": {
                    "id": { "type": "keyword" },
                    "doc_id": { "type": "keyword" },
                    "knowledge_base_id": { "type": "keyword" },
                    "title": { "type": "text" },
                    "content": { "type": "text" },
                    "content_tokens": { "type": "text" },
                    "keywords": { "type": "keyword" },
                    "keyword_tokens": { "type": "text" },
                    "questions": { "type": "text" },
                    "question_tokens": { "type": "text" },
                    "tags": { "type": "keyword" }
                }
            }
        }),
    )
    .await?;
```

生产项目里，mapping 应该由业务侧维护，不要写死在 Store。document 原文和文件元信息如果需要入库，也应由业务侧选择独立索引或其他存储，不放进默认 `Builder` 入库参数。

## 6. 构建并写入 Chunks

`DefaultBuilder` 默认支持：

```text
format:
  Text
  Qa

chunker:
  Fixed
  Delimiter
  Semantic
```

切分实现：

```text
Fixed + chunk_overlap = 0:
  固定大小切分。

Fixed + chunk_overlap > 0:
  滑动窗口切分。

Delimiter:
  按固定分隔符聚合切分。

Semantic:
  第一版先按换行近似切分，后续可替换为真正语义切分。
```

当前阶段的检索 token 字段只要求 `content_tokens`。不设计 `title_tokens`。

入库阶段可以由业务侧或私有化大模型统一生成 chunk 相关字段：

```text
原始文件 / 图片 / OCR / 表格
-> 大模型或业务解析服务
-> 语义切割
-> 生成 chunk content
-> 生成 content_tokens
-> 可选生成 keywords / keyword_tokens
-> 可选生成 questions / question_tokens
-> 写入 chunks index
```

框架不规定 `content_tokens` 必须由传统本地 tokenizer 生成。它可以来自规则 tokenizer、ES/OpenSearch analyzer、Infinity/RAGFlow tokenizer、业务分词服务，或私有化大模型在语义切割时同步生成。关键边界是：`content_tokens` 是入库产物，写入索引后参与检索，不是用户 query 临时生成的 chunk 字段。

示例：

```rust
use std::sync::Arc;

use serde_json::json;

use rag::{DefaultBuilder, Builder};
use rag::index::{BuildInput, ChunkerKind, ContentFormat};

let builder = DefaultBuilder::new(Some(store.clone()), None, "chunks")
    .with_chunking(ChunkerKind::Fixed, 800, 100, None)
    .with_keyword_top(3);

字段边界：

```text
index_name:
  必填。用于写入 chunk 检索单元；正常查询时 QueryEngine.search() 的 index_name 通常也指向这个 index。

document:
  DefaultBuilder.build() / index() 仍会返回 document 输出对象，方便业务侧自行存储原文和元信息。

QueryEngine:
  不读取 DefaultBuilder 的入库参数，也不关心 document 原文存在哪里。
```

let output = builder
    .index(BuildInput {
        content: "这里是原始文档内容。可以很长，会被切成多个 chunk。".to_owned(),
        title: "密码重置手册".to_owned(),
        kind: "manual".to_owned(),
        format: ContentFormat::Text,
        tenant_id: None,
        user_id: None,
        knowledge_base_id: Some("kb_1".to_owned()),
        metadata: json!({ "source": "upload" }),
        tags: vec!["account".to_owned()],
        chunker: None,
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

如果只想构建 JSON，不写入 Store：

```rust
let output = builder.build(input).await?;
```

## 7. 直接使用 Store 写入自定义数据

如果业务有自己的 document/chunk 字段，不需要使用 `DefaultDocument` / `DefaultChunk`。

示例：

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
                "title": "自定义文档",
                "content": "自定义 chunk 内容",
                "permissions": ["user_1"],
                "domain_field": "业务字段"
            }),
        },
    )
    .await?;
```

这也是推荐边界：

```text
业务字段由业务定义。
Store 只负责存取。
QueryEngine 只读取 Store 返回的 source。
```

## 8. 执行检索

`QueryEngine::search()` 参数：

```text
query:
  用户原始输入。

index_name:
  必传。调用方要检索的后端索引名，通常是 chunks index。

body:
  必传。调用方构造好的检索条件，里面可以是 text / vector / hybrid 之一。

page_num / page_size:
  可选。默认 page_num = 1，page_size = 10。
```

### 8.1 Query 处理链路

`QueryEngine` 的 `query` 处理不是简单分词，而是一条固定的 query 规范化链路。目标是把用户原始输入转成稳定、可检索、可解释的关键词和全文检索表达式。

处理顺序：

```text
原始 query
-> 中英文边界加空格
-> 英文转小写
-> 全角转半角
-> 繁体转简体
-> 特殊检索字符替换为空格
-> 去掉疑问词/弱语义词
-> 判断中文路径或英文路径
-> 分词
-> 词权重计算
-> 同义词扩展
-> 可控分词扩展（默认不过度细分）
-> 生成 keywords
-> 生成全文检索表达式
```

核心步骤：

```text
add_space_between_ascii_and_non_ascii():
  给中英文边界补空格，避免 GPT4模型 这类串词被粘在一起。

lower() / strQ2B() / zhconv(Variant::ZhHans):
  统一大小写、全角半角、繁简体；繁体转简体使用第三方 zhconv/OpenCC 规则集。

特殊字符清洗:
  把 :(){}[]*?~^ 等检索语法字符转义或替换掉，避免 DSL 解析报错。

rmWWW():
  去掉疑问词、弱语义词、常见停用词。

is_chinese():
  判断走中文路径还是英文路径。
```

中文路径：

```text
文本清洗
-> term_weight.split()
-> term_weight.weights()
-> 同义词扩展
-> 对长词做受控细粒度切分
-> 拼出 OR / 短语 / boost 表达式
-> 返回 MatchTextExpr + keywords
```

英文路径：

```text
文本清洗
-> rag_tokenizer.tokenize()
-> term_weight.weights(preprocess=false)
-> 同义词扩展
-> 邻接词短语 boost
-> 拼出 MatchTextExpr + keywords
```

说明：

```text
keywords:
  本次 query 的解释性词表，用于本地 rerank、高亮和调试，不是最终 DSL 本身。

全文检索表达式:
  QueryParser 生成的文本检索表达式。DefaultQueryEngine 不会自动把它塞进 Store 请求；
  调用方可以按后端语法把它转换到后端查询 body。

可控分词扩展:
  默认不过度细分，只在长词、中文词或需要召回扩展时再细分。
```

检索示例：

```rust
use std::sync::Arc;

use rag::{DefaultQueryEngine, QueryEngine};
use serde_json::json;

let engine = DefaultQueryEngine::new(store.clone(), None);

let body = json!({
    "size": 1024,
    "query": {
        "bool": {
            "must": [{
                "query_string": {
                    "query": "重置^3 密码^2",
                    "fields": [
                        "questions^8",
                        "keywords^6",
                        "title^4",
                        "content^1",
                        "question_tokens^3",
                        "keyword_tokens^3",
                        "content_tokens^1"
                    ],
                    "default_operator": "OR"
                }
            }],
            "filter": [
                { "term": { "knowledge_base_id": "kb_1" } }
            ]
        }
    }
});

let page = engine
    .search("怎么重置密码", "chunks", body, Some(1), Some(10))
    .await?;
```

返回结果：

```text
page.total:
  过滤后的总命中数。

page.hits:
  当前分页命中。

hit.id:
  后端记录 id。

hit.source:
  后端原始 source。

hit.score:
  当前最终排序分。

hit.scores:
  分数解释，例如 hybrid_score / rerank_score。
```

## 9. 后端请求体示例

文本、向量、混合请求体都由调用方或业务 adapter 构造，QueryEngine 不关心后端 DSL 长什么样。

下面的 ES body 只是业务层示例：它参考 RAGFlow 的一阶段召回形态，把文本召回表达成 `query_string`，向量召回表达成 `knn`，并把同一份 `bool query` 传给 `query` 和 `knn.filter`。其中 `bool query` 内部可以同时包含文本 must 和知识库、权限、标签等业务 filter。框架核心不会固定这种写法。

关键点：

```text
如果希望 query 处理链路影响后端召回：
  先调用 DefaultQueryParser.parse(raw_query)
  再用 parsed.normalized_query 或 parsed.text_expression 构造后端查询 body
  最后把 raw_query、index_name 和 body 一起传给 QueryEngine.search()

raw_query:
  用于日志、追踪、rerank 原始语义。

parsed.normalized_query:
  适合放进普通文本查询、embedding 输入或日志展示。

parsed.text_expression:
  适合放进 query_string / simple_query_string 这类支持 boost 表达式的后端查询。
```

示例：

```rust
use rag::query::{DefaultQueryParser, QueryParser};

let raw_query = "fen (超) 在哪里举办？";
let parsed = DefaultQueryParser.parse(raw_query)?;
let query_vector = embedder.embed(&parsed.normalized_query).await?;

let bool_query = json!({
    "bool": {
        "must": [{
            "query_string": {
                "query": &parsed.text_expression,
                "fields": [
                    "questions^8",
                    "keywords^6",
                    "title^4",
                    "content^1",
                    "question_tokens^3",
                    "keyword_tokens^3",
                    "content_tokens^1"
                ],
                "type": "best_fields",
                "minimum_should_match": "30%"
            }
        }],
        "filter": [
            { "term": { "knowledge_base_id": "kb_1" } }
        ],
        "boost": 0.05
    }
});

let body = json!({
    "size": 1024,
    "query": bool_query.clone(),
    "knn": {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 1024,
        "num_candidates": 2048,
        "boost": 0.95,
        "filter": bool_query
    },
});

let response = engine
    .search(raw_query, "chunks", body, Some(1), Some(10))
    .await?;
```

### 9.1 文本检索 body

```rust
let body = json!({
    "size": 1024,
    "query": {
        "bool": {
            "must": [{
                "query_string": {
                    "query": "重置^3 密码^2",
                    "fields": [
                        "questions^8",
                        "keywords^6",
                        "title^4",
                        "content^1",
                        "question_tokens^3",
                        "keyword_tokens^3",
                        "content_tokens^1"
                    ],
                    "type": "best_fields",
                    "minimum_should_match": "30%"
                }
            }],
            "filter": [
                { "term": { "knowledge_base_id": "kb_1" } }
            ]
        }
    },
    "highlight": {
        "fields": {
            "content": {}
        }
    }
});
```

### 9.2 向量检索 body

```rust
let body = json!({
    "size": 1024,
    "knn": {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 1024,
        "num_candidates": 2048,
        "filter": {
            "bool": {
                "filter": [
                    { "term": { "knowledge_base_id": "kb_1" } }
                ]
            }
        }
    }
});
```

### 9.3 混合检索 body

```rust
let bool_query = json!({
    "bool": {
        "must": [{
            "query_string": {
                "query": "重置^3 密码^2",
                "fields": [
                    "questions^8",
                    "keywords^6",
                    "title^4",
                    "content^1",
                    "question_tokens^3",
                    "keyword_tokens^3",
                    "content_tokens^1"
                ],
                "type": "best_fields",
                "minimum_should_match": "30%"
            }
        }],
        "filter": [
            { "term": { "knowledge_base_id": "kb_1" } }
        ],
        "boost": 0.05
    }
});

let body = json!({
    "size": 1024,
    "query": bool_query.clone(),
    "knn": {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 1024,
        "num_candidates": 2048,
        "boost": 0.95,
        "filter": bool_query
    },
    "highlight": {
        "fields": {
            "content": {}
        }
    }
});
```

### 9.4 调用 QueryEngine

```rust
let response = engine
    .search("怎么重置密码", "chunks", body, Some(1), Some(10))
    .await?;
```

调用流程：

```text
1. 按 Query 处理链路生成 keywords 和全文检索表达式。
2. 把调用方传入的 index_name 和 body 直接发给 Store。
3. 如果配置了外部 reranker，用外部 reranker 二阶段排序。
4. 如果没有外部 reranker，用默认本地 scorer 做 token + vector 二阶段 rerank。
5. 按 score_threshold 过滤。
6. 分页返回。
```

本地 rerank fallback：

```text
query 侧:
  使用 QueryParser 生成的 keywords。
  对 keywords 做词权重归一化。
  每个 token 贡献 0.4。
  相邻 token 拼成 phrase，贡献 0.6。

chunk 侧:
  content_tokens 作为正文 token。
  keywords / keyword_tokens 参与 token 候选。
  questions / question_tokens 参与 token 候选。

token_similarity:
  命中的 query token/phrase 权重 / query token/phrase 总权重。

vector_similarity:
  如果调用方传入的 body 里有 knn.query_vector，并且 hit.source 里有 embedding 或 q_{dim}_vec，
  本地 rerank 会计算 query vector 与 chunk vector 的 cosine similarity。

最终 score:
  没有外部 reranker:
    rerank_score = term_score * (1 - hybrid_score_weight) + vector_score * hybrid_score_weight

  没有 query vector 或 chunk vector:
    rerank_score = term_score

  有外部 reranker:
    rerank_score = term_score * (1 - hybrid_score_weight) + model_score * hybrid_score_weight
```

说明：

```text
这个逻辑参考 RAGFlow 的本地 rerank 思路：
  不是 raw content contains。
  不是简单命中词数 / 总词数。
  而是 query token 权重命中比例，并补充相邻 phrase 权重。

当前实现没有绑定外部词频词典、NER、POS tagger，所以 term weight 是轻量启发式版本；
整体排序结构、token/phrase 权重、向量 cosine 组合方式保持一致。
```

本地 scorer 是可替换策略：

```rust
use std::sync::Arc;

use rag::{DefaultQueryEngine, LocalScorer};

let scorer = LocalScorer::default()
    .with_token_weights(0.4, 0.6)
    .with_field_weights(
        0, // title 当前阶段不参与本地 rerank
        5, // keywords / keyword_tokens
        6, // questions / question_tokens
    )
    .with_vector_fields(["embedding"]);

let engine = DefaultQueryEngine::with_scorer(
    store.clone(),
    Arc::new(scorer),
    None,
    100,
    0.2,
    0.95,
);
```

如果业务要完全换一套本地打分逻辑，实现 `QueryScorer` trait 后传给 `DefaultQueryEngine::with_scorer()` 即可。

注意：

```text
DefaultQueryEngine 的 top 默认是 100，只截断 Store 返回后的候选集。
ES 查询 body 里的 size 仍然需要调用方自己设置。
```

## 10. 覆盖默认检索参数

默认参数：

```text
top:
  100

score_threshold:
  0.2

hybrid_score_weight:
  0.95

hybrid_score_weight 含义:
  二阶段本地排序中的 vector_score 或 model_score 权重。
  默认 0.95 时，term_score 权重就是 0.05。
```

需要覆盖时：

```rust
let engine = DefaultQueryEngine::with_search_settings(
    store.clone(),
    None,
    512,
    0.35,
    0.95,
);
```

## 11. 接入外部关键词抽取

`KeywordExtractor` 是可选能力。默认 `DefaultBuilder::new(...)` 不会自动生成关键词，只会使用 `BuildInput.keywords`。如果传入关键词抽取器，builder 会在每个 chunk tokenize 之前补齐 `keywords`，再生成 `keyword_tokens`。

如果业务侧使用私有化大模型做语义切割，也可以在同一步直接生成 `keywords`、`keyword_tokens`、`questions`、`question_tokens`。这属于入库构建产物，框架不绑定生成方式。

处理规则：

```text
BuildInput.keywords 非空:
  保留调用方传入的关键词，不再自动抽取。

BuildInput.keywords 为空 且 keyword_extractor 存在:
  使用 chunk.content 生成 keywords，再 tokenize 得到 keyword_tokens。

BuildInput.keywords 为空 且 keyword_extractor 不存在:
  keywords 和 keyword_tokens 保持为空。
```

接口形态：

```rust
pub trait KeywordExtractor: Send + Sync + std::fmt::Debug {
    fn extract_keywords<'a>(
        &'a self,
        input: KeywordExtractionInput<'a>,
    ) -> BoxFuture<'a, Result<Vec<String>>>;
}
```

使用方式：

```rust
let builder = DefaultBuilder::with_keyword_extractor(
    Some(store.clone()),
    Some(keyword_extractor),
    Some(embedder),
    "chunks",
)
.with_keyword_top(3);
```

这个设计对应 RAGFlow 的 `auto_keywords` 思路：关键词是 chunk 构建阶段提前生成并写入索引字段，不是用户 query 临时生成。区别是这里不把 LLM 绑死在 builder 里，具体抽取器由调用方实现。

## 12. 接入外部 Embedding

`Embedder` 是 trait。默认 `DefaultBuilder` 如果传入 embedder，会在构建 chunk 时写入 `embedding`。

接口形态：

```rust
pub trait Embedder: Send + Sync + std::fmt::Debug {
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;
}
```

使用方式：

```rust
let builder = DefaultBuilder::new(
    Some(store.clone()),
    Some(embedder),
    "chunks",
);
```

如果没有传入 embedder：

```text
chunk.embedding = None
```

## 13. 接入外部 Reranker

`Reranker` 是 trait。传给 `DefaultQueryEngine` 后，检索结果会走外部 rerank。

接口形态：

```rust
pub trait Reranker: Send + Sync + std::fmt::Debug {
    fn rerank<'a>(&'a self, query: &'a str, hits: &'a [Hit]) -> BoxFuture<'a, Result<Vec<f32>>>;
}
```

使用方式：

```rust
let engine = DefaultQueryEngine::new(store.clone(), Some(reranker));
```

要求：

```text
返回的分数数量应与 hits 数量一致。
如果不传 reranker，则使用本地 rerank fallback。
```

## 13. 推荐接入路径

第一阶段最小可用：

```text
1. 启动 ES。
2. 用业务 mapping 创建 chunks index。
3. 用 DefaultBuilder 写入 chunk 检索数据。
4. 调用方按业务需要构造 ES 查询 body，例如 query_string / knn。
5. 用 DefaultQueryEngine.search() 返回 hits。
```

业务化扩展：

```text
1. 自定义 document/chunk 字段。
2. 自定义 mapping。
3. 自定义 Builder 或在 build 后转换 Item.source。
4. 接入 embedding 服务。
5. 接入 rerank 服务。
6. 根据业务权限、知识库、用户、标签构造 ES filter。
```

## 14. 常见注意事项

```text
不要把业务 filter 放进 QueryEngine 参数。
不要让 Store 拼接 ES bool query。
不要依赖 DefaultDocument / DefaultChunk 作为全局 schema。
不要忘记在 ES 查询 body 里设置 size。
不要把对象存储、任务调度、权限系统塞进 Store。
```

如果要扩展后端：

```text
实现 Store trait。
保持 search() 接收调用方构造的 index_name 和 body。
不要在后端实现里写死业务字段。
```

如果要扩展构建流程：

```text
实现 Builder trait。
或者替换 Extractor / Parser / ChunkPipeline / Tokenizer / Embedder。
新增切分算法时，实现 Chunker trait，再挂到 Pipeline 或自定义 ChunkPipeline。
```

如果要扩展查询流程：

```text
实现 QueryEngine trait。
或者复用 DefaultQueryEngine，再传入不同 Store / Reranker / 查询 body。
```

## 15. 本地资料 Demo

项目提供了两个 demo 入口：

```text
examples/ingest.rs:
  只负责读取资料、切 chunk、向量化、写入 ES。

examples/search.rs:
  只负责构造混合检索条件并查询。

examples/common/demo.rs:
  两个入口共用的配置和 helper。
```

它会做这些事情：

```text
1. 读取 `examples/data` 下的 `.txt` 文件。
2. 使用 Fixed + chunk_overlap 的滑动窗口方式切 chunk。
3. 使用本地 `DemoEmbedder` 生成 1024 维演示向量。
4. 写入本地 Elasticsearch。
5. 后续可以重复运行 search 入口，不需要每次重新入库。
```

启动单节点 Elasticsearch：

```bash
bash scripts/start-single-es.sh
```

或者手动启动：

```bash
docker run -d \
  --name rag-es \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.3
```

首次入库：

```bash
cargo run --example ingest
```

执行召回：

```bash
cargo run --example search
```

默认配置写在 `examples/common/demo.rs` 里：

```text
ES_URL:
  http://127.0.0.1:9200

DEMO_QUERY:
  怎么重置密码？

DEMO_KNOWLEDGE_BASE_ID:
  demo_knowledge_base
```

注意：

```text
如果本地 Docker 端口转发还没就绪，先用 docker logs rag-es 查看 ES 是否完成启动。
```
