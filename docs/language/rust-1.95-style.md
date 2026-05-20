# Rust 1.95 写法规范

本文档是 `rag` 项目的 Rust 编码基线。

目标不是炫技，而是避免用旧 Rust 习惯写新项目。后续代码默认使用 Rust 1.95 stable 和 Rust 2024 edition。

## 1. 基础版本

`Cargo.toml` 统一使用：

```toml
[package]
edition = "2024"
rust-version = "1.95"
```

约定：

```text
使用 stable Rust。
不使用 nightly feature。
不引入宏驱动的大框架。
新语法只在能提升清晰度时使用。
```

## 2. Rust 1.80 到 1.95 重点变化

### 2.1 Rust 1.80

可用写法：

```rust
match value {
    0..10 => "small",
    10..100 => "medium",
    .. => "large",
}
```

说明：

```text
模式匹配里的排除式范围更完整。
cfg 检查更严格，拼错 cfg 名称更容易被发现。
```

项目建议：

```text
范围匹配可以正常用。
不要写奇怪的 cfg 名称。
```

### 2.2 Rust 1.85 / Rust 2024

Rust 1.85 是 Rust 2024 edition 的稳定版本，对新项目影响最大。

重点能力：

```text
async closures。
AsyncFn / AsyncFnMut / AsyncFnOnce。
RPIT lifetime capture 规则更新。
unsafe extern blocks。
unsafe attributes。
unsafe_op_in_unsafe_fn lint。
if let 临时值作用域变化。
gen 变成保留关键字。
```

#### async closures

推荐：

```rust
let embed = async |text: String| -> Result<Vec<f32>, Error> {
    client.embed(text).await
};
```

不要为了异步闭包手动写：

```rust
Box::pin(async move { ... })
```

除非确实需要类型擦除。

#### AsyncFn

接受异步回调时优先使用：

```rust
async fn retry<F, T, E>(mut operation: F) -> Result<T, E>
where
    F: AsyncFnMut() -> Result<T, E>,
{
    operation().await
}
```

带参数时：

```rust
async fn call<F>(f: F, input: String) -> Result<String, Error>
where
    F: AsyncFn(String) -> Result<String, Error>,
{
    f(input).await
}
```

适合：

```text
retry 操作。
pipeline hook。
临时异步回调。
异步中间件。
异步组合子。
```

不适合替代稳定组件抽象。`Store`、`Embedder`、`Reranker` 这种长期依赖仍然用 trait。

#### 不再优先使用 Box<dyn Future>

旧写法：

```rust
use std::future::Future;
use std::pin::Pin;

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
```

只有这些场景才考虑：

```text
异质异步任务集合。
运行时动态注册不同类型的 async callback。
trait object 接口必须隐藏具体 future 类型。
```

普通高阶异步函数优先使用 `AsyncFn` / `AsyncFnMut`。

#### unsafe 写法

Rust 2024 里 unsafe 边界更显式。

推荐：

```rust
unsafe extern "C" {
    fn external_call(value: i32) -> i32;
}
```

`unsafe fn` 内部仍然显式写 `unsafe` block：

```rust
unsafe fn call_external(value: i32) -> i32 {
    unsafe { external_call(value) }
}
```

项目建议：

```text
第一版业务代码尽量不写 unsafe。
如果必须写 unsafe，要把 unsafe 范围缩到最小。
```

### 2.3 Rust 1.88

#### let chains

推荐：

```rust
if let Some(value) = request.metadata.get("trace_id")
    && let Some(trace_id) = value.as_str()
    && !trace_id.is_empty()
{
    tracing::info!(trace_id, "request.trace");
}
```

适合：

```text
解析可选参数。
连续判断 Option / Result。
```

不要滥用成很长一串。超过三四个条件时，拆成局部变量更清楚。

#### cfg(true) / cfg(false)

可以用于临时编译期开关：

```rust
#[cfg(true)]
fn enabled() {}
```

项目建议：

```text
正常业务不用它。
测试或平台差异代码可以少量使用。
```

### 2.4 Rust 1.89

#### const generic `_` 推断

可用：

```rust
let values: [u8; _] = [1, 2, 3];
```

项目建议：

```text
只有类型非常明显时才用。
公共 API 和复杂泛型里不要依赖 `_`，显式更清楚。
```

### 2.5 Rust 1.95

#### match if let guard

推荐：

```rust
match mode {
    SearchMode::Hybrid if let Some(vector) = &self.vector => {
        self.search_hybrid(query, vector).await
    }
    SearchMode::Vector if let Some(vector) = &self.vector => {
        self.search_vector(query, vector).await
    }
    _ => self.search_text(query).await,
}
```

适合：

```text
根据枚举模式 + Option 能力选择分支。
根据状态 + 附加字段做分支。
```

不要把复杂业务判断全塞进 guard，分支里仍然要保持可读。

#### cfg_select!

`cfg_select!` 用于编译期平台选择。

项目建议：

```text
第一版不用。
只有将来需要不同 OS / target 的底层实现时再用。
不要拿它做业务 if/else。
```

## 3. 项目整体写法

### 3.0 模块组织

Cargo 目前官方最新 edition 是 `2024`，不是 `2025`。但模块组织采用现代 Rust 风格：不使用 `mod.rs`。

推荐：

```text
src/lib.rs
src/core.rs
src/core/error.rs
src/core/models.rs
src/core/traits.rs
src/store.rs
src/store/elastic.rs
src/index.rs
src/index/builder.rs
src/query.rs
src/query/default.rs
src/query/parse.rs
```

不推荐：

```text
src/core/mod.rs
src/store/mod.rs
src/store/elastic/mod.rs
```

原则：

```text
顶层模块用 src/<module>.rs。
子模块用 src/<module>/<child>.rs。
避免 mod.rs，减少同名文件在编辑器里混在一起。
```

### 3.1 三层架构

后续 Rust 代码保持三层：

```text
Store:
  数据访问层。负责指定 index 的 CRUD 和单次基础检索执行。

Builder:
  构建层。负责 extract / parse / chunk / tokenize / embed / index。

QueryEngine:
  查询层。负责 query 解析、执行调用方传入的召回请求、rerank、过滤、分页、返回。
```

边界：

```text
Store 不做 rerank。
Store 不构造业务查询 DSL。
Builder 不做查询。
QueryEngine 不做文件解析和入库构建。
QueryEngine 不构造业务查询 DSL；ES query_string / multi_match / knn / bool filter 等查询体由调用方或业务适配层拼好。
```

### 3.2 组件抽象

稳定依赖用 trait：

```rust
pub trait Store: Send + Sync {
    fn search<'a>(&'a self, index_name: &'a str, body: Value) -> BoxFuture<'a, Result<Vec<SearchHit>>>;
}
```

说明：

```text
长期组件需要 object safe，并放进 Arc<dyn Trait> 时，trait 方法返回 BoxFuture。
不要引入 async_trait；项目保持显式 BoxFuture，边界更清楚。
如果是泛型静态分发，可以考虑原生 async fn in trait 或 AsyncFn。
```

长期组件建议：

```text
Arc<dyn Store + Send + Sync>
Arc<dyn Embedder + Send + Sync>
Arc<dyn Reranker + Send + Sync>
```

临时异步回调用 `AsyncFn`。

### 3.3 查询入口参数

不要用：

```rust
HashMap<String, serde_json::Value>
```

表达框架自己的入口参数。

推荐：

```rust
engine
    .search(query, "chunks", body, Some(page_num), Some(page_size))
    .await?;
```

原则：

```text
分页参数直接用 page_num / page_size，不为两个字段单独造 Params。
tenant / user / knowledge_base / doc / tags / highlight 等业务条件不要塞进 QueryEngine 参数。
业务查询条件、字段权重、高亮、权限过滤直接体现在调用方构造的 body 里。
默认值在使用处或 Default 实现里定义。
```

检索策略参数有内置默认值；默认场景直接初始化：

```rust
DefaultQueryEngine::new(store, reranker);
```

需要覆盖默认值时，在具体调用处显式设置：

```rust
DefaultQueryEngine::with_search_settings(store, reranker, top, score_threshold, hybrid_score_weight);
```

### 3.4 错误处理

使用 `thiserror`：

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("external service error: {0}")]
    External(String),
}
```

建议分法：

```text
InvalidInput:
  参数错误、空 query、分页错误。

Unsupported:
  不支持的文件类型、format、chunker。

Store:
  ES / OpenSearch / 后端存储错误。

External:
  embedding / rerank 服务错误。

Internal:
  代码假设被破坏、schema 不一致。
```

库层返回：

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

### 3.5 日志

使用 `tracing`，不用 `println!`：

```rust
tracing::info!(
    query = %query,
    top = self.top,
    "search.start"
);
```

阶段事件名：

```text
index.extract.start
index.extract.done
index.parse.done
index.chunk.done
index.tokenize.done
index.embed.done
index.store.done
index.failed

search.start
search.prepare.done
search.rerank.done
search.filter.done
search.page.done
search.failed
```

原则：

```text
日志用于排查。
不单独设计 Observer。
不单独设计复杂 Trace。
不把日志字段作为默认 API 返回。
```

### 3.6 retry

retry 只用于短暂失败。

可以重试：

```text
ES 连接重置 / 超时。
embedding 服务限流 / 超时。
rerank 服务限流 / 超时。
```

不重试：

```text
参数错误。
文件格式不支持。
schema 错误。
query 解析错误。
本地算法错误。
```

推荐工具函数形态：

```rust
pub async fn retry<F, T, E>(mut operation: F, max_retries: usize) -> Result<T, E>
where
    F: AsyncFnMut() -> Result<T, E>,
{
    let mut attempt = 0;
    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) if attempt < max_retries => {
                attempt += 1;
                tokio::time::sleep(backoff(attempt)).await;
            }
            Err(error) => return Err(error),
        }
    }
}
```

注意：实际实现里需要判断错误是否可重试，不要所有错误都 retry。

## 4. 数据结构命名

统一命名：

```text
index::DefaultDocument
index::DefaultChunk
Hit
HitPage
Scores
```

说明：

```text
index::DefaultDocument / index::DefaultChunk:
  只是默认 Builder 使用的字段形状，方便第一版快速跑通。
  放在 index 模块下面，不在 crate root 暴露成全局 Document / Chunk。

业务自定义 document/chunk:
  调用方可以自己定义字段、mapping、索引结构和 Builder。

Store:
  只认 Item { id, source }，不强制 index::DefaultDocument / index::DefaultChunk schema。

QueryEngine:
  返回 HitPage { total, hits }，每个 Hit 的 source 保留后端原始字段。
```

不要使用：

```text
SearchParams
QueryRequest
SearchDefaults
Params
final_score
candidates
topk
page
tenant_ids
embed_model
rerank_model
```

分数命名：

```text
score:
  当前最终排序分。

scores:
  本次 query 产生的解释分，例如 hybrid_score/rerank_score/rank_feature。
```

返回结果命名：

```text
hits:
  当前查询返回的命中结果。
```

## 5. async Fn 使用准则

优先使用：

```rust
F: AsyncFn(Input) -> Output
F: AsyncFnMut(Input) -> Output
F: AsyncFnOnce(Input) -> Output
```

适用场景：

```text
函数参数是异步回调。
不需要把回调存进结构体长期保存。
不需要异质集合。
希望静态分发、零堆分配。
```

仍然使用 trait object 的场景：

```text
组件需要长期保存在 struct 里。
运行时替换不同实现。
需要统一对象接口。
例如 Store / Embedder / Reranker。
```

仍然可能使用 boxed future 的场景：

```text
异质 future 集合。
object safe trait 返回 future。
第三方库接口要求 BoxFuture。
```

## 6. 适合 rag 项目的依赖

第一版建议：

```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
tokio = { version = "1", features = ["time", "macros", "rt-multi-thread"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tracing = "0.1"
uuid = { version = "1", features = ["v4", "serde"] }
elasticsearch = { version = "9.1.0-alpha.1", default-features = false, features = ["rustls-tls"] }
```

说明：

```text
tokio:
  异步运行时。

serde / serde_json:
  请求、响应、索引字段、配置序列化。

thiserror:
  明确错误类型。

tracing:
  结构化日志。

uuid / chrono:
  构建 document/chunk id 和时间字段。

elasticsearch:
  官方 Elasticsearch Rust SDK，只在 Elastic 内部使用。
```

当前 ES 后端使用官方 `elasticsearch-rs` SDK，并且只封装在 `Elastic` 内部：

```toml
elasticsearch = { version = "9.1.0-alpha.1", default-features = false, features = ["rustls-tls"] }
```

注意：

```text
Store trait 不能暴露 elasticsearch-rs 类型。
QueryEngine / Builder 不能直接依赖 elasticsearch-rs。
官方 crate 当前 crates.io 可用版本是 9.x alpha，版本风险必须隔离在 Elastic 内部。
如果后续出现稳定的 8.x/9.x client，优先只替换 Elastic 内部实现。
```

## 7. 推荐代码风格

### 7.1 Result 优先

推荐：

```rust
let hits = self.store.search(index_name, body).await?;
```

不要：

```rust
let hits = self.store.search(index_name, body).await.unwrap();
```

### 7.2 小函数

每个函数只做一件事：

```text
query_parser.parse()
search_hits()
rerank()
filter_hits()
paginate_hits()
build_response()
```

可复用工具函数不要散落在业务流程文件里，按功能放到 `src/utils/<功能>.rs`：

```text
src/utils/normalization.rs:
  文本清洗、空格处理、弱词判断。

src/query/scorer.rs:
  QueryEngine 本地二阶段分数策略；属于 query 层，不放 utils。

src/utils/hit.rs:
  Store hit 到 QueryEngine hit 的转换。
```

Parser / Chunker 不放 `utils`，它们属于 Builder 体系：

```text
src/index/parser/text.rs:
  PlainText，普通文本格式解析对象。

src/index/parser/qa.rs:
  Qa，QA 格式解析对象。

src/index/chunker/pipeline.rs:
  Pipeline，负责按 ChunkerKind 分发并统一修正 chunk 顺序。

src/index/chunker/fixed_size.rs:
  Fixed，固定大小切分，不做 overlap。

src/index/chunker/sliding_window.rs:
  SlidingWindow，固定窗口 + overlap 滑动切分。

src/index/chunker/delimiter.rs:
  Delimiter，按固定分隔符聚合切分。

src/index/chunker/semantic.rs:
  Semantic，语义切分入口；第一版先按段落/换行近似实现，后续可替换为真正语义算法。
```

Parser / Chunker 算法不要写成散落的 `pub fn parse()` / `pub fn chunk()`。
每种算法优先定义成结构体，并通过 `ParseInput` / `ChunkInput` 接收参数。

### 7.3 明确所有权

输入内容尽量用引用，跨 await 或需要持有时再 clone。

需要进入 async task 或长期保存时：

```rust
let query = query.to_owned();
```

### 7.4 不过度泛型

内部工具函数可以泛型。
公共业务结构优先清晰。

不要为了“零成本”把业务类型写成难读的泛型森林。

## 8. 官方资料

参考：

```text
Rust 1.80:
  https://blog.rust-lang.org/2024/07/25/Rust-1.80.0/

Rust 1.85 / Rust 2024:
  https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/

Rust 1.88:
  https://blog.rust-lang.org/2025/06/26/Rust-1.88.0/

Rust 1.89:
  https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/

Rust 1.95:
  https://blog.rust-lang.org/2026/04/16/Rust-1.95.0/

Rust 2024 Edition Guide:
  https://doc.rust-lang.org/edition-guide/rust-2024/index.html
```
