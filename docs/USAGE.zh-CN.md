# Heart 使用手册

本手册介绍如何在 Rust 应用中使用 Heart 构建 Agent 和 Workflow。示例按当前公开 API 编写。

[English version](USAGE.en.md)

## 1. 安装

Heart 的 Cargo 包名是 `halo-agent`，Rust 库名是 `halo`，要求 Rust 1.99.1 或更新版本。

```toml
[dependencies]
halo = { package = "halo-agent", version = "0.1.2" }
futures-util = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "0.8"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

### 1.1 从零运行第一个示例

仓库中的 `examples/` 是可直接编译运行的完整程序，不是伪代码。进入 Heart 仓库后，可以先用不需要网络和密钥的示例验证环境：

```bash
cargo check --offline --examples
cargo run --offline --example agent_middleware
cargo run --offline --example node_macro
cargo run --offline --example workflow_events
```

如果是在自己的 Cargo 项目中使用，把上面的依赖加入 `Cargo.toml`，再把任意示例复制到 `src/main.rs`。示例与本手册的对应关系如下，
每一行都可以直接作为运行入口：

| 能力 | 可运行示例 | 命令 |
| --- | --- | --- |
| Chat Agent、Memory | [`examples/agent_memory.rs`](../examples/agent_memory.rs) | `cargo run --example agent_memory` |
| ReAct、Tool、Permission | [`examples/agent_permission.rs`](../examples/agent_permission.rs) | `cargo run --example agent_permission` |
| Middleware | [`examples/agent_middleware.rs`](../examples/agent_middleware.rs) | `cargo run --example agent_middleware` |
| Workflow 事件流 | [`examples/workflow_events.rs`](../examples/workflow_events.rs) | `cargo run --example workflow_events` |
| Checkpoint 保存与恢复 | [`examples/checkpoint.rs`](../examples/checkpoint.rs) | `cargo run --offline --example checkpoint` |
| `#[node]` 和强类型节点 | [`examples/node_macro.rs`](../examples/node_macro.rs) | `cargo run --example node_macro` |
| Planner、Evaluator | [`examples/planning_evaluation.rs`](../examples/planning_evaluation.rs) | `cargo run --example planning_evaluation` |
| Prompt、压缩 | [`examples/compress.rs`](../examples/compress.rs) | `HALO_API_KEY=... cargo run --example compress` |
| Session/Memory CRUD | [`examples/short_memory.rs`](../examples/short_memory.rs) | `cargo run --example short_memory` |
| Document Loader/Parser/Transformer | [`examples/document_parse.rs`](../examples/document_parse.rs) | `cargo run --example document_parse -- ./input.csv` |
| PDF/Office/图片解析 | [`examples/document_liteparse.rs`](../examples/document_liteparse.rs) | `cargo run --example document_liteparse -- ./input.pdf` |
| OpenAI-compatible Chat | [`examples/provider_chat.rs`](../examples/provider_chat.rs) | `OPENAI_API_KEY=... cargo run --example provider_chat` |
| DeepSeek Chat Completions | [`examples/provider_deepseek.rs`](../examples/provider_deepseek.rs) | `DEEPSEEK_API_KEY=... cargo run --example provider_deepseek` |
| 真实 DeepSeek Agent、Tool 和 fastrace | [`examples/agent_live_deepseek.rs`](../examples/agent_live_deepseek.rs) | `DEEPSEEK_API_KEY=... cargo run --example agent_live_deepseek` |
| 三个并发 SubAgent | [`examples/agent_live_three_subagents.rs`](../examples/agent_live_three_subagents.rs) | `HEART_VLLM_BASE_URL=... HEART_VLLM_API_KEY=... HEART_VLLM_MODEL=... cargo run --example agent_live_three_subagents` |
| VLLM Chat Completions | [`examples/provider_vllm.rs`](../examples/provider_vllm.rs) | `HEART_VLLM_BASE_URL=... HEART_VLLM_API_KEY=... HEART_VLLM_MODEL=... cargo run --example provider_vllm` |
| 基础组件契约（Function、Prompt、Search、Embed、Sandbox、Event） | [`examples/component_contracts.rs`](../examples/component_contracts.rs) | `cargo run --offline --example component_contracts` |

网络示例只在明确执行命令时访问外部服务；密钥只从环境变量读取，不写入源码。`cargo check --examples` 不会发起模型或 MCP 请求，适合先验证代码。

目录职责：

```text
components/   公共契约：Agent、Model、Message、Event、Tool、Memory、Checkpoint
core/         具体实现：Provider、Chat/ReAct、Workspace、Sandbox、Memory 后端
runtime/      Agent 与 Workflow 的状态、图和执行控制
harness/      面向长任务 Agent 的高阶组合能力
```

没有配置的能力不会自动启用：没有 Model 不能运行 Agent，没有 Sandbox 就没有命令工具，没有 Checkpoint 就不能跨进程恢复。

### 1.2 组件逐项索引

下面的索引覆盖 `halo::agent` 下的公共组件契约。每一行都说明组件负责什么、如何使用以及对应的完整示例；具体实现位于
`halo::agent::core`，运行现场仍由 `halo::agent::workflow` 或 Agent 运行时负责。`component_contracts` 是离线的基础契约示例，
不会伪造模型响应，也不会访问外部服务。

| 组件 | 作用和边界 | 使用示例 |
| --- | --- | --- |
| `agent::Agent`、`agent::chat`、`agent::react` | Agent 只接收统一 `Input`，通过事件流交付模型/工具调用结果；运行现场、等待和恢复由运行时拥有。 | [`agent_live_deepseek.rs`](../examples/agent_live_deepseek.rs)、[`agent_permission.rs`](../examples/agent_permission.rs) |
| `agent::Input`、`Message`、`Messages` | 统一用户输入、模型上下文和节点间业务数据；`metadata` 与有序 `payload` 分开保存。 | [`component_contracts.rs`](../examples/component_contracts.rs)、[`agent_live_deepseek.rs`](../examples/agent_live_deepseek.rs) |
| `error::Error` | 所有组件共享 `code/message/details` 错误外壳；各组件只增加自己的错误码，不改变外层结构。 | 所有示例的 `Result<_, halo::agent::Error>`；[`component_contracts.rs`](../examples/component_contracts.rs) |
| `event::Event` | 对外观察流的 `Delta`/`Complete`；不承担 Runtime State 或恢复。 | [`component_contracts.rs`](../examples/component_contracts.rs)、[`workflow_events.rs`](../examples/workflow_events.rs) |
| `model::chat` | Chat Provider 的 `Request`、工具描述、`Event` 流和 `Usage` 约定；Provider 负责网络适配。 | [`provider_chat.rs`](../examples/provider_chat.rs)、[`provider_deepseek.rs`](../examples/provider_deepseek.rs)、[`provider_vllm.rs`](../examples/provider_vllm.rs) |
| `model::embed`、`model::token`、`model::content` | 单条向量请求、共享 Token Usage 和规范化内容 Part；Embed 返回统一 `Message`。 | [`component_contracts.rs`](../examples/component_contracts.rs)；真实 Provider 配置见模型章节 |
| `function::Function` | 不知道图、节点或执行 ID 的独立异步业务函数；输入输出类型由调用方决定。 | [`component_contracts.rs`](../examples/component_contracts.rs)、[`node_macro.rs`](../examples/node_macro.rs) |
| `tool::Tool` | 具有定义和 `before/call/after/eval` 生命周期的工具契约；执行结果和错误由调用方收口。 | [`agent_live_tool.rs`](../examples/agent_live_tool.rs)、[`agent_permission.rs`](../examples/agent_permission.rs) |
| `workspace::Workspace` | 文件、命令、Glob/Grep 等工作区能力的统一入口；Local 和 Server 由实现决定，默认能力按挂载启用。 | [10.8 Workspace 与 Skill](#108-workspace和skill本地发现内容读取和安全边界) |
| `skill::Skill` | 技能的元数据、说明和内容读取契约；发现和缓存由 Workspace 实现，模型只看到可用描述。 | [10.8 Workspace 与 Skill](#108-workspace和skill本地发现内容读取和安全边界) |
| `mcp::Server` | MCP 连接和工具发现契约；挂载前完成连接与发现，运行时直接调用已发现工具。 | [10.9 MCP](#109-mcp先连接和发现再挂载) |
| `permission::Permission` | 在实际 Tool/MCP 调用前给出允许或拒绝；不执行工具，也不持有 Agent State。 | [`agent_permission.rs`](../examples/agent_permission.rs) |
| `middleware::Middleware` | 对模型、工具、权限、压缩等阶段提供统一 `before/after` 钩子；重试策略是可选实现。 | [`agent_middleware.rs`](../examples/agent_middleware.rs) |
| `prompt::Template`、`Variables` | 按定义顺序渲染单条 Message 和命名历史 `Messages`，不执行模型调用。 | [`component_contracts.rs`](../examples/component_contracts.rs)、[`compress.rs`](../examples/compress.rs) |
| `compression::Compression` | 将历史 Message 压缩为新的摘要 Message；阈值和触发时机由 Harness/应用配置。 | [`compress.rs`](../examples/compress.rs) |
| `memory` | 短期 Session/Chat 和长期 `summary/fact/preference` 的存储契约；后端可为内存、PG、ES 等。 | [`short_memory.rs`](../examples/short_memory.rs)、[`agent_memory.rs`](../examples/agent_memory.rs) |
| `checkpoint::Checkpoint` | 保存和读取一次运行的完整可恢复现场；不等同于 Chat 审计或 Trace。 | [`checkpoint.rs`](../examples/checkpoint.rs) |
| `sandbox::Sandbox` | 受明确请求约束的执行边界；`core::sandbox::Local` 是本地进程实现，不自动提供隔离。 | [`component_contracts.rs`](../examples/component_contracts.rs) |
| `knowledge::Search` | `keyword`、`vector`、`hybrid` 三种请求契约；不内部创建 Embed 或决定索引后端。 | [`component_contracts.rs`](../examples/component_contracts.rs)、Document/Knowledge 章节 |
| `document` | Loader、Parser、Transformer、Chunker、RAPTOR 和 Graph 的文档处理能力；写入哪个知识库由应用决定。 | [`document_parse.rs`](../examples/document_parse.rs)、[`document_liteparse.rs`](../examples/document_liteparse.rs) |
| `node::Node`、`node::Schema` | Workflow 节点输入输出契约和强类型 Schema；节点函数不直接管理 Runtime State。 | [`node_macro.rs`](../examples/node_macro.rs)、[`workflow_events.rs`](../examples/workflow_events.rs) |
| `plan::Plan`、`reason::Reason`、`evaluation::Evaluator` | Agent 的计划图、推理与业务结果评估；不负责保存 Runtime 现场。 | [`planning_evaluation.rs`](../examples/planning_evaluation.rs) |
| `workflow` | START/END、普通边、条件边、Join、Fan-out、子图和 Workflow Checkpoint 的运行入口。 | [`workflow_events.rs`](../examples/workflow_events.rs)、[`checkpoint.rs`](../examples/checkpoint.rs) |

组件契约与具体实现的使用顺序固定为：先创建或注入实现，再把它传给 Agent/Workflow；没有注入的组件不会隐式启用。
例如 `model::chat::Chat`、`memory`、`checkpoint` 和 `workspace` 都是独立能力，可以按应用需要组合，不能通过一次构造偷偷创建第二套运行时。

## 2. 统一输入、消息和事件

### 2.1 Agent 输入

Chat Agent 和 ReAct Agent 共同使用 `agent::Input`，不分别定义业务输入类型：

```rust
use halo::agent::{Input, Payload};
use serde_json::{Map, json};

let input = Input {
    metadata: Map::from_iter([
        ("chat_id".into(), json!(1001)),
        ("session_id".into(), json!(2001)),
        ("agent_id".into(), json!(3001)),
    ]),
    payload: vec![Payload {
        r#type: "text".into(),
        data: json!({"value": "北京今天的天气怎么样？"}),
        mime_type: None,
    }],
};
```

`metadata` 保存应用身份、语言和其他上下文；`payload` 是用户提交的有序内容。音频、图片和文件仍使用
`type/data/mime_type`，二进制数据应由应用保存后传稳定引用，不直接把 Base64 大对象放入输入。

### 2.2 Message 和 Event

模型上下文使用 `Message`/`Messages`。`Message` 包含 `role`、`content`、`values`、`metadata` 和可选 `usage`。
模型思考统一使用 `content` 中 `type = "think"` 的 Part，普通输出使用 `type = "text"`。

Agent、Workflow 和 Chat Provider 对外使用统一事件：

```rust
pub enum Event {
    Delta { message: Message },
    Complete {
        message: Message,
        usage: Option<model::token::Usage>,
        finish_reason: Option<String>,
    },
}
```

事件是观察流，不是 Runtime State。Workflow 下游从节点 State 读取业务结果，不从事件副本拼装依赖数据。

### 2.3 链路追踪

Heart 使用 `fastrace` 记录 Agent 和 Workflow 的执行层级。应用在启动时选择 Reporter，运行结束或进程退出前调用一次 `flush`：

```rust
use halo::agent::trace::fastrace::{flush, set_reporter, Config, ConsoleReporter};

fn init_trace() {
    set_reporter(ConsoleReporter, Config::default());
}

fn shutdown_trace() {
    flush();
}
```

Reporter 由应用选择，Heart 不把某个后端写进 Runtime 契约。除了控制台 Reporter，还可以传入 fastrace 的 Jaeger 或 OpenTelemetry Reporter。链路中记录运行层级、节点和执行 ID、Delta/Complete、结束原因、Usage、等待关联 ID 和错误码；完整 Message、Checkpoint 和 Runtime State 仍由各自的状态与持久化机制保存。

## 3. Chat Agent

Chat Agent 是普通对话循环。它可以使用已挂载的 Tool、MCP 和 Skill；需要动态计划、迭代和重规划时使用 ReAct Agent。

### 3.1 构造和运行

下面是 Agent 入口的最小结构片段；其中 `chat` 和 `input` 需要由应用按第 6 节初始化。它不是模型实现，也不会自行产生回答。
需要真实大模型的完整程序见[10.1 真实 VLLM Chat Agent](#101-真实-vllm-chat-agent从-input-到-deltacomplete)。

```rust
use futures_util::StreamExt;
use halo::agent::{Cancellation};
use halo::agent;

let agent = agent::chat()
    .model(chat)
    .system_prompt("你是一个可靠的中文助手")
    .build()?;

let mut events = agent.run(input, true, Cancellation::new()).await?;
while let Some(event) = events.next().await {
    println!("{:?}", event?);
}
```

`chat` 是已初始化的 `model::chat::Chat` 实例。`run(input, true, cancellation)` 对外交付 Delta 和 Complete；
传 `false` 时只交付 Complete，但返回类型不变。一次运行中的每次模型调用或工具调用都可以产生自己的 Complete。

取消时仍应继续消费事件流直到收口，以取得取消 Complete 和已产生的 Usage。

### 3.2 挂载能力

```rust
let agent = agent::chat()
    .model(chat)
    .tools(vec![search_tool])
    .mcps(servers)
    .skills(skills)?
    .permission(permission)
    .middleware(middleware)
    .memory(memory_config)
    .checkpoint(checkpoint)
    .build()?;
```

Chat 和 ReAct 都支持这些入口。MCP Server 应在挂载前连接并完成工具发现；运行期间不会临时重新发现工具。
Permission 只在实际 Tool/MCP 调用前检查；Tool 的 `before/call/after/eval` 是工具生命周期，不是 Agent Middleware。

### 3.3 Memory 和 Checkpoint

Memory 由应用配置后传给 Agent，负责短期 Session/Chat（Chat.state 保存最终完整 State）和长期
`summary`/`fact`/`preference`。Checkpoint 保存一次 Agent 运行的完整恢复现场；Chat.state 是收口审计快照，不是恢复来源：

```rust
let agent = agent::chat()
    .model(chat)
    .memory(memory_config)
    .checkpoint(checkpoint)
    .build()?;
```

没有配置 Memory 就不会自动读取历史；没有配置 Checkpoint 就不能跨进程恢复。默认内存实现只在当前进程有效。

### 3.4 Agent Checkpoint：保存终态并重建

Checkpoint 是按 `chat_id` 保存的一份完整 Agent 运行现场。它不只保存最终文本，还保存当前消息、Usage、等待信息、节点状态和必要的执行事实。
下面的代码接着 10.1 中已经创建好的 `chat` 和 `input` 使用：

```rust
use std::sync::Arc;
use futures_util::StreamExt;
use halo::agent::{self, checkpoint::Checkpoint as _, Cancellation};
use halo::agent::core;

let checkpoint = Arc::new(core::checkpoint::InMemory::new());
let agent = agent::chat()
    .model(chat)
    .checkpoint(checkpoint.clone())
    .build()?;

let chat_id = input
    .metadata
    .get("chat_id")
    .and_then(serde_json::Value::as_i64)
    .expect("input.metadata.chat_id is required when Checkpoint is configured");
let mut events = agent.run(input.clone(), false, Cancellation::new()).await?;
while let Some(event) = events.next().await {
    println!("event = {:?}", event?);
}
drop(events);

let saved = checkpoint.load(chat_id, &Cancellation::new()).await?;
println!("saved status = {}", saved["status"]);
```

进程重启后，应用用同一个 `chat_id`、同一个 Checkpoint 后端和重新初始化的 Model/Agent 配置再次调用 `run`；Runtime 会先读取已保存现场。
终态现场直接返回已保存的 Complete，等待现场则由同一个 `reply` 入口继续。`InMemory` 只用于当前进程示例，跨进程时替换为应用自己的持久化 Checkpoint 实现。

## 4. ReAct Agent

ReAct 使用同一个 `Input` 和事件流入口，但内部增加迭代、工具批次和可选的 Agent 计划图：

```rust
let agent = agent::react()
    .model(chat)
    .system_prompt("先判断任务，再选择合适的已挂载能力")
    .tools(vec![search_tool, calculator_tool])
    .mcps(servers)
    .skills(skills)?
    .with_max_iters(20)
    .with_max_concurrency(100)
    .memory(memory_config)
    .checkpoint(checkpoint)
    .build()?;
```

边界如下：

- Agent 决定是否调用工具、调用哪个工具以及参数；
- 挂载能力负责真正执行 Tool、MCP 和 Skill；
- Agent 运行图负责任务依赖、并发、状态和结果保存；
- Tool 失败反馈给当前模型上下文，不伪装成正常业务结果，也不直接把整个 Agent 打成失败；
- Agent `evaluator` 判断最终结果，不替代 Tool 错误处理；
- 计划只能引用已经注册的能力，未知能力在计划更新时返回输入错误。

### 4.1 等待和反馈

审批、人工输入或外部回调会使运行进入 `waiting`。反馈通过同一个 Agent 对象的 `reply` 提交，不另造 `resume` 输入类型：

```rust
agent.reply(
    Input {
        metadata: serde_json::Map::from_iter([
            ("chat_id".into(), serde_json::json!(1001)),
            ("reply_id".into(), serde_json::json!(9001)),
        ]),
        payload: vec![Payload {
            r#type: "json".into(),
            data: serde_json::json!({"approved": true}),
            mime_type: None,
        }],
    },
    Cancellation::new(),
).await?;
```

反馈按等待队列 FIFO 消费；收齐当前等待步骤所需反馈后，原 `run` 流继续输出。原进程结束后，从完整 Checkpoint 重建 Agent，
再提交相同的 reply；已成功的模型、工具和兄弟节点不会重放。

## 5. Workflow

Workflow 是显式图编排，不使用 Agent Planner。节点可以是普通异步函数、Agent 或已编译子图。Runtime 负责图调度、状态、Join、
Fan-out、等待和 Checkpoint。

### 5.1 普通函数节点

普通函数直接交给 `WorkflowBuilder::add_node`；不需要包装类。普通节点的输入输出应为对象根 Schema，`Messages` 是消息型例外：

```rust
use halo::agent::{Cancellation, Error};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct SearchInput { query: String }

#[derive(Serialize, JsonSchema)]
struct SearchOutput { documents: Vec<String> }

async fn search(input: SearchInput, _: Cancellation) -> Result<SearchOutput, Error> {
    Ok(SearchOutput { documents: vec![input.query] })
}
```

Runtime 根据节点的 `attrs.input` 从上游 State 组织 `SearchInput`，每个节点可以有自己的业务输入结构。

### 5.2 构图和运行

```rust
use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages};
use halo::agent::workflow::{graph::{END, START, GraphError}, Runtime, WorkflowBuilder};
use std::sync::{Arc, atomic::{AtomicI64, Ordering}};

fn graph_error(error: GraphError) -> Error {
    Error { code: error.code, message: error.message, details: error.details }
}

let mut builder = WorkflowBuilder::new("search-workflow");
builder.add_node("search", search).map_err(graph_error)?;
builder.add_edge(START, "search").map_err(graph_error)?;
builder.add_edge("search", END).map_err(graph_error)?;
let workflow = builder.compile().map_err(graph_error)?;
let runtime = Runtime::new(1001, workflow)?;
let ids = Arc::new(AtomicI64::new(1));
let mut events = runtime.run(
    Messages::default(),
    Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
    Cancellation::new(),
).await?;
while let Some(event) = events.next().await {
    println!("{:?}", event?);
}
```

ID 生成函数由应用提供，例如雪花 ID；Heart 不初始化 ID 服务。读取 Runtime State 前先消费并释放运行流。

### 5.3 关系、Join 和 Fan-out

```rust
builder.add_conditional_edges("router", route, ["search", "fallback"])?;
builder.add_join(["search", "fallback"], "merge")?;
builder.add_fan_out_edges("planner", "worker")?;
```

普通关系按任一成功来源激活；Join 等待所有声明来源成功；条件关系只能选择预先声明的目标；Fan-out 可以并发执行，恢复时只重试失败子集。

### 5.4 嵌入 Agent 和子图

```rust
builder.add_agent_node("researcher", researcher_agent)?;
builder.add_graph_node("child", &compiled_child_workflow)?;
builder.add_edge(START, "researcher")?;
builder.add_edge("researcher", "child")?;
builder.add_edge("child", END)?;
```

Agent 使用自己的 Agent State，子图保留自己的图边界；父 Workflow 只把它们作为节点调度，不把 Workflow Node State 和 Agent State 混成一个类型。

### 5.5 Workflow 反馈和恢复

Workflow 反馈入口是 `Runtime::reply(Messages, Cancellation)`。原运行仍消费原事件流；进程结束后从完整快照重建：

```rust
let snapshot = checkpoint.load(1001, &Cancellation::new()).await?;
let runtime = Runtime::from_snapshot(1001, workflow, snapshot)?;
runtime.reply(feedback, Cancellation::new()).await?;
```

恢复加载完整 Runtime State：已完成节点、失败标记、等待队列、图版本和业务数据都来自快照，不从头重放模型、工具和节点。

## 6. Model Provider

Provider 初始化参数和单次调用参数分开。当前 OpenAI、DeepSeek 和 VLLM 使用 Chat Completions；不实现 Responses API。

```rust
use halo::agent::core::provider::vllm::Vllm;
use halo::agent::model::chat;
use serde_json::{Map, json};

let provider = Vllm::builder()
    .base_url("http://127.0.0.1:8000/v1")
    .api_key(std::env::var("HEART_VLLM_API_KEY")?)
    .build()?;
let chat = provider.chat(
    chat::Config::new("Qwen3.5-122B-A10B").options(Map::from_iter([(
        "chat_template_kwargs".into(),
        json!({"enable_thinking": true}),
    )])),
)?;
```

VLLM 的 `delta.reasoning` 转成统一的 `think` Part，普通输出转成 `text` Part。供应商特有参数放在 `Config.options` 或 `Request.options`。

DeepSeek 使用自己的 Provider 类型，模型名也只放在 DeepSeek 的 Chat 配置中：

```rust
let provider = halo::agent::core::provider::deepseek::DeepSeek::builder()
    .base_url("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .build()?;
let chat = provider.chat(halo::agent::model::chat::Config::new("deepseek-flash"))?;
```

VLLM 示例中的 Qwen 模型属于 VLLM 服务，不与 DeepSeek Provider 混用。

智谱 Embedding-3 是独立的 Embed Provider，不依赖 Chat Provider。默认请求地址为智谱 V4 API，`dimensions` 可省略（服务默认 2048），或在 256–2048 范围内显式设置：

```rust
use halo::agent::{
    model::embed::{self, Embed as _},
    provider::zhipu::Zhipu,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = Zhipu::builder()
        .api_key(std::env::var("HEART_EMBED_API_KEY")?)
        .build()?;
    let embed = provider.embed(embed::Config {
        model_name: "embedding-3".into(),
        options: Default::default(),
    })?;
    let message = embed
        .embed(embed::Request {
            input: "语义搜索的测试文本".into(),
            dimensions: Some(1024),
            options: Default::default(),
        })
        .await?;
    let vector = message.values.get("vector").expect("embedding vector");
    println!("dimension={}, usage={:?}", vector.as_array().unwrap().len(), message.usage);
    Ok(())
}
```

本地 `.env` 至少配置 `HEART_EMBED_API_KEY`；如使用兼容网关或其他模型名，可另配 `HEART_EMBED_BASE_URL`、`HEART_EMBED_MODEL`。不要提交密钥。

OpenAI 的 Embeddings API 也已由 `core::provider::openai::OpenAI` 提供，无需额外 Provider 类型。默认地址为 `https://api.openai.com/v1`，例如使用 `text-embedding-3-small`；请求维度通过 `embed::Request.dimensions` 传入。配置 `OPENAI_API_KEY` 后，`scripts/test-real.sh` 会执行真实 OpenAI 向量验收。

## 7. Workspace、Skill 和 MCP

Workspace 决定执行环境是本地文件还是服务器数据库。Local Workspace 默认提供列目录、读取、写入、精确编辑、Glob、Grep 和单文件删除；
命令执行必须显式配置 Sandbox。

```rust
let workspace = halo::agent::core::workspace::Local::new("./workspace")?
    .sandbox(halo::agent::core::sandbox::Local::default())
    .tools(custom_tool)
    .mcps(servers)
    .skills(skills)?;
```

Skill 使用 `.skills(loader)` 载入：Local Loader 从目录发现，Server Loader 从数据库读取。Loader 可以覆盖 `version` 返回数据库修订号或内容指纹；
Workspace 会在同一实例内复用已发现的 Skill 快照，只有版本变化才重新读取。MCP Server 必须先连接并完成工具发现，
再挂载到 Workspace 或 Agent。Provider Key 和 MCP 地址只放本地 `.env`，不要提交到 Git。

## 8. 取消、错误和测试

- `Cancellation` 是一次独立运行的取消范围；每个独立 Agent/Workflow 运行创建新的实例。
- `Complete` 是一次模型、工具或节点的可观察结果，不等同于整个 Runtime 成功。
- Checkpoint 是恢复真相；Chat.state 是最终审计快照。
- 工具错误保留为工具错误并反馈给当前 Agent，不伪装成正常结果。

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

## 9. 当前版本边界

PDF/Office/图片的通用解析默认使用 LiteParse；领域专用 Parser 方法、Python 函数接入、TTS 和完整多模态契约属于后续版本。具体格式和运行依赖见 [Document 解析手册](DOCUMENT.zh-CN.md)。

## 10. 完整可运行配方

前面的片段用于解释概念；本节把每个常用入口都写成可以直接复制到一个新 Cargo 项目的
`src/main.rs` 的完整程序。先建立项目并把 Heart 作为依赖：

```bash
cargo new heart-guide-demo
cd heart-guide-demo
```

将本手册第 10 节中的任意一个程序保存为 `src/main.rs`，再执行：

```bash
cargo run
```

示例只使用当前公开契约。需要网络、凭据或外部服务的示例会在标题中明确标出；离线示例中的本地 Chat 仅用于测试 Agent 契约，不代表大模型能力。

### 10.1 真实 VLLM Chat Agent：从 Input 到 Delta/Complete

下面使用真实的 OpenAI-compatible VLLM Chat Completions 服务，不伪造模型输出。先设置服务地址、密钥和模型名：

```bash
export HEART_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export HEART_VLLM_API_KEY=replace-me
export HEART_VLLM_MODEL=Qwen3.5-122B-A10B
```

这个例子展示：

1. `Vllm::new` 初始化 Provider，`provider.chat(...)` 绑定具体模型；
2. Agent 的输入仍然是 `Input`，由 Agent 内部组织模型 `Messages`；
3. `stream = true` 同时观察模型返回的 `Delta` 和 `Complete`；
4. 模型的思考内容由 Provider 统一映射为 `think`，普通回答映射为 `text`。

```rust
use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Input, Payload};
use halo::agent::{event::Event, model::chat, provider::vllm::{Config, Vllm}};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), halo::agent::Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".into()),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("HEART_VLLM_API_KEY is required"),
        ..Config::default()
    })?;
    let chat = provider.chat(chat::Config {
        model_name: std::env::var("HEART_VLLM_MODEL").expect("HEART_VLLM_MODEL is required"),
        options: serde_json::Map::from_iter([
            ("chat_template_kwargs".into(), json!({"enable_thinking": true})),
        ]),
    })?;
    let agent = agent::chat()
        .model(chat)
        .system_prompt("你是一个简洁的中文助手")
        .build()?;

    let input = Input {
        metadata: serde_json::Map::from_iter([
            ("chat_id".into(), json!(1001)),
            ("session_id".into(), json!(2001)),
            ("agent_id".into(), json!(3001)),
        ]),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value": "北京今天适合出门吗？"}),
            mime_type: None,
        }],
    };

    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => println!("delta = {:?}", message.content),
            Event::Complete { message, usage, finish_reason } => {
                println!("complete = {:?}", message.content);
                println!("usage = {usage:?}, finish_reason = {finish_reason:?}");
            }
        }
    }
    Ok(())
}
```

### 10.2 ReAct：工具、Permission 和工具错误反馈

下面用确定性的模型替身模拟“第一次返回工具调用、第二次读取工具结果”的闭环，不代表真实模型推理。真实模型只需把模型替身替换为 10.1 中初始化的 Provider Chat。工具本身实现 `Tool` 契约；
Permission 只在真正执行工具时检查，模型只负责选择工具和参数。把 `.tools(Lookup)` 改成多个工具时可以传
`vec![Lookup, OtherTool]`，不需要额外的工具集合公共对象。

```rust
use std::sync::{LazyLock, atomic::{AtomicUsize, Ordering}};
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error};
use halo::agent::{
    event::Event,
    message::{Message, Role},
    model::{chat::{Chat, Request}, content::part::Part},
    permission::{Behavior, Decision, Request as PermissionRequest},
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::json;

struct Lookup;
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "lookup".into(),
            description: "读取一个固定的天气结果".into(),
            parameters: serde_json::from_value(json!({
                "type": "object", "properties": {}, "additionalProperties": false
            })).unwrap(),
        });
        &DEFINITION
    }

    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        Ok(Message::function(json!({"temperature": 22, "condition": "晴"})))
    }
}

struct DemoModel(AtomicUsize);
impl Chat for DemoModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut output = Message::new(Role::Assistant);
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            output.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"call_1","name":"lookup","arguments":{}})
                    .as_object().unwrap().clone(),
            });
        } else {
            let saw_tool = request.messages.as_slice().iter()
                .any(|message| message.role == Role::Tool);
            output.values.insert("answer".into(), json!(if saw_tool {
                "北京 22 度，天气晴朗。"
            } else {
                "没有工具结果。"
            }));
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: output,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let permission = halo::agent::core::permission::Permission::new(
        |request: PermissionRequest, _cancel: Cancellation| async move {
            if request.action == "call" && request.resource == "lookup" {
                Ok(Decision { behavior: Behavior::Allow, reason: None })
            } else {
                Ok(Decision { behavior: Behavior::Deny, reason: Some("not allowed".into()) })
            }
        },
    );

    let agent = agent::react()
        .model(DemoModel(AtomicUsize::new(0)))
        .tools(Lookup)
        .permission(permission)
        .with_max_iters(20)
        .build()?;
    let mut events = agent.run(agent::Input::default(), true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
```

如果 `Tool::call` 返回错误，ReAct 会把原始错误作为当前模型下一轮可见的工具错误上下文；这不是一个成功的业务
结果，也不会把整个 Agent 直接改成失败。`Tool::eval` 只有在用户实现它时才判断业务结果，失败后由当前 Agent 的
规划/执行流程决定是否继续。

### 10.3 Workflow：普通节点、State 和事件流

这个例子是一个完整的单节点 Workflow。它展示了 `START/END`、应用提供的执行 ID 函数、事件流和最终 State。
注意：必须在事件流被消费并释放后再读取 `runtime.state()`。

```rust
use std::sync::{Arc, atomic::{AtomicI64, Ordering}};
use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages, event::Event, workflow::{GraphError, Runtime, WorkflowBuilder, graph::{START, END}}};
use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, JsonSchema)]
struct Answer { answer: i64 }

fn graph_error(error: GraphError) -> Error {
    Error { code: error.code, message: error.message, details: error.details }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let mut builder = WorkflowBuilder::new("answer-workflow");
    builder.add_node("answer", |_: Messages, _: Cancellation| async {
        Ok::<_, Error>(Answer { answer: 42 })
    }).map_err(graph_error)?;
    builder.add_edge(START, "answer").map_err(graph_error)?;
    builder.add_edge("answer", END).map_err(graph_error)?;

    let workflow = builder.compile().map_err(graph_error)?;
    let runtime = Runtime::new(1001, workflow)?;
    let ids = Arc::new(AtomicI64::new(1));
    let mut events = runtime.run(
        Messages::default(),
        Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
        Cancellation::new(),
    ).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => println!("delta = {:?}", message.content),
            Event::Complete { message, usage, .. } => println!("complete = {:?}, usage = {usage:?}", message.content),
        }
    }
    drop(events);
    println!("workflow state = {:?}", runtime.state());
    Ok(())
}
```

普通节点的输入输出是强类型对象；只有消息型节点才直接使用 `Messages`。Runtime 按节点的 `attrs.input` 从上游
State 读取字段，不把多个节点结果偷偷拼成一个新的全局消息。

### 10.4 Workflow 图关系：条件、Join、Fan-out 和子图

图的关系在编译期声明，执行时才产生本次运行的 `State`。下面的程序只用 `Messages` 作为节点 I/O，重点演示四种
编排动作；它会编译并打印版本，便于在应用构图阶段先验证非法边、重复边和 Join 冲突。

```rust
use halo::agent::{Cancellation, Error, Messages, workflow::{GraphError, WorkflowBuilder, graph::{START, END}}};

fn graph_error(error: GraphError) -> Error { Error { code: error.code, message: error.message, details: error.details } }

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut builder = WorkflowBuilder::new("relations");
    for id in ["router", "left", "right", "merge", "worker"] {
        builder.add_node(id, |input: Messages, _cancel: Cancellation| async move {
            Ok::<_, Error>(input)
        }).map_err(graph_error)?;
    }
    builder.add_edge(START, "router").map_err(graph_error)?;
    builder.add_conditional_edges(
        "router",
        |_messages| Ok(vec!["left".to_owned(), "right".to_owned()]),
        ["left", "right"],
    ).map_err(graph_error)?;
    builder.add_join(["left", "right"], "merge").map_err(graph_error)?;
    builder.add_fan_out_edges("merge", "worker").map_err(graph_error)?;
    builder.add_edge("worker", END).map_err(graph_error)?;
    let compiled = builder.compile().map_err(graph_error)?;
    println!("definition_id = {}, version = {}", compiled.definition().definition_id, compiled.definition().version);
    Ok(())
}
```

语义要点：普通边不等待其他前驱；Join 必须等所有声明来源成功；条件边只能返回声明过的目标；Fan-out 的子项可以
并发，但恢复时只重试失败子集。若要把一个已编译 Workflow 嵌入父图，使用 `add_graph_node("child", &child)`，不要把
子图伪装成普通函数节点。

### 10.5 Workflow Checkpoint：保存现场后恢复

Checkpoint 保存的是完整 Runtime State，而不是 Chat.state 审计快照。生产环境通常让 Runtime 在成功或最终异常收口时写一次快照；示例使用
内存后端演示同一进程内的保存和恢复，跨进程时只需替换 `core::checkpoint::InMemory`。

```rust
use std::sync::{Arc, atomic::{AtomicI64, Ordering}};
use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages, core, workflow::{GraphError, Runtime, WorkflowBuilder, graph::{START, END}}};
use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, JsonSchema)]
struct Output { value: String }
fn graph_error(e: GraphError) -> Error { Error { code: e.code, message: e.message, details: e.details } }

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut builder = WorkflowBuilder::new("checkpoint-demo");
    builder.add_node("done", |_: Messages, _: Cancellation| async {
        Ok::<_, Error>(Output { value: "saved".into() })
    }).map_err(graph_error)?;
    builder.add_edge(START, "done").map_err(graph_error)?;
    builder.add_edge("done", END).map_err(graph_error)?;
    let workflow = builder.compile().map_err(graph_error)?;
    let checkpoint = core::checkpoint::InMemory::new();
    let runtime = Runtime::new(7001, workflow.clone())?.with_checkpoint(checkpoint);
    let ids = Arc::new(AtomicI64::new(1));
    let mut events = runtime.run(Messages::default(), Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)), Cancellation::new()).await?;
    while let Some(_event) = events.next().await {}
    drop(events);
    let snapshot = runtime.snapshot()?;
    println!("snapshot status = {}", snapshot["status"]);

    // Persisted deployments normally load this JSON from the Checkpoint backend.
    let mut restored = Runtime::from_snapshot(7001, workflow, snapshot)?;
    println!("restored status = {:?}", restored.state().status);
    Ok(())
}
```

如果进程在等待人工反馈时结束，重新创建 Runtime 后调用 `reply(Messages, Cancellation)`；如果只是同一进程的活跃运行，
直接调用原 Runtime 的 `reply`，不要再创建第二个运行流。

### 10.6 VLLM：真实 Chat Completions 调用

这是唯一需要外部服务和密钥的基础示例。把配置写进项目根目录的 `.env`（该文件已经被 `.gitignore` 忽略），例如：

```dotenv
HEART_VLLM_BASE_URL=http://127.0.0.1:8000/v1
HEART_VLLM_API_KEY=replace-me
HEART_VLLM_MODEL=Qwen3.5-122B-A10B
```

运行前把变量导入当前 Shell：

```bash
set -a; source .env; set +a
cargo run
```

将下面完整程序保存为 `src/main.rs`：

```rust
use futures_util::StreamExt;
use halo::agent::{event::Event, message::{Message, Messages, Role}, model::{chat, chat::Chat, content::part::Part}, core::provider::vllm::Vllm};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), halo::agent::model::Error> {
    let provider = Vllm::builder()
        .base_url(std::env::var("HEART_VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".into()))
        .api_key(std::env::var("HEART_VLLM_API_KEY").map_err(|_| halo::agent::model::Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required"))?)
        .build()?;
    let chat = provider.chat(chat::Config::new(std::env::var("HEART_VLLM_MODEL").map_err(|_| halo::agent::model::Error::new("INVALID_ARGUMENTS", "HEART_VLLM_MODEL is required"))?).options(serde_json::Map::from_iter([("chat_template_kwargs".into(), json!({"enable_thinking": true}))])))?;
    let mut user = Message::new(Role::User);
    user.content.push(Part { r#type: "text".into(), data: json!({"value":"介绍平路运河"}).as_object().unwrap().clone() });
    let mut events = chat.stream(chat::Request { messages: Messages::new([user]), stream: true, max_tokens: Some(512), ..Default::default() }).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => for part in message.content { print!("{}", part.data["value"].as_str().unwrap_or_default()); },
            Event::Complete { usage, finish_reason, .. } => println!("\nusage={usage:?}, finish_reason={finish_reason:?}"),
        }
    }
    Ok(())
}
```

当前 VLLM 适配的是 `/v1/chat/completions`，不是 Responses API。VLLM 的 `delta.reasoning` 对外统一为 `think`，普通输出统一为
`text`；Provider 初始化参数和单次 `chat::Request` 参数不能混用。

### 10.7 Memory：短期记录、长期记忆和显式召回

下面的程序完全离线运行，先写入 Session、Chat 和一条长期 Preference，再构造 Agent。它说明 Memory 不会在没有配置时自动启用，
短期数据和长期数据也不是同一张记录。

```rust
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::agent::{
    self, Agent, Input, Payload,
    core,
    event::Event,
    memory::{self, Memory},
    memory::short::{self, chat, session},
    message::Role,
    model::{chat::{Chat, Request}, content::part::Part},
};
use serde_json::{Map, json};
use memory::short::chat::Store as _;
use memory::short::session::Store as _;

struct Model;
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _request: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part { r#type: "text".into(), data: json!({"value":"晴天，适合露营。"}).as_object().unwrap().clone() });
        Ok(stream::iter(vec![Ok(Event::Complete { message, usage: None, finish_reason: Some("stop".into()) })]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let cancellation = Cancellation::new();
    let short = core::memory::short::InMemory::new();
    let long = core::memory::InMemory::new();
    let config = memory::Config::new(short, long).with_scopes(["user", "agent"]);
    config.session().create(session::Session { id: 1, tenant_id: 1, user_id: 2, r#type: short::Type::Agent, title: Some("天气".into()), summary: None, usage: None, status: session::Status::Active, metadata: Map::new(), created_time: 0, updated_time: 0 }, &cancellation).await?;
    config.chat().create(chat::Chat { id: 2, tenant_id: 1, user_id: 2, agent_id: 3, session_id: 1, r#type: short::Type::Agent, status: chat::Status::Completed, input: Some(chat::Request { metadata: Map::new(), payload: vec![chat::Payload { r#type: "text".into(), data: json!({"value":"北京天气？"}), mime_type: None }] }), message: None, usage: None, graph: None, state: None, metadata: Map::new(), started_time: None, completed_time: None, created_time: 0, updated_time: 0 }, &cancellation).await?;
    config.long().write(memory::Request { id: 3, r#type: memory::Type::Preference, user_id: 2, agent_id: None, session_id: None, from: 1, to: 2, content: "用户喜欢露营".into(), metadata: None, vector: None }, &cancellation).await?;
    let agent = agent::chat().model(Model).memory(config).build()?;
    let input = Input { metadata: Map::from_iter([("chat_id".into(), json!(3)), ("session_id".into(), json!(1)), ("user_id".into(), json!(2)), ("agent_id".into(), json!(3))]), payload: vec![Payload { r#type: "text".into(), data: json!({"value":"天气好就去露营"}), mime_type: None }] };
    let mut events = agent.run(input, false, Cancellation::new()).await?;
    while let Some(event) = events.next().await { println!("{:?}", event?); }
    Ok(())
}
```

压缩是独立能力：只有应用或 Harness 配置了压缩组件并主动触发检查，才会用摘要替换已覆盖的 Chat 范围；本次请求不会中途
切换到新摘要。

内置 Chat 压缩器可以限制摘要输入和保留的近期历史：

```rust
let compressor = core::compression::Compressor::new(chat, request)?
    .with_size(10_000)  // 达到该阈值才启动后台压缩
    .with_limit(8_000)  // 摘要模型最多接收的历史 Token 估算
    .with_retain(1_000); // 留给后续请求继续使用的最新未覆盖历史
```

默认使用粗略 Token 估算；需要时可以通过 `.with_count(...)` 传入应用自己的估算函数。`limit` 和 `retain` 只作用于摘要输入视图，原始 Chat、Tool 结果和 Runtime Step 不会被截断。

### 10.8 Workspace 和 Skill：本地发现、内容读取和安全边界

Workspace 的本地实现只操作它的根目录。以下程序创建临时目录、写入一个标准 `SKILL.md`、读取 Skill 内容并列出目录；它不执行
Skill 中的脚本，也不自动赋予 Skill 工具权限。

```rust
use halo::agent::{Cancellation, Error, core, skill::Skill as _, workspace::Workspace};

#[tokio::main]
async fn main() -> Result<(), Error> {
    let root = std::env::temp_dir().join(format!("heart-guide-{}", std::process::id()));
    std::fs::create_dir_all(&root).map_err(|e| Error::new("IO_ERROR", e.to_string()))?;
    let workspace = core::workspace::Local::new(&root)?;
    let cancel = Cancellation::new();
    workspace.write("research/SKILL.md", br#"---
name: research
description: 查找资料并列出来源
---
# Instructions
先确认来源，再给出结论。
"#.to_vec(), &cancel).await?;
    let paths = workspace.list(".", true, &cancel).await?;
    let skill = core::skill::Skill::load(&workspace, "research/SKILL.md", &cancel).await?;
    println!("workspace mode = {:?}", workspace.definition().mode);
    println!("files = {paths:?}");
    println!("skill = {:?}", skill.read(&cancel).await?);
    let _ = std::fs::remove_dir_all(root);
    Ok(())
}
```

需要让 Agent 使用这个 Skill 时，在构造 Agent 时调用 `.skills([skill])?`；需要目录发现时实现 `skill::Loader`，再通过
`.skills(loader)` 挂载。模型可通过共享 `skill` 阅读入口显式给出相对 `resource` 路径读取引用的文本资源；不会自动跟随引用或执行脚本。Server Workspace 不假装存在本地路径，而是由后端 Loader 从数据库读取同样的定义和资源。

### 10.9 MCP：先连接和发现，再挂载

MCP 连接不是运行中临时发现。下面是完整的初始化形态；如果网络不可用，程序会在 `Server::http` 返回统一错误，Agent 不会被构造出来。

```rust
use futures_util::stream;
use halo::agent::{Cancellation, Error, Message};
use halo::agent::{
    self,
    event::Event,
    mcp::{Definition, Server as _},
    message::Role,
    model::chat::{Chat, Request},
};
use serde_json::Map;

struct MyChat;
impl Chat for MyChat {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _request: Request) -> Result<Self::Stream, Error> {
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: Message::new(Role::Assistant), usage: None, finish_reason: Some("stop".into())
        })]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let endpoint = std::env::var("HEART_MCP_SSE_URL")
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_MCP_SSE_URL is required"))?;
    let server = halo::agent::core::mcp::Server::http(
        Definition { name: "weather".into(), description: Some("天气 MCP".into()), metadata: Map::new() },
        &endpoint,
        Map::new(),
        &Cancellation::new(),
    ).await?;
    println!("discovered tools = {:?}", server.tools().iter().map(|t| &t.name).collect::<Vec<_>>());
    let _agent = agent::chat().model(MyChat).mcps([server]).build()?;
    Ok(())
}

// 这里的 MyChat 只是让程序在 MCP 连接成功后仍能完整构造 Agent；真实应用替换为 VLLM/OpenAI Chat。
```

不要用内存 `Server` 替身把连接失败伪装成成功。需要本地验证 MCP 协议时，项目提供了一个独立 stdio MCP 服务进程：它的 `current_time` 工具读取真实系统时钟，测试通过 Heart 的 MCP 客户端发现并调用该工具，不使用预置业务响应：

```bash
cargo test --offline --test mcp_stdio_acceptance -- --nocapture
cargo test --offline --test mcp_http_acceptance -- --nocapture
```

HTTP 验收测试会启动本地 Streamable HTTP MCP 服务，通过 Heart 的 HTTP 客户端执行初始化、工具发现和真实工具调用，并校验返回时间戳与系统时钟一致。

MCP 验收使用项目自带的独立 stdio 服务进程和本地 Streamable HTTP 服务，不依赖远端 MCP 地址。Provider Key 等私密配置只放本地 `.env`，不能提交。

### 10.10 Prompt、压缩、Permission 和 Middleware 的组合

这些是可选组件，不配置就不产生对应能力。实际项目可以先用确定性模型验证生命周期，再换成 VLLM：

```rust
let agent = agent::chat()
    .model(chat)
    .system_prompt("只回答用户真正提出的问题")
    .permission(permission)
    .middleware(middleware)
    .memory(memory_config)
    .checkpoint(checkpoint)
    .build()?;
```

`Middleware::before_model` 可调整本次模型请求，`after_execute` 可检查或补充完成消息；Tool 自身的 `before/after/eval` 只属于该 Tool
调用。重试、取消和并发限制由 Runtime/Agent 控制，不由业务 Tool 私自重跑。

## 11. 验证清单

复制示例后，建议按这个顺序验证，先排除格式问题，再接入网络服务：

```bash
cargo fmt --all --check
cargo check --offline --lib --all-targets
cargo check --offline --examples
cargo test --offline --lib
```

真实模型验收必须调用实际 Provider，不用固定模型响应替代。配置好本地 `.env` 后，可运行 Agent 工具调用和父子 Agent 委派的真实测试：

```bash
./scripts/test-real.sh
```

该脚本从未提交的本地 `.env` 读取 Provider 凭据，不输出密钥；它逐项运行真实 Provider、Planner、Evaluator、Compressor、长期记忆提取/合并、Chat/ReAct Agent 的真实工具调用、父子 Agent、Agent 用户反馈恢复、流式/非流式 Agent、Chat 持久化收口、Workflow 内嵌 Agent、Skill、以及本地 MCP stdio/HTTP 服务验收。配置 `HEART_EMBED_API_KEY` 后，会使用智谱 Embedding-3 运行 Semantic Chunker 的真实向量验收；`HEART_EMBED_BASE_URL` 和 `HEART_EMBED_MODEL` 可选，默认分别为 `https://open.bigmodel.cn/api/paas/v4` 和 `embedding-3`。未配置 API Key 时会明确跳过，不以模拟向量代替。配置 `HEART_ES_URL` 与隔离测试索引后，也会运行真实 Elasticsearch 测试；配置 `HEART_WORKSPACE_TEST_IMAGE` 后，会运行真实 Docker Workspace 测试。它不连接远端 MCP 地址。也可以按需单独执行下方的测试命令。

真实验收中的 Provider、Planner、Evaluator、Compressor、记忆提取/合并、Chat/ReAct Agent 工具调用、父子 Agent 委派、MCP 调用、Skill、用户反馈恢复和 Workflow 内嵌 Agent 都使用真实 Provider/实际运行数据；模型没有执行预期调用、Provider 出错或外部服务不可用都应判失败，不能用预置模型响应补成通过。DeepSeek 验收需在 `.env` 配置 `DEEPSEEK_API_KEY`；`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 可选，默认分别为官方 API 地址和 `deepseek-flash`。
`component_real_acceptance` 验证真实 Planner 产出完整 DAG、Reasoner 返回 Provider Complete、Evaluator 对最终结果的判断、真实 Chat Agent 完成后由真实 Evaluator 复核、Extractor 的长期记忆结构化提取与合并，以及 Compressor 的真实 Complete 摘要。
`workspace_skill_real_acceptance` 使用本地真实 `SKILL.md`，检查 Harness/Workspace 发现、模型按需读取 Skill 并依据其内容作答；`workflow_real_acceptance` 检查 Workflow 内嵌 Agent 节点调用真实 DeepSeek 并保存执行状态与 Usage。
`agent_feedback_real_acceptance` 由真实 DeepSeek 选择等待反馈的 Tool，测试用户通过 `reply_id` 回答后同一 Agent 继续运行并输出最终结果。
`agent_stream_real_acceptance` 使用真实 DeepSeek 分别验证 `stream=true` 会发出 Delta/Complete、`stream=false` 只发出 Complete。
`agent_chat_closeout_real_acceptance` 中的真实收口测试还验证真实 Provider 返回的最终 Message 和 Usage 被写入 Chat 状态。
配置 `HEART_EMBED_API_KEY` 后，`semantic_chunking_real_acceptance` 会调用智谱 Embedding-3 并用实际向量验证 Semantic Chunker；可选通过 `HEART_EMBED_BASE_URL`、`HEART_EMBED_MODEL` 覆盖默认地址和模型。
默认 `cargo test` 还包含离线单元与状态机测试，用于直接校验状态转移、输入校验和事件协议，不会合成成功的 Chat 或 Embed 响应。这些测试不等同于真实模型端到端验收，也不会替代上述真实验收脚本。
Docker Workspace 的真实容器、挂载和取消测试不会自动拉取镜像；指定已经下载的镜像后显式运行：

```bash
export HEART_WORKSPACE_TEST_IMAGE=docker.elastic.co/elasticsearch/elasticsearch:9.5.4
cargo test --offline --test workspace_docker docker_workspace_real_container_smoke -- --ignored --nocapture
cargo test --offline --test workspace_docker docker_workspace_real_bind_mount_smoke -- --ignored --nocapture
cargo test --offline --test workspace_docker docker_workspace_real_cancellation_smoke -- --ignored --nocapture
```

若 `cargo test --all-targets` 报缺少被忽略的测试 fixture，说明本地测试目录不是发布内容，不要把 `.env`、测试数据或设计文档强行加入源码提交。
