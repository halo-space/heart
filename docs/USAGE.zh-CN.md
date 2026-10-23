# Heart 使用手册

本手册介绍如何在 Rust 应用中使用 Heart 构建 Agent 和 Workflow。示例按当前公开 API 编写。

[English version](USAGE.en.md)

## 1. 安装

Heart 的 Cargo 包名是 `halo-agents`，Rust crate 名是 `halo_agents`，要求 Rust 1.99.1 或更新版本。

```toml
[dependencies]
halo-agents = { git = "https://github.com/halo-space/heart.git" }
futures-util = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "0.8"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

目录职责：

```text
components/   公共契约：Agent、Model、Message、Event、Tool、Memory、Checkpoint
core/         具体实现：Provider、Chat/ReAct、Workspace、Sandbox、Memory 后端
runtime/      Agent 与 Workflow 的状态、图和执行控制
harness/      面向长任务 Agent 的高阶组合能力
```

没有配置的能力不会自动启用：没有 Model 不能运行 Agent，没有 Sandbox 就没有命令工具，没有 Checkpoint 就不能跨进程恢复。

## 2. 统一输入、消息和事件

### 2.1 Agent 输入

Chat Agent 和 ReAct Agent 共同使用 `agent::Input`，不分别定义业务输入类型：

```rust
use halo_agents::agent::{Input, Payload};
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

## 3. Chat Agent

Chat Agent 是普通对话循环。它可以使用已挂载的 Tool、MCP 和 Skill；需要动态计划、迭代和重规划时使用 ReAct Agent。

### 3.1 构造和运行

```rust
use futures_util::StreamExt;
use halo_agents::{agent, Cancellation};

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

Memory 由应用配置后传给 Agent，负责短期 Session/Chat/Trace 和长期 `summary`/`fact`/`preference`。
Checkpoint 保存一次 Agent 运行的完整恢复现场，不是 Trace 审计副本：

```rust
let agent = agent::chat()
    .model(chat)
    .memory(memory_config)
    .checkpoint(checkpoint)
    .build()?;
```

没有配置 Memory 就不会自动读取历史；没有配置 Checkpoint 就不能跨进程恢复。默认内存实现只在当前进程有效。

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
use halo_agents::{Cancellation, Error};
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
use halo_agents::{Cancellation, Error, Messages};
use halo_agents::runtime::workflow::{graph::{END, START, GraphError}, Runtime, WorkflowBuilder};
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
use halo_agents::provider::vllm::{Config, Vllm};
use halo_agents::model::chat;
use serde_json::{Map, json};

let provider = Vllm::new(Config {
    provider_id: 1,
    provider_name: "vllm".into(),
    base_url: "http://127.0.0.1:8000/v1".into(),
    api_key: std::env::var("HEART_VLLM_API_KEY")?,
    ..Default::default()
})?;
let chat = provider.chat(chat::Config {
    model_name: "Qwen3.5-122B-A10B".into(),
    options: Map::from_iter([(
        "chat_template_kwargs".into(),
        json!({"enable_thinking": true}),
    )]),
})?;
```

VLLM 的 `delta.reasoning` 转成统一的 `think` Part，普通输出转成 `text` Part。供应商特有参数放在 `Config.options` 或 `Request.options`。

## 7. Workspace、Skill 和 MCP

Workspace 决定执行环境是本地文件还是服务器数据库。Local Workspace 默认提供列目录、读取、写入、精确编辑、Glob、Grep 和单文件删除；
命令执行必须显式配置 Sandbox。

```rust
let workspace = halo_agents::core::workspace::Local::new("./workspace")?
    .sandbox(halo_agents::sandbox::Local::default())
    .tools(custom_tool)
    .mcps(servers)
    .skills(skills)?;
```

Skill 使用 `.skills(loader)` 载入：Local Loader 从目录发现，Server Loader 从数据库读取。MCP Server 必须先连接并完成工具发现，
再挂载到 Workspace 或 Agent。Provider Key 和 MCP 地址只放本地 `.env`，不要提交到 Git。

## 8. 取消、错误和测试

- `Cancellation` 是一次独立运行的取消范围；每个独立 Agent/Workflow 运行创建新的实例。
- `Complete` 是一次模型、工具或节点的可观察结果，不等同于整个 Runtime 成功。
- Checkpoint 是恢复真相；Trace 是最终审计投影。
- 工具错误保留为工具错误并反馈给当前 Agent，不伪装成正常结果。

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

## 9. 当前版本边界

完整 PDF/Office/OCR Parser、Python 函数接入、TTS 和完整多模态契约属于后续版本。本手册只描述当前公开的 Rust 能力。
