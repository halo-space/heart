# Heart 使用手册

Heart 是一个 Rust Agent/Workflow 执行库。它提供模型、工具、MCP、Skill、Workspace、Memory、Checkpoint、压缩、权限、文档和协议适配能力；应用可以把这些能力嵌入自己的产品，运行时负责现场、状态、事件和恢复，Server 层负责按需提供 HTTP 接入。

当前版本是 Alpha `0.1.x`，要求 Rust `1.99.1` 或更高版本。Cargo 包名是 `halo-agent`，Rust 导入名是 `halo`。

完整可运行示例在 [`examples/`](../examples/)；本手册只展示最小代码，遇到参数细节应直接运行对应示例。

---

## 架构和安装

### 0.1 分层

Heart 的单 crate 分层固定为：

```text
components  -> 公共契约和数据结构
core        -> 组件的具体实现
runtime     -> Agent/Workflow 的状态、图和异步执行
harness     -> 高阶组合，如历史、压缩、权限和生命周期
server      -> HTTP、A2A、JSON-RPC、SSE 和 Webhook 外层绑定
```

应用只需要从 `halo::agent` 进入：

```rust
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload};
```

不要从 `halo::core::agent` 或内部 `runtime` 路径接入 Agent。`core` 是实现层，`runtime` 是执行层；应用通过 `halo::agent::chat()`、`halo::agent::react()` 和 `halo::agent::workflow` 使用它们。

### 0.2 添加依赖

```toml
[dependencies]
halo-agent = "0.1"
futures-util = "0.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

项目要求 Rust `1.99.1`：

```bash
rustup override set 1.99.1
cargo check
```

### 0.3 真实模型的环境变量

密钥只通过环境变量传入，不写入源码和 README：

```bash
export DEEPSEEK_API_KEY='...'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
export DEEPSEEK_MODEL='deepseek-flash'

# VLLM/OpenAI-compatible 服务
export HEART_VLLM_BASE_URL='http://127.0.0.1:8000/v1'
export HEART_VLLM_API_KEY='...'
export HEART_VLLM_MODEL='Qwen3.5-122B-A10B'
```

真实 Provider 示例：[`provider_deepseek.rs`](../examples/provider_deepseek.rs)、[`provider_vllm.rs`](../examples/provider_vllm.rs)。

---

## 第一次运行

一次最小的真实 Chat Agent 由四步组成：创建 Provider、绑定 Chat、构建 Agent、消费事件流。

```rust
use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload};
use halo::agent::event::Event;
use halo::agent::model::chat;
use halo::agent::provider::deepseek::DeepSeek;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let provider = DeepSeek::builder()
        .base_url(std::env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".into()))
        .api_key(std::env::var("DEEPSEEK_API_KEY").map_err(|_| Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required"))?)
        .build()?;
    let chat = provider.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    let agent = agent::chat().model(chat).build()?;
    let input = Input {
        metadata: Default::default(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value": "用一句话介绍 Heart"}),
            mime_type: None,
        }],
    };
    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => println!("delta: {message:?}"),
            Event::Complete { message, usage, .. } => println!("complete: {message:?}, usage={usage:?}"),
        }
    }
    Ok(())
}
```

`run(input, true, cancellation)` 固定返回共享事件流；`true` 让调用方看到 `Delta` 和 `Complete`，`false` 只交付每次完整的 `Complete`。Agent 不是守护进程：一次 `run` 有明确开始和结束，等待人工反馈时通过 `reply` 恢复。

Workflow 的最小示例见 [第 06–08 章](#06-08-workflow图和执行关系) 和 [`workflow_events.rs`](../examples/workflow_events.rs)。

---

## Runtime：运行现场

### 2.1 身份

一次顶层执行使用应用提供的 `chat_id`。它同时是 Chat、Checkpoint 和事件关联的业务身份；组件不会替应用生成业务 ID。需要生成新执行 ID 时，通过 Builder 的 `with_id_fn(|| i64)` 传入应用函数。

```rust
let agent = agent::react()
    .model(chat)
    .with_id_fn(|| next_id())
    .build()?;
```

`Cancellation` 是一次调用的取消范围。每次独立 `run` 创建新的实例；把 clone 传给同一执行的并发工作，不要在多个独立运行之间复用。

Agent 和 Workflow 都有 Runtime，但公开方式不同。Agent 把执行现场封装在 Agent 对象中，应用直接使用
`agent.run(...)` 和 `agent.reply(...)`；Workflow 需要让应用管理图 State、Checkpoint 和等待恢复，因此应用
显式持有 `workflow::Runtime`。`workflow` 变量指向 Runtime 实例，`CompiledWorkflow` 只是不可变图定义。

### 2.2 State、Event 和持久化的关系

```text
Event Stream       -> 对外实时观察
Runtime State      -> 当前执行现场和恢复真相
Checkpoint         -> State 的外部持久化快照
短期/长期 Memory   -> 会话历史和记忆内容
OpenTelemetry Trace -> 可选的链路观察投影，不是短期记忆记录
```

事件不是 State 的替代物。`Delta` 只用于实时观察；完整消息由 `Complete` 交付。取消时，已经交付的内容保留在最终状态，未发生的步骤不会被伪造为成功。

### 2.3 取消、等待和回复

取消是执行控制信号，`waiting` 是业务状态。节点需要人工输入、审批或外部回调时，Runtime 保存等待事实并停止当前调用；用户随后用同一个对话身份调用 `reply`，Runtime 从等待队列中按先进先出匹配并继续。

```rust
// Agent：提交一条用户回复；回复结果继续从原 Agent 运行的事件流交付。
agent.reply(reply_input, Cancellation::new()).await?;

// workflow 变量的类型是 workflow::Runtime：提交 Messages 回复。
workflow.reply(response_messages, Cancellation::new()).await?;
```

运行已正常收口后，Checkpoint 可以标记完成；进程仍在运行但等待用户输入时，继续使用当前 Runtime 的等待状态，不要另建一套状态。

---

## Agent

### 3.1 Chat Agent

Chat Agent 负责普通模型对话，也可以配置 Tool、MCP、Skill、Permission、Middleware、Memory、Checkpoint 和 Evaluator。Chat 不拥有 Planner 循环；如果最终 Evaluator 拒绝结果，运行以错误收口。

```rust
let agent = agent::chat()
    .model(chat)
    .system_prompt("你是一个简洁、准确的助手")
    .tools(vec![weather_tool])
    .mcps(vec![connected_mcp])
    .skills(vec![research_skill])?
    .permission(permission)
    .middleware(retry)
    .checkpoint(checkpoint)
    .memory(memory)
    .evaluator(evaluator)
    .with_max_concurrency(100)
    .build()?;
```

Chat 使用 [`agent_live_tool.rs`](../examples/agent_live_tool.rs) 可验证真实工具调用。

### 3.2 ReAct Agent

ReAct 自带模型调用、工具循环和内置 Planner。模型决定是否调用 Tool、调用哪些 Tool 以及传递哪些参数；内部能力组装层负责把已配置的 Tool、MCP、Skill 描述提供给模型并执行实际调用。工具调用成功后结果作为下一轮模型上下文，工具直接错误也直接进入下一轮上下文；不额外包装成 `tool_result(error)`。

```rust
let agent = agent::react()
    .model(chat)
    .system_prompt("先查资料，再给出可核验的结论")
    .tools(vec![search_tool, calculator_tool])
    .mcps(mcp_servers)
    .skills(skills)?
    .permission(permission)
    .evaluator(evaluator)
    .with_max_iters(20)
    .with_max_concurrency(100)
    .checkpoint(checkpoint)
    .build()?;
```

ReAct 运行由 `max_iters` 限制模型循环次数，工具批次由 `max_concurrency` 限制并发。工具自己的 `before / call / after / eval` 是同一契约的可选 Hook；重试和状态归 Runtime/中间件控制，Tool 不自行创建 Attempt。

### 3.3 Agent 输入和返回

Agent 不把 `Messages` 当成业务 API 的输入类型。业务输入统一是 `agent::Input`；运行时再把 payload 组织为模型 `Messages`。

```rust
let input = Input {
    metadata: serde_json::Map::from_iter([
        ("chat_id".into(), 1001.into()),
        ("session_id".into(), 2001.into()),
    ]),
    payload: vec![Payload {
        r#type: "text".into(),
        data: serde_json::json!({"value": "北京天气怎么样？"}),
        mime_type: None,
    }],
};
```

图片、音频、文件使用同一个 Payload 结构，数据放在 `data`，可选 MIME 放在 `mime_type`。未配置对应 Provider 或处理能力时，输入会被明确拒绝，不会静默转换。

### 3.4 SubAgent 和 Team

SubAgent 是一个已经构建好的 Agent 实例，通过 `.subagents(...)` 提供给父 Agent 的委派路径；它不是 Tool，也不是动态注册表。

Team 是静态成员组合：成员在 Builder 阶段确定，Leader 决定把任务交给哪个成员，成员并发执行后回到 Leader。当前版本不支持运行中动态创建成员。示例：[`agent_live_team.rs`](../examples/agent_live_team.rs)、[`agent_live_three_subagents.rs`](../examples/agent_live_three_subagents.rs)。

---

## 输入、消息和事件

### 4.1 Input

`Input` 只有两个公共字段：

```rust
pub struct Input {
    pub metadata: Map<String, Value>,
    pub payload: Vec<Payload>,
}

pub struct Payload {
    pub r#type: String,
    pub data: Value,
    pub mime_type: Option<String>,
}
```

`metadata` 放 `chat_id`、`session_id`、语言、位置等业务关联信息；`payload` 表示用户实际提交的有序内容。`payload.type` 是数据类型，例如 `text`、`image`、`audio`、`file`、`json`；`data` 是该类型的数据。

### 4.2 Message

```rust
pub struct Message {
    pub role: Role,
    pub content: Vec<model::content::part::Part>,
    pub values: Map<String, Value>,
    pub metadata: Map<String, Value>,
    pub usage: Option<model::token::Usage>,
}
```

`content` 保存模型可观察的内容片段；`values` 保存结构化业务结果；`metadata` 保存来源和关联信息；`usage` 保存本次模型调用的 Token 使用。模型思考统一使用 `content` 中 `type = "think"` 的 Part，普通输出使用 `type = "text"`。

角色固定为 `system`、`developer`、`user`、`assistant`、`tool`、`function`。Tool 调用消息和工具返回消息都使用这个统一 Message 结构，不再额外发明一套工具结果包装。

### 4.3 Event

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

事件是所有可流式组件、Agent 和 Workflow 的共同观察契约。Provider 的供应商字段在 Core Adapter 中转换为统一 `think`、`text`、`tool_call` 等 Part；上层不需要知道 DeepSeek、VLLM 或其他供应商的原始字段。

### 4.4 Chat Request

```rust
let request = model::chat::Request {
    messages,
    tools,
    tool_choice: Some(model::chat::ToolChoice::Auto),
    stream: true,
    temperature: Some(0.7),
    max_tokens: Some(1024),
    top_p: Some(0.8),
    stop: None,
    response_format: None,
    options: serde_json::Map::new(),
};
```

模型公共参数放在 Request；供应商差异参数放在 `model::chat::Config.options` 或 Request 的 `options`。Provider 初始化参数（地址、密钥、超时）与单次调用参数分开。

---

## Checkpoint 和恢复

Checkpoint 保存一次 Runtime 的完整 JSON 现场，包含状态、图、节点执行事实、等待队列和恢复所需数据。它不是 Chat 历史，也不是 Trace；后端只实现 `save/load/delete/validate_state`。

```rust
let checkpoint = std::sync::Arc::new(halo::agent::core::checkpoint::InMemory::new());
let runtime = halo::agent::workflow::Runtime::new(chat_id, workflow.clone())?
    .with_checkpoint(checkpoint.clone());
// 运行并消费事件流后，保存完整 State。
runtime.save_checkpoint(&*checkpoint, &Cancellation::new()).await?;
let snapshot = checkpoint.load(chat_id, &Cancellation::new()).await?;
let restored = halo::agent::workflow::Runtime::from_snapshot(chat_id, workflow, snapshot)?;
```

恢复使用同一个 Workflow Definition 和 Version，再把保存的现场加载到新的 Runtime。Runtime 只从 Checkpoint 读取已保存事实，不从“某个节点是否有输出”猜测成功；错误 Attempt、等待节点和已完成节点都以 State 中的事实为准。

Checkpoint 后端可以是内存、数据库或应用自己的服务。没有配置 Checkpoint，就没有跨进程恢复能力；没有配置 Memory，也不会自动保存会话历史。

---

## Workflow 图和执行关系

### 6.1 构建节点

普通异步函数可以直接作为 Node。输入和输出由函数的类型决定，`Messages` 是唯一允许的非 object 根类型。

```rust
use halo::agent::workflow::{WorkflowBuilder, graph::{START, END}};
use halo::agent::{Cancellation, Error, Messages};
use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, JsonSchema)]
struct Answer { value: String }

let mut builder = WorkflowBuilder::new("weather");
builder.add_node("answer", |_: Messages, _: Cancellation| async {
    Ok::<_, Error>(Answer { value: "晴朗".into() })
})?;
builder.add_edge(START, "answer")?;
builder.add_edge("answer", END)?;
let workflow = builder.compile()?;
```

`add_node` 不要求用户先手工包装一个 Node 对象；组件如果要作为 Workflow 节点使用，可以实现 Node 契约，或者通过普通异步函数调用组件。`add_agent_node` 和 `add_graph_node` 用于明确嵌入 Agent 或已编译子图。

### 6.2 四类常用关系

```rust
builder.add_edge(START, "a")?;
builder.add_edge("a", "b")?;
builder.add_join(["b", "c"], "merge")?;
builder.add_conditional_edges("route", route, ["a", "b"])?;
builder.add_fan_out_edges("split", "worker")?;
builder.add_loop_edge("retry", "work", |messages| should_retry(messages))?;
```

- `add_edge` 是普通关系：任一成功来源满足激活条件即可；多个来源的结果仍保存在各自 Node State，目标节点按 `attrs.input` 取所需字段。
- `add_join` 是显式等待关系：列出的所有来源都成功后才能激活目标；一个来源最终失败，Join 不会伪造成功。
- `add_conditional_edges` 先执行来源节点的路由函数，实际选择必须属于声明的目标集合。
- `add_fan_out_edges` 表示同一来源向多条执行分支分发，分支执行可并发；每个执行实例有自己的执行身份，结果回收遵循运行时状态。
- `add_loop_edge` 是受条件控制的循环关系；不能创建自环、重复边或非法图环路。

节点和边都是编译期图定义。`START` 和 `END` 是边界端点，应用不把它们当普通 Node 保存。

### 6.3 属性绑定和失败关系

节点的输入通过 `NodeAttrs.input` 绑定上游 Node State 的字段；模型、函数和工具不需要知道 Workflow 的全局状态结构。失败关系用 `add_failure_edges` 声明，Runtime 负责记录失败 Attempt、应用重试策略和决定后续路径。

```rust
builder.add_failure_edges(["search"], ["fallback"], "node_failed")?;
builder.add_failure_edges(["a", "b"], ["recover"], "join_failed")?;
```

普通 Node 的重试次数、退避和取消由 Runtime/中间件决定；Node 本身只返回成功结果或统一 Error。

### 6.4 Workflow Runtime

```rust
let mut runtime = halo::agent::workflow::Runtime::new(1001, workflow)?
    .with_max_concurrency(100)?;
let mut events = runtime
    .run(Messages::default(), next_exec_id, Cancellation::new())
    .await?;
while let Some(event) = events.next().await {
    println!("{:?}", event?);
}
let state = runtime.state();
```

Workflow 一次按当前可执行前沿推进。一个 Step 内可以并发执行多个节点，但 Join 必须等待全部来源完成；`State` 保存图、节点状态、执行实例、Attempt、结果和等待队列。子图通过 `add_graph_node` 作为一个明确的图边界嵌入父图，不会变成父图的普通节点集合。

### 6.5 Workflow 与 Agent 的关系

Workflow 可以通过 `add_agent_node` 调用一个已经构建好的 Agent；Agent 也可以把 Workflow 作为外部能力使用。两者共享 Message/Event/Checkpoint 语义，但各自保有自己的 Runtime State。不要把 Agent 的计划图当成 Workflow 编排图。

---

## 与其他框架的关系

Heart 借鉴 LangGraph 的 `START/END`、条件边、子图和 Checkpoint 现场；借鉴 Eino 的组件契约、Provider 适配和编排边界；借鉴 AgentScope 的工具组装、Workspace、Permission、Plan 和 Harness 组合方式。

这些框架的 DSL 不直接成为 Heart Runtime 的内部状态。应用可以把 Dify、Coze 或自定义 DSL 转成 Heart Native Definition，再交给 Core 解析和校验：

```text
外部 DSL JSON
    -> 转换为 Heart Native DSL
    -> core::workflow::dsl 解析、校验和规范化
    -> Core 绑定 Core 组件
    -> Runtime 绑定 Agent、Workflow 和 CompiledWorkflow 执行器
    -> WorkflowBuilder::compile()
    -> Runtime
```

应用在初始化阶段提供静态执行配置，请求阶段只传入 DSL 和业务输入。DSL 只描述图和参数，不能绕过应用已配置的函数、工具、Agent 或子图；未知能力在构建或计划提交阶段失败。Heart 不增加公共 Registry、Capabilities 或动态能力注册表。

页面代码函数节点由 Core 根据 DSL 直接创建执行器，再像普通 Node 一样交给 Workflow Builder；不会按
`name` 查找本地函数，也不会把闭包或本地路径写进 DSL：

```rust
use halo::agent::Cancellation;
use halo::agent::core::sandbox::{Config, Local};
use halo::agent::core::workflow::{code::{Code, Object}, dsl::parse};
use halo::agent::node::Node;
use serde_json::json;
use std::collections::BTreeMap;

let document = parse(json!({
    "version": 1,
    "definition_id": "reload-flow",
    "nodes": {
        "reload": {
            "type": "function",
            "name": "reload",
            "config": {
                "language": "python",
                "entry": "reload",
                "source": "def reload(input): return {'value': input['value'] + 1}"
            },
            "schema": {
                "input": {"type": "object", "required": ["value"]},
                "output": {"type": "object", "required": ["value"]}
            }
        }
    },
    "edges": [
        {"from": "START", "to": "reload"},
        {"from": "reload", "to": "END"}
    ]
}))?;
let code = Code::from_dsl(Local::new(Config::default())?, &document.nodes["reload"])?;
let output = code.call(
    Object(BTreeMap::from([(String::from("value"), json!(1))])),
    &Cancellation::new(),
).await?;
```

如果要放入 Workflow，使用 `builder.add_node_with_schema("reload", code, Schema { input: node.schema.input.clone(), output: node.schema.output.clone() })`；这样 DSL 的精确 Schema 会进入 Definition，页面代码仍只接收 JSON 对象并返回 JSON 对象。

A2A、AG-UI、JSON-RPC、SSE 和 Webhook 是协议/传输层适配，不改变 Runtime 的 State 和 Event 契约。HTTP 层可以将已有 Event 转换成对外协议事件。

---

## 已经固定的使用规则

1. 公共入口统一从 `halo::agent` 开始；不使用 `halo::core::agent` 兼容路径。
2. 一次顶层执行用 `chat_id`；不要再引入第二套 `run_id` 业务身份。
3. Agent 业务输入是 `agent::Input`；模型输入是 `model::chat::Request` 和 `Messages`。
4. `Event::Delta` 是实时增量，`Event::Complete` 是一次完整交付；不要从 Delta 伪造 Complete。
5. 模型思考统一为 `content` Part 的 `type = "think"`。
6. 工具执行错误直接返回统一 Error 并进入后续模型上下文；不额外包装 `tool_result(error)`。
7. Tool 的 `before / call / after / eval` 是同一 Tool 契约的可选阶段；重试由 Runtime/中间件控制。
8. Memory、Checkpoint 和 Chat 各有边界：Checkpoint 是恢复现场，Memory 是历史/长期记忆，Chat 是外部对话收口；OpenTelemetry Trace 只负责链路观察，不增加另一张短期记忆表。
9. Session 是短期会话聚合；长期记忆只使用 `summary`、`fact`、`preference` 等固定类型，是否向量化由应用提供 Embed 和后端决定。
10. 没有配置某项能力，就没有该能力；框架不会隐式创建模型、数据库、Sandbox、MCP 或 Memory。
11. 所有公开异步能力使用 Rust 原生 `async fn`，不要求 `async_trait`。
12. `core` 实现组件，`runtime` 管运行现场，`harness` 做组合；不在组件中复制运行循环。

---

## 当前限制和后续范围

以下内容属于后续版本或应用层职责，当前手册不把它们当作已完成能力：

- 更完整的 Document Parser/布局解析和 OCR 策略；当前保留 Parser、Chunker、RAPTOR、Graph 的契约和 LiteParse 适配入口。
- Provider 的完整 TTS、多模态输入输出和供应商特有字段扩展。
- 更完整的服务器部署、任务队列、跨进程等待恢复和分布式锁。
- 业务应用的 DSL 编辑器、账号权限、数据库 schema、向量库/图数据库写入编排。
- 更复杂的 Team 动态成员管理；当前 Team 成员在构建时确定。

这些边界不会阻止应用实现自定义组件。应用可以实现契约并传入 Builder，不需要修改 Runtime。

---

## 组件使用索引

### 13.1 Model Provider

Provider 初始化客户端，`.chat(chat::Config)` 绑定模型；每次调用使用 `chat::Request`。

```rust
let provider = halo::agent::provider::vllm::Vllm::new(config)?;
let chat = provider.chat(halo::agent::model::chat::Config::new("model-name"))?;
```

当前实现包含 DeepSeek、VLLM/OpenAI-compatible、OpenAI、Gemini 和 Zhipu。真实调用示例：[`provider_deepseek.rs`](../examples/provider_deepseek.rs)、[`provider_vllm.rs`](../examples/provider_vllm.rs)。

### 13.2 Tool、Function 和 Permission

Tool 必须提供稳定 `ToolDefinition` 和异步 `call`；参数 Schema 根必须是 object。`before` 可以校验/补齐参数，`after` 可以修改返回 Message，`eval` 可以反馈业务结果。

如果工具只是一个普通的异步 Rust 函数，可以使用 `#[halo::tool]` 生成现有 Tool 契约的包装器：

```rust
use halo::Cancellation;
use halo::agent::Error;

#[halo::tool(description = "查询指定城市天气")]
async fn get_weather(
    location: String,
    cancellation: &Cancellation,
) -> Result<String, Error> {
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "weather request cancelled"));
    }
    Ok(format!("{location} 今天晴，26°C。"))
}

let agent = halo::agent::chat()
    .model(chat)
    .tools(get_weather())
    .build()?;
```

宏会根据函数参数生成 `ToolDefinition.parameters`，函数名默认作为工具名，`description` 必须显式提供；返回的可序列化值会转换为既有的 `Message::function(...)`，也可以直接返回 `Message`。该方式不会把普通 `Function` 自动暴露给模型，也不新增 `ToolContext`、`tool::Output` 或 `tool_result` 包装。复杂方法、闭包或需要自定义 `before/after/eval` 的工具继续直接实现 `halo::agent::tool::Tool`。

```rust
impl halo::agent::tool::Tool for Weather {
    fn definition(&self) -> &ToolDefinition { &DEFINITION }
    async fn call(&self, call: &Call, cancel: &Cancellation) -> Result<Message, Error> {
        // 访问真实服务，返回 Message::function(...)
    }
}
```

Permission 只在实际 Tool/MCP 调用前检查。内置策略可从 `halo::agent::core::permission` 使用：`default`、`explore`、`edit`、`bypass`；没有挂载 Permission 就不启用权限阶段。参考 [`agent_permission.rs`](../examples/agent_permission.rs)。

### 13.3 Workspace、MCP、Skill

Workspace 决定 local/server 数据源；local 操作本地文件，server 操作应用配置的数据库或远端存储。Workspace 自带基础文件工具，MCP 和 Skill 需要应用显式接入。

```rust
let agent = agent::react()
    .model(chat)
    .tools(local_tools)
    .mcps(connected_servers)
    .skills(discovered_skills)?
    .build()?;
```

Skill 可以按 Workspace 的 loader 发现，应用负责在初始化阶段处理 hash、缓存和同步；运行过程中只使用已加载的技能描述和脚本。MCP 连接和工具发现应在 Agent 构建前完成，运行时只负责调用。

### 13.4 Prompt

Prompt Template 只负责模板渲染，不负责 Tool 调度：

```rust
let template = halo::agent::prompt::Template::new()
    .message(system_message("你是助手"))
    .messages("history")
    .message(user_message("问题：{question}"));
let variables = halo::agent::prompt::Variables::new()
    .value("question", serde_json::json!("北京天气怎么样"))
    .messages("history", history);
let messages = template.format(&variables)?;
```

### 13.5 Memory 和 Compression

短期 Memory 由一个后端负责 Session/Chat；长期 Memory 使用统一 `memory::Request/Item`，支持 `summary`、`fact`、`preference`。Agent 通过 `memory::Config::new(short, long)` 组合：

```rust
let memory = halo::agent::memory::Config::new(short_store, long_store)
    .with_scopes(["user", "agent"])
    .with_compression(compressor)
    .with_extractor(extractor)
    .with_id_fn(next_id);
```

没有 `with_compression` 就不启用自动压缩；压缩是用户触发请求时检查并后台执行，不阻塞当前主流程。压缩阈值使用摘要覆盖范围之后的 Chat Usage 累加；新摘要只保留当前一份，并记录用于展示的 Usage。

### 13.6 Checkpoint、Sandbox、Document、Search 和 Embed

- `Checkpoint` 保存完整 Runtime State；实现 `save/load/delete/validate_state`。
- `Sandbox` 是执行能力契约；本地实现从 `halo::agent::core::sandbox::Local` 使用。
- `Document` 组件定义 Source/Document/Chunk/Parser/Chunker 等数据边界；解析策略可以由应用选择。
- `knowledge::Search` 的 keyword/vector/hybrid 请求各自有明确参数；具体索引后端由应用实现。
- `model::embed::Embed` 是模型组件；调用返回统一 Message，向量放在 `values`，模型和维度放在 `metadata`。

`Local` 只执行显式 argv 命令，继承宿主机的环境、工作目录、权限和外部副作用，
不是安全隔离边界，只适合可信任务。页面函数或其他不可信输入不能把 `Local` 当作
沙箱使用；需要进程、资源和网络隔离时，应由应用配置 `core::sandbox::docker::Docker` 或
服务端 Workspace/Worker。Docker 镜像必须由应用预先准备，框架不会自动拉取或回退到
宿主机执行。

综合组件示例见 [`component_contracts.rs`](../examples/component_contracts.rs)。

---

## 短期、长期记忆和持久化

### 14.1 短期记忆

短期记忆由 `session` 和 `chat` 组成：Session 聚合长期对话线程，Chat 保存一次外部输入和最终 Assistant Message。运行过程中的完整 State 属于 Checkpoint；`runtime::trace` 是可选的 OpenTelemetry 链路观察，不是另一种短期记忆记录。

### 14.2 长期记忆

长期记忆记录使用固定字段：`id`、`type`、`user_id`、`agent_id`、`session_id`、`from`、`to`、`content`、`version`、`created_time`、`updated_time`、`metadata`。`from/to` 表示它覆盖的短期记录范围；应用负责选择合适的用户、Agent 或 Session 归属。

长期后端不持有 Embed 实例。应用先调用 `model::embed`，再把文本和向量一起传给支持向量字段的后端；不支持向量的后端只保存文本和元数据。

### 14.3 读写边界

Memory 的最小契约是异步 CRUD 和按 Filter 查询；删除按 ID 物理删除，更新使用版本控制，写入/更新失败由后端返回统一 Error。Session 同时写入多个请求时由应用选择乐观锁或串行策略，框架不假设一个全局锁。

参考：[`short_memory.rs`](../examples/short_memory.rs)、[`agent_memory.rs`](../examples/agent_memory.rs)。

---

## 测试和发布

### 15.1 本地检查

```bash
cargo fmt --check
cargo check --all-targets
cargo test --lib --no-fail-fast
cargo clippy --lib --all-features -- -D warnings
```

### 15.2 真实服务验证

需要模型、MCP、ES 或 Sandbox 的测试只从环境变量读取配置；不要把 Mock 响应当作真实验收。示例：

```bash
DEEPSEEK_API_KEY=... \
DEEPSEEK_MODEL=deepseek-flash \
cargo run --example provider_deepseek

HEART_VLLM_BASE_URL=... \
HEART_VLLM_API_KEY=... \
HEART_VLLM_MODEL=... \
cargo run --example agent_live_tool
```

没有外部服务的组件契约测试可以使用确定性的本地组件验证类型和状态转换；它们不应被描述成模型能力验收。

`bash scripts/test-real.sh` 会读取被 Git 忽略的 `.env`，并区分三种结果：必需变量缺失时立即以非零状态退出，真实测试失败时保留失败状态，可选后端未配置时输出 `SKIP` 并在最后汇总。脚本不会打印密钥。

真实验收环境矩阵：

| 能力 | 必需环境或资源 | 覆盖测试 |
| --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY`；可选 `DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` | Agent、Workflow、Memory、A2A |
| VLLM/OpenAI-compatible | `HEART_VLLM_BASE_URL`、`HEART_VLLM_API_KEY`、`HEART_VLLM_MODEL` | Provider、Chat/ReAct、Team、Workflow |
| Zhipu Embedding | `HEART_EMBED_API_KEY`；可选 `HEART_EMBED_BASE_URL`、`HEART_EMBED_MODEL` | 向量 Memory、Semantic Chunker |
| OpenAI Embedding | `OPENAI_API_KEY`；可选 `OPENAI_BASE_URL`、`OPENAI_MODEL` | OpenAI Embedding |
| Gemini TTS/多模态 | `GEMINI_API_KEY`；多模态另需真实 `GEMINI_TEST_IMAGE_DATA_URL` | TTS、多模态 |
| Elasticsearch Workspace | `HEART_ES_URL`、隔离的 `HEART_ES_TEST_INDEX`；可选 `HEART_ES_API_KEY`、`HEART_ES_PIPELINE_TEST_INDEX` | Workspace/ES Harness |
| Docker Workspace | 已存在的 `HEART_WORKSPACE_TEST_IMAGE` 和可用 Docker daemon | 容器、挂载、取消 |
| MCP | 测试会启动仓库内真实协议服务进程 | HTTP MCP、stdio MCP |

`cargo test --all-targets` 默认只运行确定性测试，标记为 `ignored` 的真实验收不会被伪装成通过；需要真实后端时使用上面的脚本。

### 15.3 版本和发布

```bash
cargo package --allow-dirty
cargo publish --dry-run
```

当前 Alpha 版本不承诺 API 稳定性。发布前应同时检查 README 链接、中文/英文手册、examples 编译和环境变量说明。

---

## 压缩与 A2A

### 16.1 压缩

压缩组件只生成新的 Session 摘要，不改写原始 Chat。触发时读取摘要覆盖范围之后的 Chat，按每条 Chat 已记录的输入/输出 Usage 累加；压缩模型本次调用的 Usage 写到 Session 当前摘要信息中用于展示，不重复计入下一轮阈值。

摘要提交后，后续请求使用“当前摘要 + 尚未覆盖的 Chat”；正在执行的请求不会中途替换成新摘要。压缩失败不影响原始 Chat，下一次用户主动请求可以再次检查并启动后台压缩。

### 16.2 A2A

A2A 是协议层，不是新的 Runtime。Server 将 `chat_id` 映射为远端任务的 `context_id`，把 Agent Event 转换成 JSON-RPC/SSE 事件，用户回复仍回到 Agent 的 `reply`。协议转换放在 HTTP 层，核心 Agent 和 Workflow 不需要开启“AG-UI/A2A 模式”。

---

## Server 和 HTTP

Server 由 Axum 提供外层绑定，Runtime 继续管理事件和运行状态。当前 JSON-RPC 方法包括 `message/send`、`message/stream`、`tasks/get`、`tasks/cancel` 和 `tasks/resubscribe`；SSE 只是事件传输形式。

```rust
let service = std::sync::Arc::new(halo::agent::server::a2a::Service::with_store(agent, store));
let router = halo::agent::server::axum::router(service);
let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
axum::serve(listener, router).await?;
```

Server 的 Task Store 保存外部任务投影；Agent/Workflow Runtime 仍是实际执行真相。Server 不复制 State、事件排序或等待恢复逻辑。Webhook、客户端和 SSE 适配可从 `halo::agent::server` 访问。

---

## Team 和多 Agent

Team 是静态组合，不是运行中动态创建 Agent。构建阶段确定成员，Leader 负责把任务路由给成员，成员使用各自的 Agent Runtime 执行，结果回到 Team 的事件流。

```rust
let researcher = agent::react()
    .model(research_chat)
    .tools(vec![search_tool])
    .build()?;
let writer = agent::chat()
    .model(writer_chat)
    .build()?;

let team = agent::team()
    .leader(agent::react().model(leader_chat))
    .member(agent::SubAgent::new("researcher", researcher)?)
    .member(agent::SubAgent::new("writer", writer)?)
    .handoff("leader", "writer")
    .build()?;
```

团队成员共享外部 `chat_id/session_id` 关联，但保留各自执行状态。成员失败时由 Team/Agent Runtime 根据错误和计划决定重试、反馈或重规划；不要在成员内部偷偷创建第二个 Team 调度器。

`team_assign` 是委派：成员完成后结果回到 Leader，Leader 继续推理。`handoff` 是控制权转交：Leader 把任务交给声明的成员后不再恢复，成员的最终 Assistant Message 就是本次运行的最终 `Complete`，其 `finish_reason` 为 `handoff`。目标只能是构建阶段已经注册的成员，模型只能从声明目标中选择；一次模型响应不能同时发起 Handoff 和其他调用。成员如果进入 `waiting`，运行仍按现有 `reply`/Checkpoint 规则等待和恢复。

完整的真实 VLLM 示例见 [`agent_live_handoff.rs`](../examples/agent_live_handoff.rs)。它展示了 Leader、固定成员、`handoff` 声明、真实模型调用和最终事件收口。

---

## 附录 A：示例索引

| 示例 | 说明 |
| --- | --- |
| [`agent_live_deepseek.rs`](../examples/agent_live_deepseek.rs) | DeepSeek Chat Completions、think/text 映射 |
| [`agent_live_tool.rs`](../examples/agent_live_tool.rs) | 真实 VLLM + 真实本地工具调用 |
| [`agent_live_subagent.rs`](../examples/agent_live_subagent.rs) | SubAgent 委派 |
| [`agent_live_team.rs`](../examples/agent_live_team.rs) | Team Leader 和成员 |
| [`agent_live_handoff.rs`](../examples/agent_live_handoff.rs) | Team Leader Handoff，真实 VLLM |
| [`agent_live_three_subagents.rs`](../examples/agent_live_three_subagents.rs) | 多个静态 SubAgent |
| [`agent_memory.rs`](../examples/agent_memory.rs) | Agent 与短/长期 Memory |
| [`agent_middleware.rs`](../examples/agent_middleware.rs) | Middleware 生命周期 |
| [`agent_permission.rs`](../examples/agent_permission.rs) | Permission 策略 |
| [`workflow_events.rs`](../examples/workflow_events.rs) | Workflow Event Stream 和 State |
| [`checkpoint.rs`](../examples/checkpoint.rs) | Checkpoint 保存与恢复 |
| [`component_contracts.rs`](../examples/component_contracts.rs) | 基础组件契约 |
| [`planning_evaluation.rs`](../examples/planning_evaluation.rs) | Planner、Evaluator |
| [`tool_macro.rs`](../examples/tool_macro.rs) | `#[halo::tool]` 函数式 Tool |
| [`compress.rs`](../examples/compress.rs) | Prompt 与压缩 |
| [`short_memory.rs`](../examples/short_memory.rs) | Session/Chat/Long Memory CRUD |
| [`provider_deepseek.rs`](../examples/provider_deepseek.rs) | DeepSeek Provider |
| [`provider_vllm.rs`](../examples/provider_vllm.rs) | VLLM Provider |

## 附录 B：排错顺序

1. 确认 Rust 版本和 `cargo check --all-targets`。
2. 确认 Provider 的 `base_url`、`api_key`、`model_name`。
3. 确认 Request 的 `messages`、Tool Schema 和 `stream`。
4. 如果 Agent 没有调用工具，检查工具描述、模型能力、系统提示和服务日志。
5. 如果出现 `WAITING`，通过 `reply` 提交对应输入，不要重新创建一个运行。
6. 如果恢复失败，确认使用同一个 Definition/Version 和完整 Checkpoint JSON。
7. 如果 Memory 没有数据，确认 Agent 挂载了 Memory，输入 metadata 含有应用所需的身份字段。
8. 如果 HTTP 返回错误，先查看 Heart `Error { code, message, details }`，再查看 JSON-RPC 映射。

## 附录 C：英文手册

English version: [`USAGE.en.md`](USAGE.en.md)。
