# Heart 使用手册

本手册介绍如何在 Rust 应用中使用 Heart 构建 Agent 和 Workflow。示例按当前公开 API 编写。

[English version](USAGE.en.md)

## 1. 安装

Heart 的 Cargo 包名和 Rust crate 名是 `halo`，要求 Rust 1.99.1 或更新版本。

```toml
[dependencies]
halo = { git = "https://github.com/halo-space/heart.git" }
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

## 3. Chat Agent

Chat Agent 是普通对话循环。它可以使用已挂载的 Tool、MCP 和 Skill；需要动态计划、迭代和重规划时使用 ReAct Agent。

### 3.1 构造和运行

```rust
use futures_util::StreamExt;
use halo::{agent, Cancellation};

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
use halo::{Cancellation, Error};
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
use halo::{Cancellation, Error, Messages};
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

## 7. Workspace、Skill 和 MCP

Workspace 决定执行环境是本地文件还是服务器数据库。Local Workspace 默认提供列目录、读取、写入、精确编辑、Glob、Grep 和单文件删除；
命令执行必须显式配置 Sandbox。

```rust
let workspace = halo::agent::core::workspace::Local::new("./workspace")?
    .sandbox(halo::agent::sandbox::Local::default())
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

示例只使用当前公开契约。需要网络、凭据或外部服务的示例会在标题中明确标出；其余示例离线即可运行。

### 10.1 离线 Chat Agent：从 Input 到 Delta/Complete

这个例子使用一个确定性的本地 Chat 实现，因此不需要 API Key。它展示了四件事：

1. `Input.metadata` 放业务身份，`payload` 放有序用户输入；
2. Agent 的输入不是模型 `Messages`，Agent 内部才把 Input 组织成模型消息；
3. `stream = true` 同时观察 Delta 和 Complete；
4. `Complete.message` 才是该次调用的完整结果，不能用 Delta 反推 Runtime State。

```rust
use futures_util::{StreamExt, stream};
use halo::{agent, event::Event, message::{Message, Role}, model::{chat::{Chat, Request}, content::part::Part}, Cancellation, Error};
use serde_json::json;

struct DemoChat;

impl Chat for DemoChat {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, _request: Request) -> Result<Self::Stream, Error> {
        let mut delta = Message::new(Role::Assistant);
        delta.content.push(Part {
            r#type: "text".into(),
            data: json!({"value": "北京今天晴朗。"}).as_object().unwrap().clone(),
        });
        let mut complete = Message::new(Role::Assistant);
        complete.content.push(Part {
            r#type: "text".into(),
            data: json!({"value": "北京今天晴朗，适合出行。"}).as_object().unwrap().clone(),
        });
        Ok(stream::iter(vec![
            Ok(Event::Delta { message: delta }),
            Ok(Event::Complete {
                message: complete,
                usage: None,
                finish_reason: Some("stop".into()),
            }),
        ]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let agent = agent::chat()
        .model(DemoChat)
        .system_prompt("你是一个简洁的中文助手")
        .build()?;

    let input = agent::Input {
        metadata: serde_json::Map::from_iter([
            ("chat_id".into(), json!(1001)),
            ("session_id".into(), json!(2001)),
            ("agent_id".into(), json!(3001)),
        ]),
        payload: vec![agent::Payload {
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

下面的模型第一次返回一个工具调用，第二次读取工具结果并给出最终答案。工具本身实现 `Tool` 契约；
Permission 只在真正执行工具时检查，模型只负责选择工具和参数。把 `.tools(Lookup)` 改成多个工具时可以传
`vec![Lookup, OtherTool]`，不需要 `Toolkit` 公共对象。

```rust
use std::sync::{LazyLock, atomic::{AtomicUsize, Ordering}};
use futures_util::{StreamExt, stream};
use halo::{agent, event::Event, message::{Message, Role}, model::{chat::{Chat, Request}, content::part::Part}, permission::{Behavior, Decision, Request as PermissionRequest}, tool::{Call, Tool, ToolDefinition}, Cancellation, Error};
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

Checkpoint 保存的是完整 Runtime State，而不是 Trace。生产环境通常让 Runtime 在成功或最终异常收口时写一次快照；示例使用
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
use halo::{Cancellation, Error, Message, agent::{self, Agent, Input, Payload}, core, event::Event, memory::{self, Memory}, memory::short::{self, chat, session}, message::Role, model::{chat::{Chat, Request}, content::part::Part}};
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
    let config = memory::Config::new(core::memory::short::session::InMemory::new(), core::memory::short::chat::InMemory::new(), core::memory::short::trace::InMemory::new(), core::memory::InMemory::new()).with_scopes(["user", "agent"]);
    config.session.create(session::Session { id: 1, tenant_id: 1, user_id: 2, r#type: short::Type::Agent, title: Some("天气".into()), summary: None, usage: None, status: session::Status::Active, metadata: Map::new(), created_time: 0, updated_time: 0 }, &cancellation).await?;
    config.chat.create(chat::Chat { id: 2, tenant_id: 1, user_id: 2, agent_id: 3, session_id: 1, r#type: short::Type::Agent, status: chat::Status::Completed, input: Some(chat::Request { metadata: Map::new(), payload: vec![chat::Payload { r#type: "text".into(), data: json!({"value":"北京天气？"}), mime_type: None }] }), message: None, usage: None, graph: None, metadata: Map::new(), started_time: None, completed_time: None, created_time: 0, updated_time: 0 }, &cancellation).await?;
    config.long.write(memory::Request { id: 3, r#type: memory::Type::Preference, user_id: 2, agent_id: None, session_id: None, from: 1, to: 2, content: "用户喜欢露营".into(), metadata: None, vector: None }, &cancellation).await?;
    let agent = agent::chat().model(Model).memory(config).build()?;
    let input = Input { metadata: Map::from_iter([("chat_id".into(), json!(3)), ("session_id".into(), json!(1)), ("user_id".into(), json!(2)), ("agent_id".into(), json!(3))]), payload: vec![Payload { r#type: "text".into(), data: json!({"value":"天气好就去露营"}), mime_type: None }] };
    let mut events = agent.run(input, false, Cancellation::new()).await?;
    while let Some(event) = events.next().await { println!("{:?}", event?); }
    Ok(())
}
```

压缩是独立能力：只有应用或 Harness 配置了压缩组件并主动触发检查，才会用摘要替换已覆盖的 Chat 范围；本次请求不会中途
切换到新摘要。

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
`.skills(loader)` 挂载。Server Workspace 不假装存在本地路径，而是由后端 Loader 从数据库读取同样的定义和资源。

### 10.9 MCP：先连接和发现，再挂载

MCP 连接不是运行中临时发现。下面是完整的初始化形态；如果网络不可用，程序会在 `Server::http` 返回统一错误，Agent 不会被构造出来。

```rust
use futures_util::stream;
use halo::{Cancellation, Error, Message, agent, event::Event, mcp::{Definition, Server as _}, message::Role, model::chat::{Chat, Request}};
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

如需让该段也在没有 MCP 服务时离线运行，可把 `Server::http` 替换成测试中的内存 `Server` 实现。不要把连接失败当成“可选工具”静默吞掉；
应用应先决定是否继续构造 Agent。当前示例中的 `HEART_MCP_SSE_URL` 也只能放在 `.env`，不能提交。

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

最后再运行需要外部服务的 VLLM/MCP 示例。若 `cargo test --all-targets` 报缺少被忽略的测试 fixture，说明本地测试目录不是发布内容，
不要把 `.env`、测试数据或设计文档强行加入源码提交。
