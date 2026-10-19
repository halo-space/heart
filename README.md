# Heart

Heart 是一个使用 Rust 构建的 Agent 与 Workflow 执行框架，目标是用统一的消息、事件、组件和运行时契约，支持从简单对话 Agent 到复杂工作流的组合与执行。

项目提供模型、工具、记忆、文档处理、权限、中间件、检查点、Sandbox 等基础能力，并将组件契约、具体实现和运行时执行状态分层组织，便于替换模型供应商、存储后端和执行策略。

## 目录结构

```text
src/
├── components/   公共组件契约，例如 Model、Tool、Memory、Message、Event
├── core/         Core 能力及其实现，包括可复用组件实现和基础 Agent 实现
├── runtime/      并列且独立的 Agent 与 Workflow 运行状态、图结构和执行控制
└── harness/      DeepAgents 式的高阶 Agent 组装能力，不是 Agent/Workflow 的通用中间层
```

`components` 只定义契约边界；Agent 与 Workflow 是两类独立能力，各自使用自己的 Runtime，并共同复用 `core` 中适用的能力。基础 Agent 实现与 DeepAgents 式高阶 Harness 都组合 Core 能力。Harness 是面向长任务 Agent 的场景化组装与默认策略层，不承接所有 Agent 和 Workflow 共用的生命周期逻辑，也不另造执行 Runtime。当前 Harness 基于 Core ReAct，提供长任务默认提示词，以及可选 Workspace 的文件工具；迭代上限沿用 Core ReAct 默认值并可单独配置。Memory、Checkpoint、Permission、Middleware、Toolkit 和 SubAgent 仍复用现有能力。

## 快速开始

项目名为 Heart，当前 Cargo 包名为 `halo-agents`，Rust 中使用 `halo_agents`。在调用方项目的
`Cargo.toml` 中添加以下依赖；两个示例各自完整，可任选一个放入 `src/main.rs` 后执行 `cargo run`。
示例仅使用内存和固定的演示模型，运行时不需要 API Key、外部服务或数据库，也不调用真实付费模型。
首次获取依赖仍需网络；依赖已缓存后可以离线构建和运行。

```toml
[dependencies]
halo-agents = { git = "https://github.com/halo-space/heart.git" }
futures-util = "0.3"
serde_json = "1"
serde = { version = "1", features = ["derive"] }
schemars = "0.8"
tokio = { version = "1", features = ["macros", "rt"] }
```

### Agent：输入到事件流

演示模型实现现有 `model::chat::Chat` 契约，返回固定的 Complete；实际使用时将 `DemoChat`
替换成配置好的 Provider Chat 实例即可，Agent 的构造与 `run` 用法不变。

```rust
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, Payload},
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
    },
};
use serde_json::{Map, json};

struct DemoChat;

impl Chat for DemoChat {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: "text".into(),
            data: Map::from_iter([("value".into(), json!("你好，Heart！"))]),
        });
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let agent = agent::chat()
        .model(DemoChat)
        .system_prompt("你是一个助手")
        .build()?;
    let input = Input {
        metadata: Map::new(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value": "你好"}),
            mime_type: None,
        }],
    };
    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
```

这里没有配置 Memory 或 Checkpoint，因此不要求业务身份字段。配置 Memory 后，应用先创建 Session，
再在 `Input.metadata` 中提供 `tenant_id/user_id/agent_id/session_id/chat_id`。配置 Checkpoint 时需要
`chat_id`；默认内存 Checkpoint 只在本进程有效，进程重启恢复需要调用方提供持久化后端。
ReAct 使用同一 `Input` 和事件流入口，通过 `agent::react()` 构造，并按需配置 `.toolkit(...)` 等组件。

`run(input, true, cancellation)` 交付实际产生的 Delta 和 Complete，`false` 仅交付 Complete；
不会把多次模型或工具调用合成一个 Complete，也不会给只返回 Complete 的模型伪造 Delta。

### Workflow：普通函数到图执行

普通异步函数可以直接加入图。这个示例执行一个节点，再从 Runtime State 读取业务结果；
事件是观察副本，不能用事件中的 `values` 替代下游读取 Runtime 中的数据。
普通 Node 的业务输入/输出需要对象根 Schema（`Messages` 是例外）；示例使用具名结构体和
`Serialize/JsonSchema` 声明输出，构图时会校验，不用无约束的 `Value` 绕过契约。

```rust
use futures_util::StreamExt;
use halo_agents::{
    Cancellation, Error, Messages,
    runtime::workflow::{
        GraphError, Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use schemars::JsonSchema;
use serde::Serialize;
use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};

#[derive(Serialize, JsonSchema)]
struct Answer {
    answer: i64,
}

async fn answer(_: Messages, _: Cancellation) -> Result<Answer, Error> {
    Ok(Answer { answer: 42 })
}

fn graph_error(error: GraphError) -> Error {
    Error {
        code: error.code,
        message: error.message,
        details: error.details,
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let mut builder = WorkflowBuilder::new("hello-workflow");
    builder.add_node("answer", answer).map_err(graph_error)?;
    builder.add_edge(START, "answer").map_err(graph_error)?;
    builder.add_edge("answer", END).map_err(graph_error)?;
    let mut runtime = Runtime::new(1, builder.compile().map_err(graph_error)?)?;
    let ids = AtomicI64::new(100);
    {
        let mut events = runtime
            .run(
                Messages::default(),
                Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
                Cancellation::new(),
            )
            .await?;
        while let Some(event) = events.next().await {
            println!("{:?}", event?);
        }
    }
    let execution = &runtime.state().nodes["answer"].executions[&100];
    println!("业务结果：{:?}", execution.messages.as_slice()[0].values);
    Ok(())
}
```

示例中的递增 ID 仅用于演示。实际业务由应用提供统一的 i64 执行 ID 生成函数，例如已有的雪花 ID 函数；
框架不初始化 ID 服务。读取 State 前先消费并释放运行流，不在流存活期间另开运行或复制执行现场。

内置 Chat/ReAct 与 Workflow 返回的流可直接通过 `let mut events` 和 `.next().await` 消费，
不需要调用方额外固定。自定义 Agent/Chat 的公共契约不要求流实现 `Unpin`；
泛型代码消费任意实现时仍可能需要固定，不因此收紧公共契约。

## Workspace 与 Harness

工具、MCP 和 Skill 先组装进 Toolkit，再挂载到 Workspace；Harness 接收配置好的 Workspace：

```rust
let workspace = core::workspace::Local::new("./workspace")?
    .toolkit(toolkit);

let agent = harness::builder()
    .model(chat)
    .workspace(workspace)
    .build()?;
```

`./workspace` 是应用准备的已有目录，`toolkit` 和 `chat` 是已经配置的实例。
Local 默认自带列目录、读取、写入、精确编辑、Glob、Grep 和单文件删除七种基础工具，无需逐个挂载。
命令能力只有显式配置执行环境后才自动提供；与自定义工具共用同一个 Toolkit、权限和执行机制：

```rust
let workspace = core::workspace::Local::new("./workspace")?
    .sandbox(halo_agents::sandbox::Local::default()) // 可选；不配置则没有命令工具
    .toolkit(toolkit); // 可选的自定义 Tool、MCP、Skill
```

命令与文件工具使用同一目录，默认文件工具的删除不递归删除目录。Local 命令具有宿主机权限，不是隔离沙箱；
超时、取消和输出限制沿用所配置的 Sandbox，取消不回滚已发生的修改。

Docker Workspace 让文件工具和命令共用一个持续存在的容器，构造不启动容器，首次异步操作时初始化：

```rust
let workspace = core::workspace::Docker::new(core::workspace::docker::Config {
    sandbox: core::sandbox::docker::Config {
        image: "应用已准备的镜像".into(), // image 与资源限制复用 Sandbox 配置
        ..Default::default()
    },
    workdir: "/workspace".into(),
    host_workdir: None, // 需要挂载时显式提供已有宿主机目录
    env: Default::default(),
})?.toolkit(toolkit);
```

同样交给 Harness 的 `.workspace(workspace)`，自动提供七种文件工具与命令工具，不逐个挂载。
镜像须自带 `/bin/sh`、`sleep/cat/mkdir/mv/rm/find`（find 支持 NUL 输出及深度限制）；框架不拉取、构建或安装。
默认无宿主机挂载、无网络，复用 Sandbox 的非 root、只读根文件系统与资源限制。显式挂载的目录权限由应用准备。
直接使用 Workspace 时可以异步 `close()`；工具持有环境到最后一个拥有者释放，Drop 清理需 Tokio 仍存活且为尽力执行。
超时/输出限制同时约束文件协议与命令。执行中超时、取消、输出超限或连接异常会关闭该容器，
避免未停止的命令在后续文件操作背后继续写入；不会自动换一个空容器重试。
这不回滚修改，未挂载的临时文件随容器释放，宿主机挂载的数据不删除。

协议回归：`cargo test --offline --test workspace_docker`。真实容器测试需显式设置已有的
`HEART_WORKSPACE_TEST_IMAGE`，再运行 `cargo test --test workspace_docker docker_workspace_real_ -- --ignored`；不自动拉取镜像。
单次 Docker Sandbox 的真实测试使用 `HALO_SANDBOX_TEST_IMAGE`，运行
`cargo test --test sandbox_docker -- --ignored`。已使用本机 Docker 29.7.2 与已有的 `redis:7-alpine`
验证上述四项真实测试；镜像仅作为隔离命令环境，不启动 Redis 服务。测试只清理自有容器，不删除宿主机挂载数据。

Workspace 的存储模式区分本地文件（Local）与数据库（Server）。Server 直接操作数据库，不模拟文件路径；
默认只提供 write(data)、read(id)、update(id, data)、delete(id) 四个基础工具，复杂查询由用户实现自己的 Tool，
继续复用 Toolkit、Permission 和 Message/Error。公共 Workspace 只保留 definition()，文件方法归具体 Local；
不再提供公开的 Workspace InMemory 实现或兼容别名；文件测试使用临时 Local 目录。
这不影响 Memory、Checkpoint 等其他组件的内存实现。
ES 后端已提供 `core::workspace::es::new(client, index)?.toolkit(toolkit)`，client 为应用初始化的官方 `elasticsearch::Elasticsearch`。
当前接入官方 9.1.0-alpha.1 客户端，已在单节点 Elasticsearch 9.5.4 上完成真实基础操作验收；其他主版本、TLS/鉴权及多节点故障场景不在这次验收范围。
不增加 config.filter/config.schema，不内置固定字段补齐策略。参数校验沿工具 Schema，业务校验和补齐可用用户 Hook。
索引和 mapping 由应用准备；四个工具及 Toolkit/Harness 接线均经过协议与真实集群回归。
write/update 先检查索引存在；write 使用 `_create` 防覆盖，再通过实时 GET 返回实际落库记录。
update 不 upsert，delete 物理删除；仅 `doc_as_upsert=false` 不能阻止 ES 为缺失索引自动建空索引，因此更新同样保留存在检查。
工具内部不自动重试；请求发出后的取消/连接中断不代表写入回滚。若需严格禁止并发删索引后的自动重建，应由应用配置 ES 权限/自动创建策略。

本地协议回归：`cargo test --offline --test workspace_es`。
真实集群测试默认忽略；应用预先准备隔离测试索引后，设置 `HEART_ES_URL`、`HEART_ES_TEST_INDEX`，按需设置 `HEART_ES_API_KEY`（已编码 API Key），
再运行 `cargo test --test workspace_es es_real_cluster_crud_smoke -- --ignored --exact`。该测试不建索引或改 mapping，只清理它生成的记录 ID。
扩展验收运行 `cargo test --test workspace_es_real -- --ignored`，覆盖 i64 极值、并发新增冲突、缺索引、实际 Harness/权限和 Pipeline 结果。
完整扩展组还需 `HEART_ES_PIPELINE_TEST_INDEX` 指向独立的既有索引：其默认 ingest pipeline 将 `content` 转小写，
并设置 `metadata.ingested=true`。索引、mapping 与 pipeline 都由测试调用方准备，不由 Workspace 创建。

## 当前状态

当前版本已提供 Chat/ReAct、Agent 计划图、Tool/MCP/Skill、短期与长期 Memory、压缩、权限、中间件、Checkpoint、基础文档处理及 Sandbox 能力。Workflow 支持普通/条件关系、Join、动态 Fan-out、Loop 和内嵌子图；实际内嵌 Agent、父子 Agent 与子图的现场保存、反馈路由和恢复已有回归验收。

项目使用 Rust 2024 Edition，要求 Rust 1.99.1 或更新版本；当前验证工具链为 Rust 1.99.1。

Agent 和 Workflow 的 `run(...)` 均返回统一的 `Event::Delta/Complete` 流。调用方持续消费流来驱动运行；Workflow 中的嵌套子图共用这一条输出流。完整业务结果保留在 Runtime 状态中，事件副本不包含供下游使用的 `values`。

Agent 使用 `agent.run(input, stream, cancellation)`：`stream = true` 对外交付 Delta 和 Complete，`false` 仅交付 Complete；两种模式的返回类型相同，多次模型或工具调用的 Complete 不会被合并丢弃。Workflow 的运行参数不因此改变。

运行中的工具审批通过原对象的 `reply(...)` 提交：Agent 接收 `Input`，Workflow 接收 `Messages`，均附带 `Cancellation`，并通过消息元数据中的 `chat_id`、`reply_id` 关联实际提问。提交成功只表示反馈已接收；继续消费原 `run` 流，才能取得后续执行结果。不需要单独调用子 Agent、取得反馈句柄或另开输出流。

Workflow 的原流独占本次执行现场；释放流后才能直接读写 State，流存活期间重复 `run` 或直接获取快照会返回冲突。运行中反馈通过原对象的 `reply` 提交，继续消费同一运行流；原调用已经结束时，才从保存的完整 Checkpoint 重新挂载现场，再从已有等待步骤继续。独立 Agent 由配置的 Checkpoint 接线读取现场；Workflow 使用已有的 `Runtime::from_snapshot(...)` 接入完整快照，再走 `run/reply`，不是仅配置保存后端就能自动重建 Runtime。

独立 Chat/ReAct、计划任务、SubAgent 和 Workflow 内嵌 Agent 已支持从顶层完整 Checkpoint 恢复审批及普通反馈等待，包括跨进程 JSON 交接。按原 `chat_id`、`reply_id` 和保存的实际节点层级定向消费，不重放已成功的模型调用、委派或兄弟任务，保留原执行身份和 Usage；子 Agent 不启动独立存储生命周期。反馈严格 FIFO，所需份数由发起等待的步骤固定，收齐后才继续。恢复不代表续传原网络连接、还原任意函数栈或恢复未保存的数据。

普通异步函数可直接交给 Workflow 的 `add_node`，不需要包装类。可选的 `#[node]` 在定义位置校验契约；已有组件可通过 `#[node(call = method)]` 或 `#[node(signal = method)]` 生成同一 Node 契约的调用胶水，原生方法仍可独立使用。

Memory 后台压缩与长期提取的处理现场保存在 Session metadata；使用持久化 Session 后端时，重启后可由下次请求继续未完成阶段，复用已保存结果和写入 ID。默认内存后端不具备落盘能力；这份后台现场不属于 Runtime Checkpoint。

取消后应继续消费原事件流直到收口，以接收实际取消 Complete 和已发生的 Usage。直接丢弃流只释放当前执行，不隐式启动异步写库；恢复只认已经成功保存的 Checkpoint。完整执行数据保存在各自 Runtime State，Trace 是独立的最终审计副本，不能当作恢复来源。

扩展领域 Parser、PDF/Office/OCR 等解析能力留到 V2；Python 函数接入暂缓；TTS 与完整多模态契约留到 V3+。这些范围不作为当前版本已经实现的能力。

## 构建与测试

```bash
cargo check
cargo test
```

完整默认验收及静态检查：

```bash
cargo test --all-targets --all-features
cargo test --all-features --doc
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

依赖已缓存时可加 `--offline`。MCP/Provider 协议测试使用本机临时服务；真实 Docker/ES 测试需要应用准备运行环境并显式启用，默认忽略不代表这些真实服务已经在本次验收中运行。

## 许可证

MIT
