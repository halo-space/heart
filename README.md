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
当前 Core 已提供 Local 的列目录、读取、写入、精确编辑、Glob 匹配和正则搜索六种默认文件工具。
这些工具与自定义工具共用 Toolkit 和已有权限、执行机制；删除和命令工具不会自动添加。
显式删除/命令工具及共享文件环境的 Docker Workspace 仍在实现中。

Workspace 的存储模式区分本地文件（Local）与数据库（Server）。Server 直接操作数据库，不模拟文件路径；
默认只提供 write(data)、read(id)、update(id, data)、delete(id) 四个基础工具，复杂查询由用户实现自己的 Tool，
继续复用 Toolkit、Permission 和 Message/Error。公共 Workspace 只保留 definition()，文件方法归具体 Local；
不再提供公开的 Workspace InMemory 实现或兼容别名；文件测试使用临时 Local 目录。
这不影响 Memory、Checkpoint 等其他组件的内存实现。
ES 后端已提供 `core::workspace::es::new(client, index)?.toolkit(toolkit)`，client 为应用初始化的官方 `elasticsearch::Elasticsearch`。
当前接入官方 9.1.0-alpha.1 客户端，真实集群及其他 ES 主版本兼容性尚未验收。
不增加 config.filter/config.schema，不内置固定字段补齐策略。参数校验沿工具 Schema，业务校验和补齐可用用户 Hook。
索引和 mapping 由应用准备；四个工具的实现及 Toolkit/Harness 接线已落地，真实 ES 集群验收尚未运行。
写入先检查索引存在，使用 `_create` 防止覆盖，再通过实时 GET 返回落库记录；update 不 upsert，delete 物理删除。
工具内部不自动重试；请求发出后的取消/连接中断不代表写入回滚。若需严格禁止并发删索引后的自动重建，应由应用配置 ES 权限/自动创建策略。

本地协议回归：`cargo test --offline --test workspace_es`。
真实集群测试默认忽略；应用预先准备隔离测试索引后，设置 `HEART_ES_URL`、`HEART_ES_TEST_INDEX`，按需设置 `HEART_ES_API_KEY`（已编码 API Key），
再运行 `cargo test --test workspace_es es_real_cluster_crud_smoke -- --ignored --exact`。该测试不建索引或改 mapping，只清理它生成的记录 ID。

## 当前状态

项目正在持续开发中，当前已经包含基础 Agent/Workflow 结构、Chat Provider、Tool/MCP/Skill、短期与长期 Memory、消息与事件契约、文档处理组件以及本地 Sandbox 等能力。完整的 runtime::agent 执行图、Workflow 持久化恢复和更多 Provider 仍在逐步完善。

项目使用 Rust 2024 Edition，最低目标版本为 Rust 1.98。

Agent 和 Workflow 的 `run(...)` 均返回统一的 `Event::Delta/Complete` 流。调用方持续消费流来驱动运行；Workflow 中的嵌套子图共用这一条输出流。完整业务结果保留在 Runtime 状态中，事件副本不包含供下游使用的 `values`。

Agent 使用 `agent.run(input, stream, cancellation)`：`stream = true` 对外交付 Delta 和 Complete，`false` 仅交付 Complete；两种模式的返回类型相同，多次模型或工具调用的 Complete 不会被合并丢弃。Workflow 的运行参数不因此改变。

运行中的工具审批通过原对象的 `reply(...)` 提交：Agent 接收 `Input`，Workflow 接收 `Messages`，均附带 `Cancellation`，并通过消息元数据中的 `chat_id`、`reply_id` 关联实际提问。提交成功只表示反馈已接收；继续消费原 `run` 流，才能取得后续执行结果。不需要单独调用子 Agent、取得反馈句柄或另开输出流。

Workflow 的原流独占本次执行现场；释放流后才能直接读写 State，流存活期间重复 `run` 或直接获取快照会返回冲突。运行中审批与原调用结束后的现场恢复是不同路径；后者的待审批消费（含跨进程恢复）仍在完善中。

独立 Chat/ReAct 的根工具审批、独立 ReAct 计划任务及其子 Agent 的工具审批，已支持在新 `run` 中从对应的顶层 Checkpoint 恢复，按原 `chat_id`、`reply_id` 消费批准/拒绝。恢复沿保存的实际节点层级继续，不重放已完成的模型调用、委派或成功兄弟任务，保留原执行身份和 Usage；其他问题仍可通过新流上的 `reply` 回复。子 Agent 不启动独立存储生命周期。Workflow 内嵌审批的跨调用恢复仍在完善中。

普通异步函数可直接交给 Workflow 的 `add_node`，不需要包装类。可选的 `#[node]` 在定义位置校验契约；已有组件可通过 `#[node(call = method)]` 或 `#[node(signal = method)]` 生成同一 Node 契约的调用胶水，原生方法仍可独立使用。

Memory 后台压缩与长期提取的处理现场保存在 Session metadata；使用持久化 Session 后端时，重启后可由下次请求继续未完成阶段，复用已保存结果和写入 ID。默认内存后端不具备落盘能力；这份后台现场不属于 Runtime Checkpoint。

## 构建与测试

```bash
cargo check
cargo test
```

## 许可证

MIT
