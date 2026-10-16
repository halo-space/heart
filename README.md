# Heart

Heart 是一个使用 Rust 构建的 Agent 与 Workflow 执行框架，目标是用统一的消息、事件、组件和运行时契约，支持从简单对话 Agent 到复杂工作流的组合与执行。

项目提供模型、工具、记忆、文档处理、权限、中间件、检查点、Sandbox 等基础能力，并将组件契约、具体实现和运行时执行状态分层组织，便于替换模型供应商、存储后端和执行策略。

## 目录结构

```text
src/
├── components/   公共组件契约，例如 Model、Tool、Memory、Message、Event
├── core/         组件的具体实现，例如 Provider、ReAct Agent、本地存储和文档处理
├── harness/      组合组件实现，组织 Hook、权限、记忆与压缩等运行流程
└── runtime/      Agent 与 Workflow 的运行状态、图结构和执行控制
```

组件层定义可复用的接口和数据结构，`core` 提供具体实现。`harness` 使用配置好的组件实例组织运行流程，`runtime` 管理执行图、节点状态、执行记录、Usage 和恢复现场。应用通过现有 Agent 构造入口配置组件，并调用 `agent.run(...)`；这些内部职责分工由框架完成。

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
