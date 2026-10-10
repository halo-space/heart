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

项目正在持续开发中，当前已经包含基础 Agent/Workflow 结构、Chat Provider、Tool/MCP/Skill、短期与长期 Memory、消息与事件契约、文档处理组件以及本地 Sandbox 等能力。完整的 AgentGraph、Workflow 持久化恢复和更多 Provider 仍在逐步完善。

项目使用 Rust 2024 Edition，最低目标版本为 Rust 1.98。

## 构建与测试

```bash
cargo check
cargo test
```

## 许可证

MIT
