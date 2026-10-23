# Heart

Heart 是一个使用 Rust 构建的 Agent 与 Workflow 执行框架，提供统一的消息、事件、组件契约和运行时状态。
组件契约、具体实现和运行时相互分层，可替换模型供应商、工具、Memory、Checkpoint、Workspace 和存储后端。

当前版本要求 Rust 1.99.1 或更新版本，Cargo 包名和 Rust crate 名为 `halo`。

公开 Agent 入口统一从 `halo::agent` 开始：Agent 契约使用 `halo::agent`，原有具体实现层通过
`halo::agent::core` 访问，Workflow 通过 `halo::agent::workflow` 访问。`src/core` 和 `src/runtime`
仍按原有内部层级组织，只由公开入口重新导出。

## 使用手册

完整的中文使用手册（Agent、Workflow、模型、Tool、MCP、Skill、Workspace、Memory、Checkpoint、事件流、等待恢复和测试）：

[阅读中文使用手册](docs/USAGE.zh-CN.md)
