# Heart

Heart 是一个用 Rust 构建的 Agent 与 Workflow 开发框架。它从公共组件契约、具体实现到异步运行时提供完整的组合能力，帮助应用构建可嵌入、可恢复、可扩展的智能体和工作流。

项目覆盖模型接入、工具执行、MCP、Skill、Workspace、Memory、Checkpoint、事件流、权限、压缩、Server 和协议适配。应用可以根据自己的产品场景选择组件和存储后端，再通过统一的 `halo::agent` 入口接入现有系统。

> **版本状态：Alpha（0.1.x 开发阶段）**
>
> 当前版本是早期 Alpha 版本，不承诺 API 稳定性。公共契约、组件实现和运行时行为仍可能在后续版本中调整。

## 当前定位

- `components`：公共契约，例如 Agent、Model、Message、Event、Tool、Memory、Checkpoint 和 Workspace。
- `core`：可替换的具体实现，例如 Provider、Chat/ReAct、Workspace、Sandbox 和 Memory 后端。
- `runtime`：Agent 与 Workflow 的状态、图关系、执行和恢复。
- `harness`：面向长任务 Agent 的高阶组合能力，例如 Memory、压缩、权限和生命周期编排。

公开入口统一从 `halo::agent` 开始：

```rust
use halo::agent::{self, Cancellation, Input};
```

Agent、Workflow 和公共运行类型都通过这个入口使用；具体实现分别从 `halo::agent::core` 与 `halo::agent::workflow` 访问。Cargo 包名是 `halo-agent`，Rust 库名是 `halo`，当前要求 Rust 1.99.1 或更新版本。

## 快速开始

完整示例位于 `examples/`，可以先在仓库根目录验证编译：

```bash
cargo check --offline --examples
cargo run --offline --example agent_middleware
cargo run --offline --example workflow_events
```

上面的离线示例使用确定性的本地替身，只用于验证 Agent/Workflow 契约，不会产生大模型回答。要调用真实模型，请使用 OpenAI-compatible Provider 示例：

需要模型或外部服务的示例只从环境变量读取地址和密钥，不把 Secret 写入源码。VLLM 示例使用 OpenAI-compatible `/v1/chat/completions` 接口：

```bash
HEART_VLLM_BASE_URL=https://example.invalid/v1 \
HEART_VLLM_API_KEY=... \
HEART_VLLM_MODEL=... \
cargo run --example provider_vllm
```

## 使用手册

中文详细手册是项目的主入口，按设计章节组织，包含 Agent、Workflow、模型 Provider、Tool、MCP、Skill、Workspace、Memory、Checkpoint、事件流、等待恢复、Server、A2A、Team、测试和可运行配方：

- [中文详细使用手册](docs/USAGE.zh-CN.md)
- [English User Manual](docs/USAGE.en.md)

手册中的示例与仓库 `examples/` 一一对应；新增能力应同时补充中文说明、英文说明和可运行示例。

## 示例索引

| 主题 | 示例 |
| --- | --- |
| Chat Agent、Memory | [`examples/agent_memory.rs`](examples/agent_memory.rs) |
| ReAct、Tool、Permission | [`examples/agent_permission.rs`](examples/agent_permission.rs) |
| Middleware | [`examples/agent_middleware.rs`](examples/agent_middleware.rs) |
| Workflow 事件流 | [`examples/workflow_events.rs`](examples/workflow_events.rs) |
| Checkpoint 保存与恢复 | [`examples/checkpoint.rs`](examples/checkpoint.rs) |
| 基础组件契约（Function、Prompt、Search、Embed、Sandbox、Event） | [`examples/component_contracts.rs`](examples/component_contracts.rs) |
| Planner、Evaluator | [`examples/planning_evaluation.rs`](examples/planning_evaluation.rs) |
| Prompt 与压缩 | [`examples/compress.rs`](examples/compress.rs) |
| Session/Memory CRUD | [`examples/short_memory.rs`](examples/short_memory.rs) |
| OpenAI-compatible Chat | [`examples/provider_chat.rs`](examples/provider_chat.rs) |
| DeepSeek Chat Completions | [`examples/provider_deepseek.rs`](examples/provider_deepseek.rs) |
| VLLM Chat Completions | [`examples/provider_vllm.rs`](examples/provider_vllm.rs) |

更多示例命令、依赖说明和故障排查见[中文详细使用手册](docs/USAGE.zh-CN.md)。

## 许可证

MIT License。

## 鸣谢

感谢 [Jancd](https://github.com/Jancd) 对项目提出的建议、审阅代码并帮助改进设计与实现。
