# Heart User Manual

This is the practical manual for Heart. It follows the design chapter order `00–18` and explains how an application uses the current Rust API. It does not change the design documents. The source code is the final authority for the current public signatures.

Heart is an embeddable Rust library for Agent and Workflow execution. It provides contracts and implementations for models, tools, MCP, Skills, Workspace, Memory, Checkpoint, compression, permissions, documents, and protocol adapters. It does not provide a hosted UI, account system, or database service.

The current release is Alpha `0.1.x` and requires Rust `1.99.1` or newer. The Cargo package is `halo-agent`; the Rust library is imported as `halo`.

## Chapter map

| Chapters | Design source | Practical topic |
| --- | --- | --- |
| 00–01 | [`00`](../design/chapters/00-architecture.zh-CN.md), [`01`](../design/chapters/01-overview.zh-CN.md) | Architecture, installation, first run |
| 02–03 | [`02`](../design/chapters/02-runtime-execution.zh-CN.md), [`03`](../design/chapters/03-agent-execution.zh-CN.md) | Runtime, Chat, ReAct, reply, Team |
| 04–05 | [`04`](../design/chapters/04-message-contract.zh-CN.md), [`05`](../design/chapters/05-checkpoint.zh-CN.md) | Input, Message, Event, Checkpoint |
| 06–08 | [`06`](../design/chapters/06-workflow-execution-relations.zh-CN.md), [`07`](../design/chapters/07-workflow-runtime-core.zh-CN.md), [`08`](../design/chapters/08-workflow-graph-relations.zh-CN.md) | Workflow graph, relations, State, subgraphs |
| 09–12 | [`09`](../design/chapters/09-framework-comparison.zh-CN.md), [`10`](../design/chapters/10-framework-comparison-relations.zh-CN.md), [`11`](../design/chapters/11-decision-checklist.zh-CN.md), [`12`](../design/chapters/12-open-questions.zh-CN.md) | Framework comparison, decisions, limits |
| 13–15 | [`13`](../design/chapters/13-component-catalog.zh-CN.md), [`14`](../design/chapters/14-application-persistence.zh-CN.md), [`15`](../design/chapters/15-todo.zh-CN.md) | Component catalogue, persistence, validation |
| 16–18 | [`16 A2A`](../design/chapters/16-a2a-protocol.zh-CN.md), [`16 compression`](../design/chapters/16-compression-plan.zh-CN.md), [`17`](../design/chapters/17-server.zh-CN.md), [`18`](../design/chapters/18-team.zh-CN.md) | Compression, A2A, HTTP Server, Team |

Runnable examples are in [`examples/`](../examples/). The Chinese manual is [`USAGE.zh-CN.md`](USAGE.zh-CN.md).

---

## 00–01. Architecture and first run

Heart is one crate with this dependency direction:

```text
components -> public contracts and data
core       -> concrete component implementations
runtime    -> Agent/Workflow state and async execution
harness    -> higher-level composition
server     -> HTTP/A2A/JSON-RPC/SSE bindings
```

Use the stable entry point:

```rust
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload};
```

Do not depend on internal `core::agent` or `runtime` paths for ordinary application code. Cargo setup:

```toml
[dependencies]
halo-agent = "0.1"
futures-util = "0.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

The smallest real Chat Agent is:

```rust
let provider = halo::agent::provider::deepseek::DeepSeek::builder()
    .base_url(std::env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".into()))
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .build()?;
let chat = provider.chat(halo::agent::model::chat::Config::new("deepseek-flash"))?;
let agent = halo::agent::chat().model(chat).build()?;
let input = Input {
    metadata: Default::default(),
    payload: vec![Payload {
        r#type: "text".into(),
        data: serde_json::json!({"value": "Introduce Heart in one sentence"}),
        mime_type: None,
    }],
};
let mut events = agent.run(input, true, Cancellation::new()).await?;
while let Some(event) = futures_util::StreamExt::next(&mut events).await {
    println!("{:?}", event?);
}
```

The `stream` argument is fixed per invocation: `true` exposes `Delta` and `Complete`, while `false` exposes complete results only. A run is finite; a waiting run receives later user input through `reply`.

Real provider examples: [`provider_deepseek.rs`](../examples/provider_deepseek.rs) and [`provider_vllm.rs`](../examples/provider_vllm.rs).

---

## 02–03. Runtime and Agent

The application-owned top-level identity is `chat_id`. `Cancellation` belongs to one run; create a fresh value for every independent run and clone it only for work in that run.

```text
Event Stream -> external observation
Runtime State -> execution and recovery truth
Checkpoint -> persisted Runtime State
Memory -> conversation history and long-term records
OpenTelemetry Trace -> optional observability projection, not short memory
```

Chat Agent can use a model, tools, MCP, Skills, Permission, middleware, memory, checkpoint, and an evaluator:

```rust
let agent = halo::agent::chat()
    .model(chat)
    .system_prompt("You are a precise assistant")
    .tools(tools)
    .mcps(servers)
    .skills(skills)?
    .permission(permission)
    .middleware(retry)
    .checkpoint(checkpoint)
    .memory(memory)
    .with_max_concurrency(100)
    .build()?;
```

ReAct has the same entry style and additionally runs the model/tool loop and built-in Planner:

```rust
let agent = halo::agent::react()
    .model(chat)
    .tools(tools)
    .mcps(servers)
    .skills(skills)?
    .permission(permission)
    .evaluator(evaluator)
    .with_max_iters(20)
    .with_max_concurrency(100)
    .build()?;
```

The model chooses registered tools and arguments. Heart's internal dispatch layer exposes registered Tool, MCP, and Skill descriptions and executes the selected capability. Tool `before`, `call`, `after`, and `eval` are one optional lifecycle. Tool errors are returned directly to the next model context; Heart does not invent an extra `tool_result(error)` envelope.

SubAgents are already-built Agents passed through `.subagents(...)`. Team members are fixed at build time; dynamic member creation is outside the current scope. See [`agent_live_subagent.rs`](../examples/agent_live_subagent.rs) and [`agent_live_team.rs`](../examples/agent_live_team.rs).

---

## 04. Input, Message, and Event

Business input is `agent::Input`, not model `Messages`:

```rust
let input = halo::agent::Input {
    metadata: serde_json::Map::from_iter([
        ("chat_id".into(), 1001.into()),
        ("session_id".into(), 2001.into()),
    ]),
    payload: vec![halo::agent::Payload {
        r#type: "text".into(),
        data: serde_json::json!({"value": "What is the weather?"}),
        mime_type: None,
    }],
};
```

`Payload.type` may be `text`, `image`, `audio`, `file`, or `json`; the corresponding value is in `data`, with an optional `mime_type`. Runtime projects the envelope into model `Messages`.

Each `Message` contains `role`, `content`, `values`, `metadata`, and optional `usage`. The common roles are `system`, `developer`, `user`, `assistant`, `tool`, and `function`. Provider-specific reasoning is normalized to a `content` part with `type = "think"`; normal output uses `type = "text"`.

All stream-capable layers use the shared event:

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

`Delta` is observation only. `Complete` is the complete delivery for one model or runtime call. Do not reconstruct a fake complete message from an interrupted delta stream.

---

## 05. Checkpoint and recovery

Checkpoint stores the complete Runtime State as JSON. It is separate from Chat history and Memory; OpenTelemetry Trace is only an observability projection. The contract is asynchronous `save`, `load`, `delete`, and `validate_state`.

```rust
let checkpoint = std::sync::Arc::new(halo::agent::core::checkpoint::InMemory::new());
let runtime = halo::agent::workflow::Runtime::new(chat_id, workflow.clone())?
    .with_checkpoint(checkpoint.clone());
runtime.save_checkpoint(&*checkpoint, &Cancellation::new()).await?;
let snapshot = checkpoint.load(chat_id, &Cancellation::new()).await?;
let restored = halo::agent::workflow::Runtime::from_snapshot(chat_id, workflow, snapshot)?;
```

Restore uses the same compiled definition and version, then loads the saved scene. Runtime reads saved facts instead of inferring success from a missing or present output. Waiting queues, attempts, failures, and completed nodes remain part of the State.

No configured Checkpoint means no cross-process recovery. [`checkpoint.rs`](../examples/checkpoint.rs) is the complete example.

---

## 06–08. Workflow graph and execution

Ordinary async functions can be added directly as nodes:

```rust
use halo::agent::workflow::{WorkflowBuilder, graph::{START, END}};
use halo::agent::{Cancellation, Error, Messages};
use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, JsonSchema)]
struct Answer { value: String }

let mut builder = WorkflowBuilder::new("example");
builder.add_node("answer", |_: Messages, _: Cancellation| async {
    Ok::<_, Error>(Answer { value: "done".into() })
})?;
builder.add_edge(START, "answer")?;
builder.add_edge("answer", END)?;
let workflow = builder.compile()?;
let mut runtime = halo::agent::workflow::Runtime::new(1001, workflow)?;
```

Relations are explicit:

```rust
builder.add_edge(START, "a")?;
builder.add_join(["b", "c"], "merge")?;
builder.add_conditional_edges("route", route, ["b", "c"])?;
builder.add_fan_out_edges("split", "worker")?;
builder.add_loop_edge("retry", "work", |messages| should_retry(messages))?;
```

An ordinary edge activates after a successful source; a Join waits for all listed sources. Conditional routes must select a declared target. Fan-out branches can run concurrently. `START` and `END` are graph boundary endpoints. `add_agent_node` embeds a built Agent; `add_graph_node` embeds a compiled subgraph with an explicit boundary.

Run and observe the shared event stream:

```rust
let mut events = runtime
    .run(Messages::default(), next_exec_id, Cancellation::new())
    .await?;
while let Some(event) = futures_util::StreamExt::next(&mut events).await {
    println!("{:?}", event?);
}
println!("state = {:?}", runtime.state());
```

Workflow State contains graph, node status, executions, attempts, values, waiting items, and recovery data. The compiled graph is immutable; the Runtime owns one mutable State for one `chat_id`.

---

## 09–10. Framework comparison and DSL

Heart takes useful boundaries from LangGraph (`START/END`, conditional edges, subgraphs, checkpoint), Eino (component contracts and graph composition), and AgentScope (Toolkit, Workspace, Permission, Plan, and Harness composition). Their DSLs are not copied into Runtime State.

An external Dify, Coze, or application DSL should be translated at the outer boundary:

```text
DSL JSON -> validate fields -> bind application capabilities
         -> WorkflowBuilder or Agent Plan -> compile -> Runtime
```

Unknown functions, tools, Agents, or subgraphs must fail at binding/plan submission. Heart does not add a public Registry, Capabilities, or dynamic runtime registration table. A2A/AG-UI/JSON-RPC/SSE adapters live at the protocol/HTTP layer and convert existing Events.

---

## 11. Fixed usage rules

- Use `halo::agent` as the public entry; do not use compatibility `halo::core::agent`.
- Use `chat_id` for the top-level execution identity.
- Use `agent::Input` for business input and `model::chat::Request` for provider calls.
- Treat `Delta` as live observation and `Complete` as complete delivery.
- Normalize model reasoning to `think` content parts.
- Return tool errors directly to the next model context; do not add a second error envelope.
- Keep Tool hooks together; Runtime/middleware owns retry and execution state.
- Keep Checkpoint (recovery scene), Memory (history/long-term records), and Chat (external close) separate; OpenTelemetry Trace is observability only and does not add another short-memory record.
- Unconfigured capabilities are absent; Heart does not silently create a model, database, Sandbox, MCP, or Memory.
- Public async contracts use native Rust async functions.

---

## 12. Current limits

The current Alpha does not promise a complete OCR/layout parser suite, every provider's TTS/multimodal extension, distributed task queues and locks, an application DSL editor, or dynamic Team member creation. These are extension/application concerns and do not prevent custom implementations of the public contracts.

---

## 13. Component catalogue

Provider initialization and model binding are separate:

```rust
let provider = halo::agent::provider::vllm::Vllm::new(config)?;
let chat = provider.chat(halo::agent::model::chat::Config::new("model"))?;
```

Available provider implementations include DeepSeek, VLLM/OpenAI-compatible, OpenAI, Gemini, and Zhipu. Tool definitions must have an object-root parameter schema. `before` may validate or fill arguments; `after` may mutate the result Message; `eval` may return business feedback.

Workspace selects local or server data sources. Local uses local files; server uses the configured database/remote store. Skills can be discovered by the Workspace loader; MCP connections and discovery should complete before Agent construction.

Prompt templates render messages and named history variables. Memory uses `memory::Config::new(short, long)`, optional scopes, compression, extraction, and an application ID function. Checkpoint, Sandbox, Document, Search, Embed, Permission, and Middleware are all independent contracts with optional concrete Core implementations. See [`component_contracts.rs`](../examples/component_contracts.rs).

---

## 14. Short/long memory and persistence

Short memory consists of Session and Chat. Session aggregates a stable conversation thread; Chat stores one external input and final Assistant Message. Full runtime State belongs to Checkpoint. `runtime::trace` is optional OpenTelemetry observability, not another short-memory record.

Long memory uses fixed types such as `summary`, `fact`, and `preference`, with `id`, owner IDs, `from/to`, `content`, version, timestamps, and optional metadata. The application calls `model::embed` first if it wants vectors; storage receives text and an optional vector and never owns an Embed instance.

Compression is enabled only by configuring a compressor. It is checked on a user-triggered request and runs in the background. The threshold uses Chat usage after the current summary coverage; the current summary is replaced, not accumulated as a second summary.

---

## 15. Testing and publishing

```bash
cargo fmt --check
cargo check --all-targets
cargo test --lib --no-fail-fast
cargo clippy --lib --all-features -- -D warnings
```

Provider/MCP/ES/Sandbox acceptance tests read environment variables and should use real services. Deterministic local tests validate contracts and state transitions; they are not model acceptance tests. Before publishing, run `cargo package --allow-dirty` and `cargo publish --dry-run`.

---

## 16. Compression and A2A

Compression creates the current Session summary without rewriting original Chats. It reads Chats after the covered range, uses their recorded input/output usage, and records the compression call usage for display. The active request is not replaced in the middle of a run; later requests use the new summary plus uncovered Chats.

A2A maps remote `context_id` to the application's `session_id`, converts Agent Events to JSON-RPC/SSE, and sends user replies back through `reply`. It is a protocol adapter, not another Runtime.

---

## 17. Server and HTTP

The Axum binding exposes JSON-RPC methods such as `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`, and `tasks/resubscribe`. SSE is a transport for the existing Event stream.

```rust
let service = std::sync::Arc::new(halo::agent::server::a2a::Service::with_store(agent, store));
let router = halo::agent::server::axum::router(service);
let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
axum::serve(listener, router).await?;
```

The external Task Store is a projection. Agent/Workflow Runtime remains the source of execution truth and the HTTP layer does not duplicate State or waiting logic.

---

## 18. Team and multi-Agent

Team members are selected during construction. A Leader routes work to fixed members; members execute with their own Agent state and results return through the Team event stream. Dynamic member creation is not part of the current boundary.

```rust
let researcher = halo::agent::react()
    .model(research_chat)
    .tools(vec![search_tool])
    .build()?;
let writer = halo::agent::chat()
    .model(writer_chat)
    .build()?;
let team = halo::agent::team()
    .leader(halo::agent::react().model(leader_chat))
    .member(halo::agent::SubAgent::new("researcher", researcher)?)
    .member(halo::agent::SubAgent::new("writer", writer)?)
    .build()?;
```

See [`agent_live_team.rs`](../examples/agent_live_team.rs) and [`agent_live_three_subagents.rs`](../examples/agent_live_three_subagents.rs).

## Example index

- [`agent_live_deepseek.rs`](../examples/agent_live_deepseek.rs): real DeepSeek Chat Completions.
- [`agent_live_tool.rs`](../examples/agent_live_tool.rs): real VLLM plus a real local tool.
- [`agent_memory.rs`](../examples/agent_memory.rs): Agent and Memory.
- [`agent_permission.rs`](../examples/agent_permission.rs): Permission.
- [`workflow_events.rs`](../examples/workflow_events.rs): Workflow events and State.
- [`checkpoint.rs`](../examples/checkpoint.rs): save and restore.
- [`component_contracts.rs`](../examples/component_contracts.rs): low-level contracts.
- [`provider_deepseek.rs`](../examples/provider_deepseek.rs): DeepSeek provider.
- [`provider_vllm.rs`](../examples/provider_vllm.rs): VLLM provider.
