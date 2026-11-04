# Heart User Manual

Heart is a Rust library for Agent and Workflow execution. It provides contracts and implementations for models, tools, MCP, Skills, Workspace, Memory, Checkpoint, compression, permissions, documents, and protocol adapters. Applications can embed these capabilities in an existing product, while the optional Server layer provides HTTP access.

The current release is Alpha `0.1.x` and requires Rust `1.99.1` or newer. The Cargo package is `halo-agent`; the Rust library is imported as `halo`.

Runnable examples are in [`examples/`](../examples/). The Chinese manual is [`USAGE.zh-CN.md`](USAGE.zh-CN.md).

---

## Architecture and first run

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

## Runtime and Agent

The application-owned top-level identity is `chat_id`. `Cancellation` belongs to one run; create a fresh value for every independent run and clone it only for work in that run.

Agent and Workflow both have a Runtime, but their public control surfaces differ. Agent keeps its execution scene
inside the Agent object and exposes `agent.run(...)` and `agent.reply(...)`. Workflow exposes
`workflow::Runtime` because applications need to manage graph State, Checkpoint, and waiting recovery. In the
examples below, `workflow` refers to a Runtime instance; `CompiledWorkflow` is only the immutable graph definition.

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

The model chooses configured tools and arguments. Heart's internal dispatch layer exposes configured Tool, MCP, and Skill descriptions and executes the selected capability. Tool `before`, `call`, `after`, and `eval` are one optional lifecycle. Tool errors are returned directly to the next model context; Heart does not invent an extra `tool_result(error)` envelope.

SubAgents are already-built Agents passed through `.subagents(...)`. Team members are fixed at build time; dynamic member creation is outside the current scope. See [`agent_live_subagent.rs`](../examples/agent_live_subagent.rs) and [`agent_live_team.rs`](../examples/agent_live_team.rs).

---

## Input, Message, and Event

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

## Checkpoint and recovery

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

## Workflow graph and execution

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

## Framework comparison and DSL

Heart takes useful boundaries from LangGraph (`START/END`, conditional edges, subgraphs, checkpoint), Eino (component contracts and graph composition), and AgentScope (tool composition, Workspace, Permission, Plan, and Harness composition). Their DSLs are not copied into Runtime State.

An external Dify, Coze, or application DSL should be translated at the outer boundary:

```text
DSL JSON -> core::workflow::dsl parse, validate, and normalize
         -> Core binds Core components
         -> Runtime binds Agent, Workflow, and CompiledWorkflow executors
         -> WorkflowBuilder::compile() -> Runtime
```

Applications provide static execution configuration during initialization; each request only supplies DSL and business input. Unknown functions, tools, Agents, or subgraphs must fail at binding/plan submission. Heart does not add a public Registry, Capabilities, or dynamic runtime registration table. A2A/AG-UI/JSON-RPC/SSE adapters live at the protocol/HTTP layer and convert existing Events.

Page-authored function nodes are created by Core from the DSL and then used like ordinary Nodes. Heart does not look up a local function by `name`, and the DSL does not contain a local path or closure:

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

To place it in a Workflow, use `builder.add_node_with_schema("reload", code, Schema { input: node.schema.input.clone(), output: node.schema.output.clone() })`. This preserves the DSL Schema in the Definition while the executor and Runtime keep the existing validation and error contract; page code receives and returns JSON objects.

---

## Fixed usage rules

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

## Current limits

The current Alpha does not promise a complete OCR/layout parser suite, every provider's TTS/multimodal extension, distributed task queues and locks, an application DSL editor, or dynamic Team member creation. These are extension/application concerns and do not prevent custom implementations of the public contracts.

---

## Component catalogue

Provider initialization and model binding are separate:

```rust
let provider = halo::agent::provider::vllm::Vllm::new(config)?;
let chat = provider.chat(halo::agent::model::chat::Config::new("model"))?;
```

Available provider implementations include DeepSeek, VLLM/OpenAI-compatible, OpenAI, Gemini, and Zhipu. Tool definitions must have an object-root parameter schema. `before` may validate or fill arguments; `after` may mutate the result Message; `eval` may return business feedback.

Workspace selects local or server data sources. Local uses local files; server uses the configured database/remote store. Skills can be discovered by the Workspace loader; MCP connections and discovery should complete before Agent construction.

Prompt templates render messages and named history variables. Memory uses `memory::Config::new(short, long)`, optional scopes, compression, extraction, and an application ID function. Checkpoint, Sandbox, Document, Search, Embed, Permission, and Middleware are all independent contracts with optional concrete Core implementations. See [`component_contracts.rs`](../examples/component_contracts.rs).

`core::sandbox::Local` executes explicit argv commands on the host. It inherits the
application environment, working directory, permissions, and external side effects;
it is not a security boundary and is intended only for trusted work. Page-authored
code or other untrusted input must not be run with `Local`. When process, resource,
or network isolation is required, the application should configure
`core::sandbox::docker::Docker` or a server-side Workspace/Worker. Docker images must be
prepared by the application; Heart never pulls an image or falls back to host execution.

---

## Short/long memory and persistence

Short memory consists of Session and Chat. Session aggregates a stable conversation thread; Chat stores one external input and final Assistant Message. Full runtime State belongs to Checkpoint. `runtime::trace` is optional OpenTelemetry observability, not another short-memory record.

Long memory uses fixed types such as `summary`, `fact`, and `preference`, with `id`, owner IDs, `from/to`, `content`, version, timestamps, and optional metadata. The application calls `model::embed` first if it wants vectors; storage receives text and an optional vector and never owns an Embed instance.

Compression is enabled only by configuring a compressor. It is checked on a user-triggered request and runs in the background. The threshold uses Chat usage after the current summary coverage; the current summary is replaced, not accumulated as a second summary.

---

## Testing and publishing

```bash
cargo fmt --check
cargo check --all-targets
cargo test --lib --no-fail-fast
cargo clippy --lib --all-features -- -D warnings
```

Provider/MCP/ES/Sandbox acceptance tests read environment variables and should use real services. Deterministic local tests validate contracts and state transitions; they are not model acceptance tests. Before publishing, run `cargo package --allow-dirty` and `cargo publish --dry-run`.

`bash scripts/test-real.sh` reads the Git-ignored `.env` and distinguishes three
outcomes: missing required variables exits non-zero, a failing real test keeps a
non-zero status, and an unconfigured optional backend is reported as `SKIP` and
included in the final summary. The script never prints secrets.

Real acceptance matrix:

| Capability | Required environment or resource | Covered tests |
| --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY`; optional `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` | Agent, Workflow, Memory, A2A |
| VLLM/OpenAI-compatible | `HEART_VLLM_BASE_URL`, `HEART_VLLM_API_KEY`, `HEART_VLLM_MODEL` | Provider, Chat/ReAct, Team, Workflow |
| Zhipu Embedding | `HEART_EMBED_API_KEY`; optional `HEART_EMBED_BASE_URL`, `HEART_EMBED_MODEL` | Vector Memory, Semantic Chunker |
| OpenAI Embedding | `OPENAI_API_KEY`; optional `OPENAI_BASE_URL`, `OPENAI_MODEL` | OpenAI Embedding |
| Gemini TTS/multimodal | `GEMINI_API_KEY`; multimodal also needs a real `GEMINI_TEST_IMAGE_DATA_URL` | TTS, multimodal |
| Elasticsearch Workspace | `HEART_ES_URL`, an isolated `HEART_ES_TEST_INDEX`; optional `HEART_ES_API_KEY`, `HEART_ES_PIPELINE_TEST_INDEX` | Workspace/ES Harness |
| Docker Workspace | Existing `HEART_WORKSPACE_TEST_IMAGE` and an available Docker daemon | Container, mount, cancellation |
| MCP | The tests start real protocol-speaking service processes from the repository | HTTP MCP, stdio MCP |

`cargo test --all-targets` runs deterministic tests by default. Tests marked
`ignored` for real backends are not counted as passing; use the script above for
live acceptance.

---

## Compression and A2A

Compression creates the current Session summary without rewriting original Chats. It reads Chats after the covered range, uses their recorded input/output usage, and records the compression call usage for display. The active request is not replaced in the middle of a run; later requests use the new summary plus uncovered Chats.

A2A maps remote `context_id` to the application's `session_id`, converts Agent Events to JSON-RPC/SSE, and sends user replies back through `reply`. It is a protocol adapter, not another Runtime.

---

## Server and HTTP

The Axum binding exposes JSON-RPC methods such as `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`, and `tasks/resubscribe`. SSE is a transport for the existing Event stream.

```rust
let service = std::sync::Arc::new(halo::agent::server::a2a::Service::with_store(agent, store));
let router = halo::agent::server::axum::router(service);
let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
axum::serve(listener, router).await?;
```

The external Task Store is a projection. Agent/Workflow Runtime remains the source of execution truth and the HTTP layer does not duplicate State or waiting logic.

---

## Team and multi-Agent

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
