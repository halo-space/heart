# Heart User Guide

This guide shows how to use Heart from a Rust application to build Agents and Workflows. The examples follow the current public API.

[中文使用手册](USAGE.zh-CN.md)

## 1. Installation

```toml
[dependencies]
halo = { package = "halo-agent", version = "0.1.0" }
futures-util = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "0.8"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

### 1.1 Run the complete examples first

The repository `examples/` directory contains complete programs that compile and run; the snippets in this guide are not pseudocode. Start with the
offline examples:

```bash
cargo check --offline --examples
cargo run --offline --example agent_middleware
cargo run --offline --example node_macro
cargo run --offline --example workflow_events
```

In your own Cargo project, add the dependencies above and copy any example to `src/main.rs`. Every row below is a runnable entry point:

| Capability | Complete example | Command |
| --- | --- | --- |
| Chat Agent and Memory | [`examples/agent_memory.rs`](../examples/agent_memory.rs) | `cargo run --example agent_memory` |
| ReAct, Tool and Permission | [`examples/agent_permission.rs`](../examples/agent_permission.rs) | `cargo run --example agent_permission` |
| Middleware | [`examples/agent_middleware.rs`](../examples/agent_middleware.rs) | `cargo run --example agent_middleware` |
| Workflow event stream | [`examples/workflow_events.rs`](../examples/workflow_events.rs) | `cargo run --example workflow_events` |
| `#[node]` and typed nodes | [`examples/node_macro.rs`](../examples/node_macro.rs) | `cargo run --example node_macro` |
| Planner and Evaluator | [`examples/planning_evaluation.rs`](../examples/planning_evaluation.rs) | `cargo run --example planning_evaluation` |
| Prompt and compression | [`examples/compress.rs`](../examples/compress.rs) | `HALO_API_KEY=... cargo run --example compress` |
| Session/Memory CRUD | [`examples/short_memory.rs`](../examples/short_memory.rs) | `cargo run --example short_memory` |
| Document Loader/Parser/Transformer | [`examples/document_parse.rs`](../examples/document_parse.rs) | `cargo run --example document_parse -- ./input.csv` |
| OpenAI-compatible Chat | [`examples/provider_chat.rs`](../examples/provider_chat.rs) | `OPENAI_API_KEY=... cargo run --example provider_chat` |
| DeepSeek Chat Completions | [`examples/provider_deepseek.rs`](../examples/provider_deepseek.rs) | `DEEPSEEK_API_KEY=... cargo run --example provider_deepseek` |
| VLLM Chat Completions | [`examples/provider_vllm.rs`](../examples/provider_vllm.rs) | `HEART_VLLM_BASE_URL=... HEART_VLLM_API_KEY=... HEART_VLLM_MODEL=... cargo run --example provider_vllm` |

Network examples contact external services only when their command is explicitly run. Credentials are read from environment variables and never
written to source. `cargo check --examples` validates all example code without making model or MCP requests.

The package and Rust crate are `halo`, and Rust 1.99.1 or newer is required.

```text
components/   Public contracts: Agent, Model, Message, Event, Tool, Memory, Checkpoint
core/         Implementations: providers, Chat/ReAct, Workspace, Sandbox and storage backends
runtime/      Agent and Workflow state, graphs and execution control
harness/      Higher-level long-running Agent assembly
```

Capabilities are opt-in. Without a model there is no runnable Agent, without a Sandbox there is no shell capability, and without a Checkpoint
backend there is no cross-process recovery.

## 2. Common input, messages and events

Chat and ReAct use the same `agent::Input`:

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
        data: json!({"value": "How is the weather in Beijing today?"}),
        mime_type: None,
    }],
};
```

`metadata` carries application identity and context. `payload` is ordered user content. Audio, images and files use the same
`type/data/mime_type` shape; keep binary data outside the input and pass stable references when appropriate.

Model context uses `Message`/`Messages`. Provider reasoning fields are normalized to a `think` content Part and ordinary output to `text`.
The shared event contract is:

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

Events are observations, not Runtime State. Workflow downstream nodes read their input from node State.

## 3. Chat Agent

Chat Agent is the ordinary conversation loop. It can use mounted Tool, MCP and Skill capabilities. Use ReAct when a task needs iterations,
planning or replanning.

The following is the smallest Agent entry-point fragment; the application must initialize `chat` and `input` as described in section 6.
It is not a model implementation and does not fabricate a response. For a complete real-model program, see
[10.1 Real VLLM Chat Agent](#101-real-vllm-chat-agent).

```rust
use futures_util::StreamExt;
use halo::agent::{Cancellation};
use halo::agent;

let agent = agent::chat()
    .model(chat)
    .system_prompt("You are a reliable assistant")
    .build()?;

let mut events = agent.run(input, true, Cancellation::new()).await?;
while let Some(event) = events.next().await {
    println!("{:?}", event?);
}
```

`stream = true` delivers Delta and Complete; `false` delivers Complete only, with the same return type. Each model or tool invocation may
produce its own Complete event. Continue consuming the stream after cancellation to receive the cancellation Complete and accumulated Usage.

Mount capabilities while building the Agent:

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

MCP servers should be connected and discovered before mounting. Permission is checked before an actual Tool/MCP call. Tool
`before/call/after/eval` is the Tool lifecycle and is separate from Agent middleware.

## 4. ReAct Agent

ReAct uses the same input and event stream while adding iterations, tool batches and an optional Agent task graph:

```rust
let agent = agent::react()
    .model(chat)
    .system_prompt("Plan the task and select suitable mounted capabilities")
    .tools(vec![search_tool, calculator_tool])
    .mcps(servers)
    .skills(skills)?
    .with_max_iters(20)
    .with_max_concurrency(100)
    .memory(memory_config)
    .checkpoint(checkpoint)
    .build()?;
```

The Agent chooses whether to call a tool, which tool to call and its arguments. Mounted capabilities execute Tool, MCP and Skill operations.
Tool errors are fed back to the current model context instead of being disguised as successful business output. An Agent evaluator judges the
final result and does not replace Tool error handling.

### Waiting and feedback

Approval, human input and external callbacks put an Agent into `waiting`. Submit feedback through the same Agent object's `reply(Input, Cancellation)`:

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

Feedback is consumed FIFO. If the original process has ended, rebuild the Agent from the complete Checkpoint and submit the same reply; successful
model calls, tools and sibling nodes are not replayed.

## 5. Workflow

Workflow is explicit graph orchestration and does not use the Agent Planner. A node can be an ordinary async function, an Agent, or a compiled child
Workflow. Runtime owns graph scheduling, state, joins, fan-out, waiting and Checkpoints.

### 5.1 Ordinary function nodes

Ordinary functions can be passed directly to `WorkflowBuilder::add_node`; no wrapper class is required. Ordinary node I/O should have object-root
schemas, with `Messages` as the message-oriented exception:

```rust
use halo::agent::{Cancellation, Error};
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

Runtime builds each node's business input from upstream State according to `attrs.input`; nodes may use different input types.

### 5.2 Build and run

```rust
use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages};
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

The application owns the `i64` ID generator, for example a Snowflake generator. Heart does not initialize an ID service. Consume and drop the run
stream before reading Runtime State.

### 5.3 Relations, Join and fan-out

```rust
builder.add_conditional_edges("router", route, ["search", "fallback"])?;
builder.add_join(["search", "fallback"], "merge")?;
builder.add_fan_out_edges("planner", "worker")?;
```

Ordinary edges activate on any successful source. Join waits for every declared source. Conditional relations can only select declared targets.
Fan-out workers may run concurrently; recovery retries only the failed subset.

### 5.4 Embedded Agent and child Workflow

```rust
builder.add_agent_node("researcher", researcher_agent)?;
builder.add_graph_node("child", &compiled_child_workflow)?;
builder.add_edge(START, "researcher")?;
builder.add_edge("researcher", "child")?;
builder.add_edge("child", END)?;
```

The Agent keeps its own Agent State and a child Workflow keeps its graph boundary. The parent Workflow only schedules them as nodes.

### 5.5 Feedback and recovery

Workflow feedback uses `Runtime::reply(Messages, Cancellation)`. A live run continues on its original event stream. An ended process is rebuilt from
the complete snapshot:

```rust
let snapshot = checkpoint.load(1001, &Cancellation::new()).await?;
let runtime = Runtime::from_snapshot(1001, workflow, snapshot)?;
runtime.reply(feedback, Cancellation::new()).await?;
```

Recovery loads the complete Runtime State: completed nodes, failure marks, wait queues, graph version and business data. It does not replay models,
tools or nodes from the beginning.

## 6. Model providers

Provider initialization is separate from per-request options. OpenAI, DeepSeek and VLLM currently use the Chat Completions contract; Responses API
is not implemented.

```rust
let provider = halo::agent::core::provider::vllm::Vllm::builder()
    .base_url("http://127.0.0.1:8000/v1")
    .api_key(std::env::var("HEART_VLLM_API_KEY")?)
    .build()?;
let chat = provider.chat(
    halo::agent::model::chat::Config::new("Qwen3.5-122B-A10B").options(serde_json::Map::from_iter([(
        "chat_template_kwargs".into(),
        serde_json::json!({"enable_thinking": true}),
    )])),
)?;
```

VLLM `delta.reasoning` becomes the shared `think` Part and ordinary output becomes `text`. Provider-specific options stay in
`model::chat::Config.options` or `chat::Request.options`.

DeepSeek uses its own Provider type, and its model name is configured in the DeepSeek Chat binding:

```rust
let provider = halo::agent::core::provider::deepseek::DeepSeek::builder()
    .base_url("https://api.deepseek.com")
    .api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .build()?;
let chat = provider.chat(halo::agent::model::chat::Config::new("deepseek-flash"))?;
```

The Qwen model in the VLLM example belongs to the VLLM server and is not mixed with the DeepSeek Provider.

Zhipu Embedding-3 is a separate Embed Provider; it does not depend on a Chat Provider. The default request URL is Zhipu's V4 API. Omit `dimensions` to use the service default (2048), or set a value from 256 through 2048:

```rust
use halo::agent::{
    model::embed::{self, Embed as _},
    provider::zhipu::Zhipu,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = Zhipu::builder()
        .api_key(std::env::var("HEART_EMBED_API_KEY")?)
        .build()?;
    let embed = provider.embed(embed::Config {
        model_name: "embedding-3".into(),
        options: Default::default(),
    })?;
    let message = embed
        .embed(embed::Request {
            input: "Text to embed for semantic search".into(),
            dimensions: Some(1024),
            options: Default::default(),
        })
        .await?;
    let vector = message.values.get("vector").expect("embedding vector");
    println!("dimension={}, usage={:?}", vector.as_array().unwrap().len(), message.usage);
    Ok(())
}
```

Set `HEART_EMBED_API_KEY` in the local `.env`. For a compatible gateway or a different model name, optionally set `HEART_EMBED_BASE_URL` and `HEART_EMBED_MODEL`. Never commit credentials.

The OpenAI Embeddings API is also available through the existing `core::provider::openai::OpenAI`; no separate Provider type is needed. It defaults to `https://api.openai.com/v1` and can use `text-embedding-3-small`; pass a requested vector size via `embed::Request.dimensions`. Setting `OPENAI_API_KEY` enables the live OpenAI vector acceptance in `scripts/test-real.sh`.

## 7. Workspace, Skill and MCP

Workspace selects local-file or server-database behavior. Local Workspace provides basic file operations; shell execution requires an explicit Sandbox.
Skills are loaded with `.skills(loader)`. A database-backed Loader can override `version` with a revision or content fingerprint;
one Workspace instance reuses its discovered Skill snapshot and refreshes it only when that version changes. MCP servers must be connected and discovered before they are mounted.

```rust
let workspace = halo::agent::core::workspace::Local::new("./workspace")?
    .sandbox(halo::agent::sandbox::Local::default())
    .tools(custom_tool)
    .mcps(servers)
    .skills(skills)?;
```

Keep provider keys and MCP endpoints in a local `.env`; never commit secrets.

## 8. Cancellation, errors and tests

`Cancellation` scopes one independent run. `Complete` is one observable model, tool or node result, not the entire Runtime status. Checkpoint is the
recovery source; the final Chat.state is the audit snapshot. Run the normal checks with:

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

## 9. Current scope

Full PDF/Office/OCR Parser extensions, Python function integration, TTS and the complete multimodal contract are future-version work.

## 10. Complete runnable recipes

The snippets above explain the contracts. This section gives complete programs that can be copied to `src/main.rs` in a fresh project:

```bash
cargo new heart-guide-demo
cd heart-guide-demo
```

The basic recipes may use deterministic local Chat doubles to test contracts. They are not model capabilities. VLLM and MCP recipes explicitly
require external services and environment variables.

### 10.1 Real VLLM Chat Agent

This program calls a real OpenAI-compatible VLLM Chat Completions service; it does not fabricate a model response. Set the service address,
credential and model first:

```bash
export HEART_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export HEART_VLLM_API_KEY=replace-me
export HEART_VLLM_MODEL=Qwen3.5-122B-A10B
```

It shows Provider initialization, model binding, the shared `Input`, Agent-to-`Messages` organization, and the fixed `Delta`/`Complete` stream.

```rust
use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Input, Payload};
use halo::agent::{event::Event, model::chat, provider::vllm::{Config, Vllm}};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), halo::agent::Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".into()),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("HEART_VLLM_API_KEY is required"),
        ..Config::default()
    })?;
    let chat = provider.chat(chat::Config {
        model_name: std::env::var("HEART_VLLM_MODEL").expect("HEART_VLLM_MODEL is required"),
        options: serde_json::Map::from_iter([("chat_template_kwargs".into(), json!({"enable_thinking": true}))]),
    })?;
    let agent = agent::chat().model(chat).system_prompt("You are concise").build()?;
    let input = Input { metadata: serde_json::Map::from_iter([("chat_id".into(), json!(1001)), ("session_id".into(), json!(2001)), ("agent_id".into(), json!(3001))]), payload: vec![Payload { r#type: "text".into(), data: json!({"value":"Can I go outside in Beijing today?"}), mime_type: None }] };
    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await { println!("{:?}", event?); }
    Ok(())
}
```

### 10.2 ReAct with a Tool and Permission

This offline recipe uses a deterministic model double to simulate one Tool call followed by an answer after the Tool message; it is not model
reasoning. Replace the model double with the Provider Chat initialized in 10.1 for a real model. The public builder uses `.tools(...)`; the
internal dispatch implementation is not an application API.

```rust
use std::sync::{LazyLock, atomic::{AtomicUsize, Ordering}};
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error};
use halo::agent::{
    event::Event,
    message::{Message, Role},
    model::{chat::{Chat, Request}, content::part::Part},
    permission::{Behavior, Decision, Request as PermissionRequest},
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::json;
struct Lookup;
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition { static D: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition { name: "lookup".into(), description: "return weather".into(), parameters: serde_json::from_value(json!({"type":"object","properties":{},"additionalProperties":false})).unwrap() }); &D }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> { Ok(Message::function(json!({"temperature":22,"condition":"sunny"}))) }
}
struct DemoModel(AtomicUsize);
impl Chat for DemoModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 { message.content.push(Part { r#type: "tool_call".into(), data: json!({"call_id":"call_1","name":"lookup","arguments":{}}).as_object().unwrap().clone() }); }
        else { message.values.insert("answer".into(), json!(if request.messages.as_slice().iter().any(|m| m.role == Role::Tool) { "22 C and sunny" } else { "no tool result" })); }
        Ok(stream::iter(vec![Ok(Event::Complete { message, usage: None, finish_reason: Some("stop".into()) })]))
    }
}
#[tokio::main]
async fn main() -> Result<(), Error> {
    let permission = halo::agent::core::permission::Permission::new(|r: PermissionRequest, _: Cancellation| async move { Ok(if r.action == "call" && r.resource == "lookup" { Decision { behavior: Behavior::Allow, reason: None } } else { Decision { behavior: Behavior::Deny, reason: Some("not allowed".into()) } }) });
    let agent = agent::react().model(DemoModel(AtomicUsize::new(0))).tools(Lookup).permission(permission).with_max_iters(20).build()?;
    let mut events = agent.run(agent::Input::default(), true, Cancellation::new()).await?;
    while let Some(event) = events.next().await { println!("{:?}", event?); }
    Ok(())
}
```

Tool errors are returned as errors in the current model context. They are not successful business values and do not automatically turn the whole
ReAct Agent into a failed run.

### 10.3 Workflow ordinary node and state

```rust
use std::sync::{Arc, atomic::{AtomicI64, Ordering}};
use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages, event::Event, workflow::{GraphError, Runtime, WorkflowBuilder, graph::{START, END}}};
use schemars::JsonSchema; use serde::Serialize;
#[derive(Serialize, JsonSchema)] struct Output { value: i64 }
fn ge(e: GraphError) -> Error { Error { code: e.code, message: e.message, details: e.details } }
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let mut b = WorkflowBuilder::new("answer");
    b.add_node("answer", |_: Messages, _: Cancellation| async { Ok::<_, Error>(Output { value: 42 }) }).map_err(ge)?;
    b.add_edge(START, "answer").map_err(ge)?; b.add_edge("answer", END).map_err(ge)?;
    let runtime = Runtime::new(1001, b.compile().map_err(ge)?)?;
    let ids = Arc::new(AtomicI64::new(1));
    let mut events = runtime.run(Messages::default(), Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)), Cancellation::new()).await?;
    while let Some(event) = events.next().await { println!("{:?}", event?); }
    drop(events); println!("state = {:?}", runtime.state()); Ok(())
}
```

### 10.4 Workflow relations and child graphs

This complete compile-time example declares conditional routing, Join and Fan-out. It prints the immutable definition version without requiring a
model or a database:

```rust
use halo::agent::{Cancellation, Error, Messages, workflow::{GraphError, WorkflowBuilder, graph::{START, END}}};
fn ge(e: GraphError) -> Error { Error { code: e.code, message: e.message, details: e.details } }
#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut b = WorkflowBuilder::new("relations");
    for id in ["router", "left", "right", "merge", "worker"] { b.add_node(id, |input: Messages, _: Cancellation| async move { Ok::<_, Error>(input) }).map_err(ge)?; }
    b.add_edge(START, "router").map_err(ge)?;
    b.add_conditional_edges("router", |_| Ok(vec!["left".into(), "right".into()]), ["left", "right"]).map_err(ge)?;
    b.add_join(["left", "right"], "merge").map_err(ge)?;
    b.add_fan_out_edges("merge", "worker").map_err(ge)?;
    b.add_edge("worker", END).map_err(ge)?;
    let graph = b.compile().map_err(ge)?; println!("{}@{}", graph.definition().definition_id, graph.definition().version); Ok(())
}
```

Ordinary edges activate on any successful source, Join waits for every declared source, conditional targets are statically declared, and Fan-out
workers can run concurrently. Use `add_graph_node("child", &child)` for an already compiled child Workflow; a child graph is not erased into a
plain function node.

### 10.5 Checkpoint recovery

```rust
use std::sync::{Arc, atomic::{AtomicI64, Ordering}}; use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages, core, workflow::{GraphError, Runtime, WorkflowBuilder, graph::{START, END}}};
use schemars::JsonSchema; use serde::Serialize;
#[derive(Serialize, JsonSchema)] struct Output { value: String }
fn ge(e: GraphError) -> Error { Error { code:e.code, message:e.message, details:e.details } }
#[tokio::main] async fn main() -> Result<(), Error> {
    let mut b=WorkflowBuilder::new("checkpoint"); b.add_node("done", |_:Messages,_:Cancellation| async { Ok::<_,Error>(Output{value:"saved".into()}) }).map_err(ge)?; b.add_edge(START,"done").map_err(ge)?; b.add_edge("done",END).map_err(ge)?;
    let workflow=b.compile().map_err(ge)?; let checkpoint=core::checkpoint::InMemory::new(); let runtime=Runtime::new(7001,workflow.clone())?.with_checkpoint(checkpoint);
    let ids=Arc::new(AtomicI64::new(1)); let mut events=runtime.run(Messages::default(),Arc::new(move||ids.fetch_add(1,Ordering::SeqCst)),Cancellation::new()).await?; while let Some(_)=events.next().await {} drop(events);
    let snapshot=runtime.snapshot()?; let mut restored=Runtime::from_snapshot(7001,workflow,snapshot)?; println!("restored={:?}",restored.state().status); Ok(())
}
```

The snapshot is the full recovery scene: node State, errors, waits, graph version and business data. Chat.state is an audit snapshot, not recovery truth.

### 10.6 VLLM Chat Completions

Create an ignored `.env` file and load it before running:

```dotenv
HEART_VLLM_BASE_URL=http://127.0.0.1:8000/v1
HEART_VLLM_API_KEY=replace-me
HEART_VLLM_MODEL=Qwen3.5-122B-A10B
```

```bash
set -a; source .env; set +a; cargo run
```

Save this complete provider program as `src/main.rs`:

```rust
use futures_util::StreamExt;
use halo::agent::{event::Event, message::{Message, Messages, Role}, model::{chat, chat::Chat, content::part::Part}, core::provider::vllm::Vllm};
use serde_json::json;
#[tokio::main]
async fn main() -> Result<(), halo::agent::model::Error> {
    let provider = Vllm::builder().base_url(std::env::var("HEART_VLLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".into())).api_key(std::env::var("HEART_VLLM_API_KEY").map_err(|_| halo::agent::model::Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required"))?).build()?;
    let chat = provider.chat(chat::Config::new(std::env::var("HEART_VLLM_MODEL").map_err(|_| halo::agent::model::Error::new("INVALID_ARGUMENTS", "HEART_VLLM_MODEL is required"))?).options(serde_json::Map::from_iter([("chat_template_kwargs".into(), json!({"enable_thinking": true}))])))?;
    let mut user = Message::new(Role::User); user.content.push(Part { r#type: "text".into(), data: json!({"value":"Introduce Pinglu Canal"}).as_object().unwrap().clone() });
    let mut events = chat.stream(chat::Request { messages: Messages::new([user]), stream: true, max_tokens: Some(512), ..Default::default() }).await?;
    while let Some(event) = events.next().await { match event? { Event::Delta { message } => for p in message.content { print!("{}", p.data["value"].as_str().unwrap_or_default()); }, Event::Complete { usage, finish_reason, .. } => println!("\nusage={usage:?}, finish_reason={finish_reason:?}") } }
    Ok(())
}
```

The current adapter targets `/v1/chat/completions`, not Responses API. VLLM reasoning deltas become `think` Parts and ordinary deltas become
`text` Parts.

### 10.7 Memory and explicit recall

Use `core::memory::short::InMemory` for short history and `core::memory::InMemory` for long-term
`summary/fact/preference`. Pass the resulting `memory::Config` through `.memory(config)`. Without that call, the Agent has no automatic history
or long-memory capability. This complete offline program writes a Session and one long-term Preference before running:

```rust
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::agent::{
    self, Agent, Input,
    core,
    event::Event,
    memory::{self, Memory},
    memory::short::{self, session},
    message::Role,
    model::{chat::{Chat, Request}, content::part::Part},
};
use serde_json::{Map, json};
use memory::short::session::Store as _;
struct Model;
impl Chat for Model { type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>; async fn stream(&self, _: Request) -> Result<Self::Stream, Error> { let mut m=Message::new(Role::Assistant); m.content.push(Part{r#type:"text".into(),data:json!({"value":"Sunny; camping is fine."}).as_object().unwrap().clone()}); Ok(stream::iter(vec![Ok(Event::Complete{message:m,usage:None,finish_reason:Some("stop".into())})])) } }
#[tokio::main] async fn main() -> Result<(), Error> {
    let cancel=Cancellation::new();
    let short=core::memory::short::InMemory::new();
    let long=core::memory::InMemory::new();
    let config=memory::Config::new(short,long).with_scopes(["user","agent"]);
    config.session().create(session::Session{id:1,tenant_id:1,user_id:2,r#type:short::Type::Agent,title:Some("weather".into()),summary:None,usage:None,status:session::Status::Active,metadata:Map::new(),created_time:0,updated_time:0},&cancel).await?;
    config.long().write(memory::Request{id:3,r#type:memory::Type::Preference,user_id:2,agent_id:None,session_id:None,from:1,to:2,content:"User likes camping".into(),metadata:None,vector:None},&cancel).await?;
    let agent=agent::chat().model(Model).memory(config).build()?; let mut events=agent.run(Input::default(),false,Cancellation::new()).await?; while let Some(e)=events.next().await { println!("{:?}",e?); } Ok(())
}
```

Compression is separate and opt-in. A configured compressor may replace the Chat range covered by the current Session summary when the application
explicitly checks the threshold; the current request is never switched to a newly written summary midway through execution.

### 10.8 Local Workspace and Skill

```rust
use halo::agent::{Cancellation, Error, core, skill::Skill as _, workspace::Workspace};
#[tokio::main]
async fn main() -> Result<(), Error> {
    let root=std::env::temp_dir().join(format!("heart-guide-{}",std::process::id())); std::fs::create_dir_all(&root).map_err(|e|Error::new("IO_ERROR",e.to_string()))?;
    let workspace=core::workspace::Local::new(&root)?; let cancel=Cancellation::new();
    workspace.write("research/SKILL.md", b"---\nname: research\ndescription: List sources\n---\n# Instructions\nAlways cite sources.\n".to_vec(), &cancel).await?;
    println!("files={:?}",workspace.list(".",true,&cancel).await?); let skill=core::skill::Skill::load(&workspace,"research/SKILL.md",&cancel).await?; println!("instructions={:?}",skill.read(&cancel).await?);
    let _=std::fs::remove_dir_all(root); Ok(())
}
```

Mount a concrete Skill with `.skills([skill])?`, or implement `skill::Loader` and use `.skills(loader)` for directory/database discovery. Skill
text is instructions; it does not execute scripts or grant permissions. Referenced text resources are read only when the model explicitly calls the shared `skill` reader with a relative `resource` path; references are never followed automatically.

### 10.9 MCP initialization

MCP connection and tool discovery happen before Agent construction. A failed endpoint returns an error from `Server::http` and is not silently
treated as an optional capability:

```rust
use halo::agent::{Cancellation, Error};
use halo::agent::mcp::{Definition, Server as _};
use serde_json::Map;
#[tokio::main] async fn main() -> Result<(), Error> {
    let endpoint=std::env::var("HEART_MCP_SSE_URL").map_err(|_|Error::new("INVALID_ARGUMENTS","HEART_MCP_SSE_URL is required"))?;
    let server=halo::agent::core::mcp::Server::http(Definition{name:"weather".into(),description:Some("weather MCP".into()),metadata:Map::new()},&endpoint,Map::new(),&Cancellation::new()).await?;
    println!("tools={:?}",server.tools().iter().map(|t|&t.name).collect::<Vec<_>>()); Ok(())
}
```

The application then passes the connected `server` to `agent::chat().model(chat).mcps([server])`. Do not commit the endpoint or credentials.

For local protocol validation, the project includes a separate stdio MCP service process. Its `current_time` tool reads the actual system clock; the acceptance test uses Heart's MCP client to discover and call it instead of substituting a canned business response:

```bash
cargo test --offline --test mcp_stdio_acceptance -- --nocapture
cargo test --offline --test mcp_http_acceptance -- --nocapture
```

The HTTP acceptance test starts a local Streamable HTTP MCP service, exercises initialization, tool discovery and a real tool call through Heart's HTTP client, and checks the returned timestamp against the host clock.

MCP acceptance uses the project's own standalone stdio service process and local Streamable HTTP service; it does not depend on a remote MCP endpoint. Keep provider credentials in the local `.env`, never in Git.

### 10.10 Validation order

Run offline checks before external services:

```bash
cargo fmt --all --check
cargo check --offline --lib --all-targets
cargo check --offline --examples
cargo test --offline --lib
```

Live model acceptance must call the actual provider; fixed model responses are not a substitute. After configuring the local `.env`, run the live Agent tool-call and parent/child delegation tests:

```bash
./scripts/test-real.sh
```

The script reads provider credentials from the untracked local `.env` without printing secrets. It runs real Provider, Planner, Evaluator, Compressor, long-memory extraction/merge, Chat/ReAct Agent tool calls, parent/child Agent, Agent human-feedback recovery, streaming/non-streaming Agent, Chat persistence closeout, Workflow with an embedded Agent, Skill, and local MCP stdio/HTTP acceptance tests. Configure `HEART_EMBED_API_KEY` to run real Semantic Chunker acceptance with Zhipu Embedding-3; `HEART_EMBED_BASE_URL` and `HEART_EMBED_MODEL` are optional and default to `https://open.bigmodel.cn/api/paas/v4` and `embedding-3`. Without the API key that case is explicitly skipped (no synthetic vectors are substituted). With `HEART_ES_URL` and an isolated test index configured, it also runs real Elasticsearch tests; with `HEART_WORKSPACE_TEST_IMAGE`, it runs real Docker Workspace tests. It does not connect to any remote MCP endpoint. Individual commands are listed below for selective runs.

The live acceptance checks use real providers and runtime data for Provider calls, Planner, Evaluator, Compressor, long-memory extraction/merge, Chat/ReAct Agent tool calls, parent/child delegation, MCP, Skill, user-feedback recovery and Workflow-embedded Agents. A provider/service failure or a model that does not perform the requested call is a test failure; never substitute a canned model response. The DeepSeek acceptance tests require `DEEPSEEK_API_KEY` in `.env`; `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL` are optional and default to the official API and `deepseek-flash`. Keep `.env`, test fixtures and design material out of the source commit.
`component_real_acceptance` validates a real complete Planner DAG, Reasoner Provider Complete, final-result Evaluator judgment, a Chat Agent completion checked by a second real Evaluator call, typed long-memory extraction and merge, and a Compressor summary from the actual model Complete.
`workspace_skill_real_acceptance` uses an actual local `SKILL.md` and checks Workspace discovery, on-demand Skill reading by the live model, and an answer grounded in that content. `workflow_real_acceptance` runs a real DeepSeek-backed Agent as a Workflow node and checks persisted execution status and Usage.
`agent_feedback_real_acceptance` has a real DeepSeek model select a feedback-waiting Tool, then verifies that a user reply correlated by `reply_id` resumes the same Agent and produces a final answer.
`agent_stream_real_acceptance` uses real DeepSeek calls to verify that `stream=true` emits Delta/Complete and `stream=false` emits Complete only.
The live closeout test in `agent_chat_closeout_real_acceptance` also verifies that the actual Provider Message and Usage are persisted in Chat state.
`semantic_chunking_real_acceptance` exercises Semantic Chunker with real Zhipu Embedding-3 vectors when `HEART_EMBED_API_KEY` is configured; optionally override the API base URL or model with `HEART_EMBED_BASE_URL` and `HEART_EMBED_MODEL`.
The default `cargo test` suite contains offline unit and state-machine tests for direct state transitions, validation, and event-protocol rules; it does not synthesize successful Chat or Embed responses. These checks are not live-model end-to-end acceptance and do not replace the real acceptance script.
Docker Workspace container, bind-mount and cancellation tests do not pull images. Point them at an image already present locally, then run explicitly:

```bash
export HEART_WORKSPACE_TEST_IMAGE=docker.elastic.co/elasticsearch/elasticsearch:9.5.4
cargo test --offline --test workspace_docker docker_workspace_real_container_smoke -- --ignored --nocapture
cargo test --offline --test workspace_docker docker_workspace_real_bind_mount_smoke -- --ignored --nocapture
cargo test --offline --test workspace_docker docker_workspace_real_cancellation_smoke -- --ignored --nocapture
```
