# Heart User Guide

This guide shows how to use Heart from a Rust application to build Agents and Workflows. The examples follow the current public API.

[中文使用手册](USAGE.zh-CN.md)

## 1. Installation

```toml
[dependencies]
halo-agents = { git = "https://github.com/halo-space/heart.git" }
futures-util = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "0.8"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

The package is `halo-agents`, the Rust crate is `halo_agents`, and Rust 1.99.1 or newer is required.

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
use halo_agents::agent::{Input, Payload};
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

```rust
use futures_util::StreamExt;
use halo_agents::{agent, Cancellation};

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
use halo_agents::{Cancellation, Error};
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
use halo_agents::{Cancellation, Error, Messages};
use halo_agents::runtime::workflow::{graph::{END, START, GraphError}, Runtime, WorkflowBuilder};
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
let provider = halo_agents::provider::vllm::Vllm::new(
    halo_agents::provider::vllm::Config {
        provider_id: 1,
        provider_name: "vllm".into(),
        base_url: "http://127.0.0.1:8000/v1".into(),
        api_key: std::env::var("HEART_VLLM_API_KEY")?,
        ..Default::default()
    },
)?;
let chat = provider.chat(halo_agents::model::chat::Config {
    model_name: "Qwen3.5-122B-A10B".into(),
    options: serde_json::Map::from_iter([(
        "chat_template_kwargs".into(),
        serde_json::json!({"enable_thinking": true}),
    )]),
})?;
```

VLLM `delta.reasoning` becomes the shared `think` Part and ordinary output becomes `text`. Provider-specific options stay in
`model::chat::Config.options` or `chat::Request.options`.

## 7. Workspace, Skill and MCP

Workspace selects local-file or server-database behavior. Local Workspace provides basic file operations; shell execution requires an explicit Sandbox.
Skills are loaded with `.skills(loader)`. MCP servers must be connected and discovered before they are mounted.

```rust
let workspace = halo_agents::core::workspace::Local::new("./workspace")?
    .sandbox(halo_agents::sandbox::Local::default())
    .tools(custom_tool)
    .mcps(servers)
    .skills(skills)?;
```

Keep provider keys and MCP endpoints in a local `.env`; never commit secrets.

## 8. Cancellation, errors and tests

`Cancellation` scopes one independent run. `Complete` is one observable model, tool or node result, not the entire Runtime status. Checkpoint is the
recovery source; Trace is an audit projection. Run the normal checks with:

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

## 9. Current scope

Full PDF/Office/OCR Parser extensions, Python function integration, TTS and the complete multimodal contract are future-version work.
