# Agent Components

## Component Contract

Each Component owns its own input, execution, output Event, and error contract. A
Component must not return an Event outside its declared contract. Such a return is
an implementation error in that Component.

Runtime coordinates Components. It is responsible for call ordering, context
passing, cancellation propagation, permission flow, and run-state progression. It
does not repair, infer, convert, or validate Component output contracts.

## Model Contract

### Model Namespaces

The Model layer is organized by capability:

```text
model
├── chat
├── audio
│   ├── speech
│   └── transcription
└── embedding
```

`audio` is a namespace only. It is not an executable Model. `audio::speech`
converts text to audio; `audio::transcription` converts audio to text.

### Invocation

A Model is called directly with `messages: Vec<Message>`, its capability-specific
`Options`, read-only Run `metadata`, and `Cancellation`. Do not introduce an
`Input`, `ModelInput`, `Item`, `Part`, or a type alias wrapping `Vec<Message>`.

`Model` is the common base trait. `Chat`, `Speech`, `Transcription`, and
`Embedding` extend it and each directly accepts its own Options type. Do not add
a Model `Kind` or an outer Options enum for runtime dispatch.

Each Model implementation validates and extracts the content it needs from
`messages`, then directly constructs and assigns its Provider SDK request fields.
Runtime does not repair, infer, or convert Model input.

`metadata` is available to the concrete Model implementation as well as its
optional Hooks. The framework does not interpret custom keys or automatically
copy them into a Provider request; a custom implementation decides how to use
them.

### Streaming Output

All Model `execute()` methods return an Event stream so the call shape stays the
same for streaming and non-streaming Providers. The selected capability Options
decide which Event sequence is valid:

- `stream = false`: a successful execution returns exactly one
  `Event::Complete` and no `Event::Delta`.
- `stream = true`: a successful execution may return zero or more
  `Event::Delta` values, followed by exactly one `Event::Complete`.
- `Event::Complete` is the complete result of that Model execution. A Delta is
  only the current increment and is never treated as the complete result.
- An execution error returns `Err(model::Error)`. No Event may follow the error.

### Options

Model input and invocation options are separate. Text, audio, files, and other
user-provided data belong in `messages`; invocation parameters belong to the
capability-specific `Options`.

- Chat options include parameters such as `max_tokens`, `temperature`, `top_p`,
  and tool settings.
- Speech options include parameters such as voice, output format, and speed.
- Transcription options include parameters such as language, prompt, and output
  format.
- Embedding options include parameters such as dimensions and encoding format.

### Message Roles

`role` describes the direction of a Model interaction. A Model preserves the
roles on its input messages. Every direct Model result is an assistant message:

```text
input to Model  -> original Message.role
output of Model -> Message.role = assistant
```

This rule applies to Chat, Speech, Transcription, and Embedding. A transcription
is therefore returned as an assistant message even when its audio input came from
a user. If later processing needs to use that transcript as Chat user input, the
input-processing layer explicitly creates a new user message. It does not change
the direct Transcription result.

An Assistant Message may carry text, audio, tool executes, or embeddings. An
Embedding keeps the Provider batch `index` and the returned `embedding` vector:

```rust
pub struct Embedding {
    pub index: u32,
    pub embedding: Vec<f32>,
}
```

`index` is the position of the corresponding text in the current batch. It is
not a database identifier.

Assistant fields are not globally mutually exclusive. A Chat result may contain
text together with one or more Tool Executes, and may contain text together with
audio when the Provider supports both. Speech, Transcription, and Embedding
usually populate only their primary result field. The concrete Model validates
whether a particular combination is supported; Schema and Runtime do not impose
cross-field exclusivity.

### User Message Content

`Message::User.content` is a `Vec<Content>`. `Content` is the content element
inside a User Message; it is not another input wrapper:

```rust
pub enum Content {
    Text { text: String },
    Image { url: String, detail: Option<ImageDetail> },
    Audio { data: String, format: String },
    File { data: Option<String>, id: Option<String>, filename: Option<String> },
}
```

A single User Message may contain both a File and a Text question. Each concrete
Model implementation directly assigns these values to its Provider SDK request
fields. There is no separate Provider Adapter or shared conversion layer.

### Usage

Model usage is optional because Providers do not expose the same measurements for
every capability:

```rust
pub struct Usage {
    pub input: Option<u64>,
    pub cached: Option<u64>,
    pub output: Option<u64>,
    pub reasoning: Option<u64>,
    pub total: Option<u64>,
    pub duration: Option<u64>,
}
```

`input`, `output`, and `total` are Provider-reported token totals. `cached` is
cached input tokens, and `reasoning` is reasoning output tokens. `duration` is a
Provider usage measured by duration, such as Transcription audio duration, and is
stored in milliseconds. Missing measurements remain `None`; the framework does
not manufacture zero values. `Usage.duration` is distinct from Trace call
duration. Audio-token breakdown fields and `cache_write` are not part of the
unified Usage contract.

## Component Hooks

Model and Tool expose optional `before` and `after` Hooks with default no-op
implementations. Hooks belong to the concrete Component and are available for a
custom Component implementation to override. The default Component path does not
perform Hook processing, and Runtime does not treat Hooks as mandatory stages.

- `before` may modify Component input and receives read-only Run `metadata` when
  the custom Component invokes it.
- A custom Component that changes its input is responsible for validating the
  changed input before execution.
- `after` may modify a successful Event and receives the same read-only `metadata`
  when the custom Component invokes it.
- Hooks must preserve their Component input and output contracts.
- Hooks do not form an Onion chain and do not implement Agent Middleware stages.

Business system-prompt transformation belongs to Agent Middleware such as
`on_system_prompt`. Model Hooks are limited to logic intrinsic to that Model.

## Cancellation

Each Run creates one root `Cancellation`. Graph and Workflow nodes derive child
tokens along the execution dependency direction. For `A -> C -> D`, `C` uses a
child of `A`, and `D` uses a child of `C`:

```text
A Cancellation
└── C Cancellation
    └── D Cancellation
```

- Cancelling `A` cancels `A`, `C`, and `D`.
- Cancelling `C` cancels `C` and `D`, but does not cancel `A` or another branch
  beside `C`.
- Cancelling the Run root cancels every active node and prevents unscheduled
  downstream nodes from starting.
- Runtime owns token creation, parent-child relationships, and scheduling.
- A running Model or Tool owns stopping its Provider request, external call, or
  local task when its `Cancellation` fires. Receiving a token alone does not stop
  that work automatically.
- Cancellation is not an ordinary execution failure and must be returned as the
  Component's cancelled error kind.

## Tool Contract

`Tool.execute()` receives the validated arguments, read-only Run `metadata`, and
`Cancellation`. It currently does not stream. On success, it returns exactly one
`Event::Complete` containing `Message::Tool`.

- `Message::Tool.execute_id` equals the `execute_id` passed to `Tool.execute()`.
- `usage` and `finish_reason` are `None`.
- A Tool does not return `Event::Delta` or another `Message` role.
- Tool failures return `Err(tool::Error)` rather than a synthetic Tool message.

## Tool Arguments

The Execution Layer that calls a Tool parses `Execute.arguments` and validates it
against `Definition.parameters`. `Tool.execute()` receives the validated JSON
value and read-only Run `metadata`.

- Invalid JSON returns `tool::Error` with `kind = InvalidInput`,
  `code = json_parse`, and `is_retry = false`.
- A parameter-schema mismatch returns `tool::Error` with `kind = InvalidInput`,
  `code = schema_validation`, and `is_retry = false`.
- Neither failure calls `Tool.execute()` or retries the same Tool invocation.
- The higher-level ReAct flow may send the error back to Model so it can generate
  a new complete `Execute.arguments`. Routing that retry is not a Tool concern.
