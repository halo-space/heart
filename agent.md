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

## Memory And Knowledge

`Memory` and `Knowledge` are separate Components. `Memory` manages short and
long conversation memory; `Knowledge` manages externally imported knowledge and
its retrieval.

```text
Memory
├── short
│   ├── chats
│   └── sessions
│       └── compress
└── long
    └── compress

Knowledge
```

Memory resources use `create`, `read`, `update`, and `delete`. Do not use an
ambiguous `add` operation. Compression is not a top-level Component. It is an
internal Memory capability exposed as `Memory.short.sessions.compress` and
`Memory.long.compress`.

### Short Memory

Short memory is the current Session's persisted conversation view. Its source of
truth is `chats` and `sessions`; it does not maintain a duplicate message cache.

The `sessions` record contains the Session's ownership and lifecycle fields,
including `tenant_id`, `app_id`, `agent_id`, `user_id`, `title`, `status`,
`summary`, `created_time`, and `updated_time`. The Session record also has a JSON
extension field for related state. The JSON extension stores the Chat interval
covered by `summary`; it does not introduce another summary table or a separate
memory progress object.

- `app_id` identifies the application that owns the Session.
- `agent_id` is the stable Agent identity supplied to the framework. Application
  mapping from an AppAgent record to this value is completed before constructing
  the schema record.
- A Chat records exactly one root target: `agent_id` or `team_id`. It also keeps
  the resolved Model snapshot in `models` and the optional telemetry identifier
  in `trace_id`.
- `trust` and `feedback` are independent integers limited to `-1 / 0 / 1`.
  `trust` controls context and memory reuse; `feedback` records the user's direct
  positive, neutral, or negative evaluation.

- `chats.create` persists the completed Chat after every completed turn. It does
  not trigger summary generation or long-memory processing.
- `sessions.summary` is a rolling summary of earlier Chats in the same Session.
  It is created or updated only before a future Model call when the selected
  context would exceed the configured budget.
- Recent raw Chats remain readable after summary generation. The current user
  input comes from the current Chat request, not from a short-memory read.
- A summary records the covered Chat interval with the unified form below. Both
  ends are inclusive.

```json
{
  "idx": {
    "from": 1,
    "to": 8
  }
}
```

Summary processing is explicit:

```text
read prior sessions.summary + selected older Chats
  -> Memory.short.sessions.compress
  -> sessions.update(summary, idx)
```

The internal compression step receives data selected by Memory. The step itself
does not read database tables or write database records. Memory performs the
required reads and writes around it.

#### Session Context Compression

Before every normal Chat Model call, the caller assembles the current Context
and estimates its input size. The estimate includes the fixed system/developer messages,
available Tool definitions, `sessions.summary`, the selected recent Chats, the
current Chat input, and any other data explicitly selected for this call.

The effective input budget is derived from the selected Model's limits:

```text
effective input budget
  = context_tokens - reserved output_tokens - safety margin
```

If the assembled Context exceeds that budget, only the current Session's older
conversation is compressed. This is Session Summary compression, not long-memory
extraction and not Knowledge retrieval. `Memory.short.sessions.compress` has its
own bounded input and output options; it does not recursively trigger Session
Summary compression.

The caller compresses the oldest contiguous Chat range first, keeps the newest
Chats as raw messages, and then rebuilds and rechecks the Context:

```text
sessions.summary covering 1..0
+ chat 1 + chat 2 + chat 3 + chat 4
+ current input chat 5
  -> Context exceeds budget
  -> Memory.short.sessions.compress(summary prompt, chat 1..2)
  -> sessions.update(summary, idx = { "from": 1, "to": 2 })
  -> sessions.summary covering 1..2
+ chat 3 + chat 4
+ current input chat 5
  -> Model
```

The example uses four completed Chats and a fifth current input. The compression
may cover a different oldest range; it does not have to compress the entire
Session. The summary update advances its `idx.to` only after compression and the
Session update both succeed. A failed, invalid, or cancelled compression does
not advance the summary range.

If all eligible historical Chats have been compressed and the Context is still
over budget, the caller returns a context-size error or applies an explicit
input-handling policy. It must not keep recursively compressing the same
summary, silently drop the current input, or use long-memory extraction as a
workaround. Long-memory processing remains an independent operation.

### Long Memory

Long memory is one aggregated record for one owner. It keeps only durable facts,
preferences, and confirmed decisions. It must not copy complete Chat input or
output, and it must not treat an assistant's final answer as a memory item.

`Memory.long` persists its result in the `profiles` table. A Profile is one
owner's aggregated long-term information, not a list of memory items. `id` is
the Profile's own primary key. Its owner is expressed by `tenant_id`, `agent_id`,
`user_id`, and `type`:

```text
type = user  -> agent_id and user_id are both actual IDs
type = agent -> agent_id is the owner ID; user_id = 0
```

The owner columns must be unique as `(tenant_id, type, agent_id, user_id)`. A
user therefore has one Profile for each Agent in the Tenant, and an Agent has
one shared Profile in the Tenant. The Profile contains one complete `content`
value rather than multiple memory items.

```text
User Profile:
tenant_id = 1
agent_id  = 10
user_id   = 100
type      = user

Agent Profile:
tenant_id = 1
agent_id  = 10
user_id   = 0
type      = agent
```

The minimal table structure is:

```sql
CREATE TABLE profiles (
  id BIGINT PRIMARY KEY,

  tenant_id BIGINT NOT NULL,
  agent_id BIGINT NOT NULL,
  user_id BIGINT NOT NULL DEFAULT 0,
  type VARCHAR(32) NOT NULL,

  content TEXT NOT NULL,
  extra JSON NOT NULL,

  version INT NOT NULL DEFAULT 0,
  created_time BIGINT NOT NULL DEFAULT 0,
  updated_time BIGINT NOT NULL DEFAULT 0,

  UNIQUE KEY uk_owner (tenant_id, type, agent_id, user_id)
);
```

- `id` is the Profile's stable primary key.
- `type` currently allows only `user` and `agent`.
- `content` is the complete aggregated Profile text.
- `extra` stores the per-Session processing ranges described below.
- `version` is incremented by each successful optimistic-lock update.
- `created_time` and `updated_time` are millisecond timestamps.
- `status`, `sort`, and `note` are not part of the Profile contract.
- Clearing an owner's long-term Profile deletes the row. A later consolidation
  creates a new row with a new Profile `id`.

- Long-memory processing selects one contiguous range of `chats.idx` in a
  Session. The range always uses `idx.from` and `idx.to`, never separate
  `start_idx`, `end_idx`, or `memory_chat_id` fields.
- A Chat is eligible only when `status = completed` and `trust >= 0`. Exclude
  `running`, `waiting`, `failed`, `partial`, `stopped`, and `trust < 0`.
- `trust` is an integer used to control contextual reuse and long-memory
  processing. Regenerating a Chat marks the earlier result untrusted. Explicit
  user feedback is stored separately as an integer and does not replace
  `trust`.
- `traces` are not a long-memory extraction source. Tool data can be too large
  and requires a separate decision before it may be included.
- Long-memory processing reads the owner's existing Profile content so the
  result can consolidate it rather than create another Profile.
- `Memory.long.compress` receives a Message list: a System Message carrying the
  compression prompt and a User Message carrying serialized selected Chats and
  the existing Profile content. It is not a single combined Message.
- It returns one complete replacement `content`. Memory creates the owner's
  Profile when absent, otherwise updates that single Profile.

```text
selected Chats + existing long memory
  -> Memory.long.compress
  -> profiles create or update
```

The long-memory processing range is independent from the Session summary range.
It is stored in the Profile's JSON `extra` field. `extra` is a map
whose key is `session_id`; it does not add fields to `sessions` and does not use
`summary_chat_id` or `memory_chat_id`:

```json
{
  "session_1": {
    "from": 1,
    "to": 10,
    "update_time": "2026-08-12T10:30:00+08:00"
  }
}
```

`from` and `to` are the inclusive Chat `idx` range that has been successfully
consolidated into this Profile for that Session. `update_time` is the
captured `sessions.updated_time` for that successfully processed range, not the
time at which the long-memory row itself was written.

All persisted Session, Chat, and Profile timestamps use `created_time` and
`updated_time`. They do not use parallel `created_at` or `updated_at` names.

When selecting Sessions for processing, the caller compares each Session's
current `sessions.updated_time` with the corresponding `extra[session_id]` value.
A missing entry, or a Session whose current update time is newer than the stored
`update_time`, is eligible. The caller then reads Chats after the stored `to` and
selects the next contiguous range. This comparison prevents unchanged Sessions
from being read repeatedly.

The selection captures both the Session's current `updated_time` and the fixed
`idx.to`. After the write succeeds, those captured values are stored together.
If another Chat updates the Session during processing, its newer update time is
not swallowed by the current progress update and the Session remains eligible
for a later pass.

Any progress update must advance only after compression output validation and
the long-memory write both succeed. On cancellation, invalid compression output,
or a write failure, it must not advance; a later execution retries the same
Chats.

#### Long-Memory Triggers

Long-memory processing is asynchronous and does not block the Chat response.
Its triggers are optional and composable; enabling one does not disable the
others:

- Count threshold: trigger when eligible, unprocessed Chats reach the configured
  count.
- Token threshold: trigger when eligible, unprocessed Chat content reaches the
  configured token estimate.
- Schedule: trigger from a configured periodic job. This supports a dream-like
  consolidation pass during idle or off-peak periods.
- Manual: an administrator, application workflow, or user action explicitly
  requests consolidation.

A trigger only schedules work. Before execution, the worker must reread the
current Profile, Session state, and eligible Chats, then fix the
exact `idx.from/to` range for that attempt. Multiple triggers for the same owner
may be coalesced; they must not execute overlapping writes concurrently.

#### Long-Memory Concurrency

The Profile contains a `version` used for optimistic locking. A write
must match the version read before `Memory.long.compress` and increment it on
success.

If the version no longer matches, another process has changed the owner's
memory. The current attempt must discard its generated replacement `content`,
reread the newest `content`, `extra`, and `version`, recalculate the unprocessed
Chat range, and run `Memory.long.compress` again. It must not apply the stale
content or retry only the database update.

Only the successful optimistic-lock write updates `content`, `extra`, and
`version` together. This keeps the aggregated memory and every Session progress
entry consistent.

#### No Memory Change

`Memory.long.compress` may determine that the selected Chats contain no durable
new fact, preference, or confirmed decision. In that case the existing `content`
remains unchanged, but the selected Session's `from`, `to`, and `update_time`
still advance through the same optimistic-lock write. This marks the range as
processed and prevents repeated compression of the same Chats.

Automatic extraction updates only the current user's `type = user` record. An
Agent's `type = agent` record is updated only by an explicit administrator or
application workflow. A user's conversation never automatically becomes
Agent-shared memory. Long memory may be reused across Sessions for its owner;
the extraction source interval remains Session-local.

### Memory Interface Boundary

`Memory` owns short-memory and Profile persistence as well as its internal
compression flows. `Knowledge` remains a separate Component:

```text
Memory
├── short
│   ├── chats
│   │   └── create / read / update / delete
│   └── sessions
│       └── create / read / update / delete / compress
└── long
    └── profiles
        └── create / read / update / delete / compress

Knowledge
```

Storage implementations are initialized once and injected into `Memory`.
`Memory` is then injected into the Agent. For each operation, Runtime supplies
the current scope and `Cancellation`; Runtime does not implement database access
or retain persisted Session, Chat, or Profile state.

Conceptual CRUD operations are:

```text
Memory.short.chats.create(chat)
Memory.short.chats.read(tenant_id, agent_id, user_id, session_id, idx)
Memory.short.chats.update(chat)
Memory.short.chats.delete(chat_id)

Memory.short.sessions.create(session)
Memory.short.sessions.read(session_id)
Memory.short.sessions.update(session)
Memory.short.sessions.delete(session_id)

Memory.long.profiles.create(profile)
Memory.long.profiles.read(tenant_id, agent_id, user_id, type)
Memory.long.profiles.update(profile, version)
Memory.long.profiles.delete(profile_id)
```

Read operations return persisted domain records. They do not return Model
`Message` values. The caller that assembles Model context converts selected Chat
and Session records into Messages.

#### Short Compression Flow

The conceptual operation is:

```text
Memory.short.sessions.compress(
  session_id,
  idx.from,
  idx.to,
  model,
  cancellation
)
```

Memory reads the current Session summary and the selected Chat range, invokes
the supplied Model, validates the compression result, and atomically updates
`sessions.summary` together with the Session JSON summary range. A failure or
cancellation leaves both values unchanged.

#### Long Compression Flow

The conceptual operation is:

```text
Memory.long.profiles.compress(
  tenant_id,
  agent_id,
  user_id,
  type,
  session_id,
  idx.from,
  idx.to,
  model,
  cancellation
)
```

Memory reads the current Profile and selected Session Chats, invokes the
supplied Model, validates the replacement Profile content, and creates or
optimistically updates the Profile. The successful write updates `content`,
`extra`, and `version` together. When no durable information is found, `content`
remains unchanged while the Session progress and `version` still advance.

The Model is supplied by the Agent or current execution. Memory does not bind a
Provider and does not implement Provider retry or streaming behavior. Memory is
responsible for selecting persisted data, running the compression flow,
validating its result, writing it, and enforcing optimistic locking.

### Memory Error Contract

Memory uses one minimal enum. It does not define a separate `ErrorKind`,
`CompressionError`, or retry flag:

```rust
pub enum Error {
    InvalidInput,
    NotFound,
    Storage,
    Model,
    InvalidOutput,
    Conflict,
    Timeout,
    Cancelled,
}
```

The variants have these boundaries:

- `InvalidInput`: an invalid owner type, `idx.from > idx.to`, or another invalid
  operation argument.
- `NotFound`: the required Session does not exist or a requested fixed Chat
  range is incomplete.
- `Storage`: a persistence read or write failed.
- `Model`: `Memory.short.sessions.compress` or `Memory.long.profiles.compress`
  failed while invoking the supplied Model.
- `InvalidOutput`: the Model call completed but its compression result did not
  satisfy the required output contract.
- `Conflict`: optimistic-lock retries were exhausted.
- `Timeout`: the Memory operation exceeded its configured deadline.
- `Cancelled`: the supplied `Cancellation` fired.

The enum is a classification boundary, not a diagnostic container. Memory
implementations record underlying storage or Model details through logging and
tracing. Retry decisions are made internally by the operation that owns the
retry: optimistic-lock retries rerun the compression flow, transient storage
failures may be retried by the storage implementation, and cancellation is
never retried. After the applicable retry limit is exhausted, Memory returns
the corresponding category, normally `Conflict` for optimistic-lock
exhaustion or `Storage` for a persistence failure.

A missing Profile is not an error during long compression; Memory creates it.
An invalid or non-contiguous fixed Chat range fails before the Model is called.
Profile version conflicts are handled inside Memory by rereading and rerunning
compression; `Conflict` is returned only after that retry limit is exhausted.
Cancellation is never retried and never advances summary or Profile progress.
