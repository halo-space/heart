# Agent Components

## Current Scope

The current implementation phase covers reusable Schema and Component contracts
only. Each Component defines and implements its own capability, input, output,
validation, cancellation, and error behavior.

This phase does not implement Runtime orchestration, run State, a complete Agent
flow, execution-close persistence, or end-to-end integration tests. References
to State and execution-close persistence below define ownership boundaries for
later work; they do not add those global modules to the current Component
implementation.

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
not manufacture zero values. `Usage.duration` is distinct from the duration
recorded for a Tool trace. Audio-token breakdown fields and `cache_write` are not
part of the
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
- A Tool does not create or update Trace records and does not call persistence.
  The caller records the execution result in State.

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
│   ├── sessions
│   │   └── compress
│   └── traces
└── long
    └── profiles
        └── compress

Knowledge
├── document
├── vector
└── hybrid
```

Memory resources use `create`, `read`, `update`, and `delete`. Do not use an
ambiguous `add` operation. Compression is not a top-level Component. It is an
internal Memory capability exposed as `Memory.short.sessions.compress` and
`Memory.long.profiles.compress`.

### Short Memory

Short memory is the current Session's persisted conversation and Tool execution
view. Its source of truth is `chats`, `sessions`, and `traces`; it does not
maintain a duplicate message cache.

#### Tool Traces

`Memory.short.traces` stores each complete Tool execution for a Chat. It is a
short-memory resource alongside `chats` and `sessions`, not an independent
Component. It is not a general Runtime log and is not a storage table for Model
calls, Plan nodes, Memory reads, Event delivery, or other internal steps.

- One Trace row represents one complete Tool execution attempt under
  `traces.chat_id = Chat.id`.
- `idx` is the required chronological index of the logical Tool execution in
  the Chat. All retries of that logical execution keep the same `idx`.
- A first execution uses `attempt = 1`. Every retry inserts another Trace row
  with the same `chat_id` and `idx`, and the next `attempt` value.
- `(chat_id, idx, attempt)` is unique. A new logical Tool execution increments
  `idx`; a retry never reuses an existing `attempt`.
- `key` has one meaning only: the registered Tool key used to select the Tool.
  It is not a display name, retry group ID, OpenTelemetry ID, or idempotency key.
- The Trace uses `input` for the Tool arguments and `message` for a successful
  Tool result, matching the Chat naming. A failed or stopped execution has no
  successful result message and records its details in `error`.
- Tool Traces do not have a `usage` field. `Chat.usage` remains the aggregate
  usage for the whole Chat.
- The OpenTelemetry `trace_id` belongs to the Telemetry layer and is not a Chat
  or Tool-trace business field.

For example, a weather Tool execution is represented conceptually as:

```json
{
  "id": 1,
  "chat_id": 100,
  "idx": 1,
  "key": "weather",
  "attempt": 1,
  "status": 3,
  "input": { "city": "天津" },
  "message": null,
  "error": { "code": "timeout" },
  "duration": 820,
  "created_time": 1786501800000,
  "updated_time": 1786501800820
}
```

If the Tool succeeds on the retry, the second row is:

```json
{
  "id": 2,
  "chat_id": 100,
  "idx": 1,
  "key": "weather",
  "attempt": 2,
  "status": 2,
  "input": { "city": "天津" },
  "message": {
    "role": "tool",
    "content": "{\"temperature\":31,\"weather\":\"rain\"}",
    "execute_id": "execute_1"
  },
  "error": null,
  "duration": 640,
  "created_time": 1786501800900,
  "updated_time": 1786501801540
}
```

The numeric status values are `1 = running`, `2 = completed`, `3 = failed`, and
`4 = stopped`. The exact Tool trace key format is an execution-layer detail; it
must not be confused with a display name or an OpenTelemetry trace ID.

Trace state lifecycle is:

```text
new logical Tool execution
  -> State allocates the next idx under the current chat_id
  -> State appends attempt = 1 with status = 1
  -> Tool succeeds: State writes message, duration, status = 2
  -> Tool fails: State writes error, duration, status = 3
  -> Tool is cancelled: State writes error, duration, status = 4
  -> retry: State appends the same idx with attempt + 1
  -> Execution reaches completed / failed / stopped
  -> the Execution close step writes all terminal Trace snapshots together
```

`idx` and `attempt` are execution facts allocated in State. `Memory.short.traces`
and its storage implementation do not calculate either value. Parallel Tool
execution must use the future State implementation's single-writer or atomic
allocation boundary so two logical executions cannot receive the same `idx`.

The Tool Component, retry branch, cancellation branch, and individual Plan Node
must not write Trace storage independently. They only update the current State.
When the current Execution reaches a terminal status, its single close step
converts all collected attempts into Trace records and writes them through
`Memory.short.traces`. This is a Runtime/Execution flow responsibility to be
implemented later, not another Component. A successful retry preserves the
earlier failed attempt as a separate terminal Trace.

`Memory.short.traces.create / read / update / delete` are storage capabilities,
not execution lifecycle hooks. In the normal execution flow, terminal attempts
are created together by the Execution close step; `update` remains an ordinary
CRUD capability and is not called after every Tool state change.

The persistence layer must enforce the retry identity with a unique index:

```sql
UNIQUE KEY uk_trace_attempt (chat_id, idx, attempt)
```

The `sessions` record contains the Session's ownership and lifecycle fields,
including `tenant_id`, `app_id`, `agent_id`, `user_id`, `title`, `status`,
`summary`, `created_time`, and `updated_time`. The Session record also has a JSON
extension field for related state. The JSON extension stores the Chat interval
covered by `summary`; it does not introduce another summary table or a separate
memory progress object.

- `app_id` identifies the application that owns the Session.
- `agent_id` is the stable Agent identity supplied to the framework. Application
  mapping from an AppAgent record to this value is completed before constructing
  the schema record. A Session has no `type` or `team_id`: short memory always
  belongs to this `agent_id`, including when a Chat is executed by a Team.
- A Chat belongs to the same stable `agent_id` as its Session. It does not have
  `type` or `team_id`; Team-internal execution identities belong to process
  records rather than short-memory ownership. A Chat also keeps the resolved
  Model snapshot in `models`.
- Runtime Model configuration may contain credentials such as `api_key`.
  `Chat.models` is a separate immutable diagnostic snapshot built from an
  explicit allowlist. Each selected Model records only `id`, `provider`, `name`,
  and `base_url`; it never serializes `api_key`, authentication headers,
  cookies, or other secrets. The snapshot describes historical execution and
  is not executable Model configuration.
- `trust` and `feedback` are independent integers limited to `-1 / 0 / 1`.
  `trust` controls context and memory reuse; `feedback` records the user's direct
  positive, neutral, or negative evaluation.

- `chats.create` persists the Chat with `status = running` when a new Chat
  starts. `chats.update` writes its terminal `completed / failed / stopped`
  state. Chat status is numeric: `1 = running`, `2 = completed`, `3 = failed`,
  `4 = stopped`. Persistence does not itself trigger summary generation or
  long-memory processing; only completed, trusted records are eligible later.
- One external user input creates one Chat record. Internal Tool calls and each
  retry row are persisted as Tool traces under the same `chat_id`. Plan nodes,
  Model calls, and intermediate Model messages are not Tool traces; their
  persistence is not defined by the Chat or Tool trace contract. `ref_id` is
  reserved for the explicit regenerate relation and is not a general parent
  relation.
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
Memory.short.sessions.compress(
  tenant_id, app_id, agent_id, user_id, session_id,
  model, metadata, cancellation
)
  -> read prior sessions.summary + the next eligible Chat range
  -> invoke Model with the selected content
  -> atomically update sessions.summary + extra.idx
```

The Model invocation receives only the content selected by Memory. It does not
read database tables or write database records; the public `compress` operation
owns those reads and writes.

#### Session Context Compression

Before every normal Chat Model call, the caller assembles the current Context
and estimates its input size. The estimate includes the fixed system/developer
messages, available Tool definitions, `sessions.summary`, the selected recent
Chats, the current Chat input, and any other data explicitly selected for this
call.

The effective input budget is derived from the selected Model's limits:

```text
effective input budget
  = context_tokens - reserved output_tokens - safety margin
```

If the assembled Context exceeds that budget, only the current Session's older
conversation is compressed. This is Session Summary compression, not long-memory
compression and not Knowledge retrieval. `Memory.short.sessions.compress` uses
the prompt and bounded input/output options configured when Memory is
constructed; it does not recursively trigger Session Summary compression.

The caller only requests compression and then rebuilds the Context. The concrete
`Sessions` implementation reads its existing summary coverage and uses its own
configured rule to select the next oldest eligible contiguous Chat range. It
keeps the newest Chats raw and then returns the updated Session:

```text
sessions.summary = none
+ chat 1 + chat 2 + chat 3 + chat 4
+ current input chat 5
  -> Context exceeds budget
  -> Memory.short.sessions.compress(
       tenant_id,
       app_id,
       agent_id,
       user_id,
       session_id,
       model,
       metadata,
       cancellation
     )
  -> implementation selects chats 1..2
  -> atomically update sessions.summary + extra.idx = { from: 1, to: 2 }
  -> sessions.summary covering 1..2
+ chat 3 + chat 4
+ current input chat 5
  -> Model
```

The example uses four completed Chats and a fifth current input. The
implementation may select a different oldest range; it does not have to compress
the entire Session. The summary update advances its `idx.to` only after
compression and the Session update both succeed. A failed, invalid, or cancelled
compression does not advance the summary range.

If all eligible historical Chats have been compressed and the Context is still
over budget, the caller returns a context-size error or applies an explicit
input-handling policy. It must not keep recursively compressing the same
summary, silently drop the current input, or use long-memory compression as a
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
one shared Profile in the Tenant. A Profile is one aggregate record; its
structured `content` holds the current set of durable facts.

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

  content JSON NOT NULL,
  extra JSON NOT NULL,

  version INT NOT NULL DEFAULT 0,
  created_time BIGINT NOT NULL DEFAULT 0,
  updated_time BIGINT NOT NULL DEFAULT 0,

  UNIQUE KEY uk_owner (tenant_id, type, agent_id, user_id)
);
```

- `id` is the Profile's stable primary key.
- `type` currently allows only `user` and `agent`.
- `content` is the complete aggregated Profile JSON. User and Agent Profiles use
  the same schema:

```json
{
  "facts": [
    {
      "content": "用户喜欢吃辣",
      "time": 1786501800000
    },
    {
      "content": "用户今天步行上班",
      "time": 1786761000000
    }
  ]
}
```

  - `facts[].content` is one durable fact retained by compression.
  - `facts[].time` is the millisecond evidence time for that fact. For an
    automatically compressed User Profile, it must be the source Chat's
    `created_time`; for an Agent Profile written by an explicit management flow,
    it is that confirmed write's time.
  - `Profile.created_time` and `Profile.updated_time` describe the aggregate row
    itself and never replace `facts[].time`.
  - An empty Profile uses `{ "facts": [] }`; `content` is never a plain text
    value.
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
  `running`, `failed`, `stopped`, and `trust < 0`.
- `trust` is an integer used to control contextual reuse and long-memory
  processing. Regenerating a Chat marks the earlier result untrusted. Explicit
  user feedback is stored separately as an integer and does not replace
  `trust`.
- Tool traces are not a long-memory compression source. Tool data can be too
  large and requires a separate decision before it may be included.
- Long-memory processing reads the owner's existing Profile content so the
  result can consolidate it rather than create another Profile.
- `Memory.long.profiles.compress` reads the selected Chats and existing Profile,
  then invokes the supplied Model with a Message list: a System Message carrying
  the compression prompt and a User Message carrying serialized selected Chats,
  their `created_time`, and existing Profile content. It is not a single
  combined Message.
- The Model produces one complete replacement `content` JSON. A compression
  implementation validates each returned fact before saving: a newly added User
  fact must use the `created_time` of one selected Chat; a fact retained from
  the existing Profile keeps its original time. The Model must not invent fact
  times.
- Memory creates the owner's Profile when absent, otherwise updates that single
  Profile, and the operation returns the persisted Profile.

```text
Memory.long.profiles.compress(
  tenant_id, agent_id, user_id, type,
  session_id, model, metadata, cancellation
)
  -> read existing Profile + choose the next unprocessed Chat range
  -> invoke Model
  -> create or optimistically update Profile
```

The long-memory processing range is independent from the Session summary range.
It is stored in the Profile's JSON `extra` field. `extra` is a map
whose key is the numeric `session_id`; JSON represents that numeric map key as
a string. It does not add fields to `sessions` and does not use
`summary_chat_id` or `memory_chat_id`:

```json
{
  "101": {
    "from": 1,
    "to": 10,
    "update_time": 1786501800000
  }
}
```

`from` and `to` are the inclusive Chat `idx` range that has been successfully
consolidated into this Profile for that Session. `update_time` is the
captured `sessions.updated_time` for that successfully processed range, not the
time at which the long-memory row itself was written.

The stored Profile progress is cumulative. On the first successful compression,
its `from` is the selected range's `idx.from`. Later successful contiguous
compressions preserve that `from` and advance only `to`. The selected range is
an internal fact of the Profile implementation and is not a `compress` argument.

All persisted Session, Chat, and Profile timestamps use `created_time` and
`updated_time`. They do not use parallel `created_at` or `updated_at` names.

When selecting Sessions for processing, the caller compares each Session's
current `sessions.updated_time` with the corresponding `extra[session_id]` value.
A missing entry, or a Session whose current update time is newer than the stored
`update_time`, is eligible. The caller only triggers `profiles.compress`; the
Profile implementation reads Chats after the stored `to` and selects the next
contiguous range according to its own configured rule. This comparison prevents
unchanged Sessions from being scheduled repeatedly.

The Profile implementation captures both the Session's current `updated_time`
and the selected `idx.to`. After the write succeeds, those captured values are
stored together.
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

A trigger only schedules work. Before execution, the Profile implementation
rereads the current Profile, Session state, and eligible Chats, then fixes the
exact `idx` range for that attempt. Multiple triggers for the same owner may be
coalesced; they must not execute overlapping writes concurrently.

#### Long-Memory Concurrency

The Profile contains a `version` used for optimistic locking. A write
must match the version read before `Memory.long.profiles.compress` and increment it on
success.

If the version no longer matches, another process has changed the owner's
memory. The current attempt must discard its generated replacement `content`,
reread the newest `content`, `extra`, and `version`, recalculate the unprocessed
Chat range, and run `Memory.long.profiles.compress` again. It must not apply the stale
content or retry only the database update.

Only the successful optimistic-lock write updates `content`, `extra`, and
`version` together. This keeps the aggregated memory and every Session progress
entry consistent.

#### No Memory Change

`Memory.long.profiles.compress` has four result cases:

- Existing Profile and eligible Chats: invoke the Model and update that Profile.
- Missing Profile and eligible Chats: use `{ "facts": [] }` as previous
  `content`, invoke the Model, and create the first Profile.
- Existing Profile and no eligible Chats: do not invoke the Model or write
  storage; return the existing Profile unchanged.
- Missing Profile and no eligible Chats: do not create an empty record; return
  `Error::NotFound`.

The Model may determine that selected Chats contain no durable new fact,
preference, or confirmed decision. The selected range was still processed, so
its `from`, `to`, and `update_time` advance through the same write while
`content` remains unchanged. If this is the owner's first processed range, the
implementation creates a Profile with `{ "facts": [] }` and the completed
progress entry. This prevents the same Chats from being compressed repeatedly;
it is different from the missing-Profile/no-eligible-Chats case, where no record
is created.

Automatic long-memory compression updates only the current user's `type = user`
record. An Agent's `type = agent` record is updated only by an explicit
administrator or application workflow. A user's conversation never
automatically becomes Agent-shared memory. Long memory may be reused across
Sessions for its owner; the compression source interval remains Session-local.

### Memory Interface Boundary

`Memory` owns short-memory and Profile persistence as well as its internal
compression flows. `Knowledge` remains a separate Component:

```text
Memory
├── short
│   ├── chats
│   │   └── create / read / update / delete
│   ├── sessions
│   │   └── create / read / update / delete / compress
│   └── traces
│       └── create / read / update / delete
└── long
    └── profiles
        └── create / read / update / delete / compress

Knowledge
├── document
├── vector
└── hybrid
```

Storage implementations, compression prompts, and compression options are
initialized once and injected into `Memory`. `Memory` is then injected into the
Agent. For each operation, Runtime supplies the scope fields required by that
operation, the selected Model and read-only `Metadata` when compression is
requested, and `Cancellation`; Runtime does not implement database access or
retain persisted Session, Chat, or Profile state. A scheduled or manual
long-memory worker supplies its own operation Metadata; Memory does not invent
Run metadata.

The CRUD methods below describe Component capabilities only. The current phase
does not implement the future Execution close step that reads terminal Trace
facts from State and invokes these methods at one boundary.

Conceptual CRUD operations are:

```text
Memory.short.chats.create(chat)
Memory.short.chats.read(tenant_id, app_id, agent_id, user_id, session_id, idx)
Memory.short.chats.update(chat_id, message, usage, status)
Memory.short.chats.delete(chat_id)

Memory.short.traces.create(trace)
Memory.short.traces.read(tenant_id, agent_id, user_id, session_id, chat_id)
Memory.short.traces.update(trace_id, message, error, duration, status)
Memory.short.traces.delete(trace_id)

Memory.short.sessions.create(session)
Memory.short.sessions.read(tenant_id, app_id, agent_id, user_id, session_id)
Memory.short.sessions.update(session)
Memory.short.sessions.delete(session_id)

Memory.long.profiles.create(profile)
Memory.long.profiles.read(tenant_id, agent_id, user_id, type)
Memory.long.profiles.update(profile, version)
Memory.long.profiles.delete(profile_id)
```

Short-memory `read` operations and Session `compress` require the complete
ownership key `tenant_id + app_id + agent_id + user_id + session_id`. The
storage operation matches all five fields in one query. Every returned Chat has
the same stable `agent_id` as the owning Session.

`Chat.message` is the optional final `Message::Assistant` delivered by the Chat.
Streaming deltas, Tool results, and intermediate Model messages are not stored
in this field. A running or unsuccessfully terminated Chat may have no final
message.

Chat runtime close updates `message`, `usage`, and `status`. Cancellation
preserves the Assistant Message and aggregate usage received before cancellation
and writes `status = 4`; the other Chat business fields are not changed.

`Chat.input` remains a JSON value because it is the persistable view of the
user's submitted input, not a Model `Message`. It preserves the text needed by
later Context assembly and memory compression. Text input is stored directly;
image or file input keeps the accompanying text plus stable references and
necessary metadata; audio input keeps its recognized text plus optional audio
reference and metadata. Inline binary or Base64 data is not persisted in the
Chat record. The caller converts this persisted view into `Message::User` only
when assembling Model input.

A Chat does not store an OpenTelemetry `trace_id`. Tool trace records reference
the Chat through `traces.chat_id = Chat.id`; OpenTelemetry trace context remains
owned by the Telemetry layer.

Each `delete` operation identifies its resource only by that resource's own ID:
`chat_id`, `session_id`, or `profile_id`. `create` and `update` receive complete
domain records instead of separate ownership parameters.

Read operations return persisted domain records. They do not return Model
`Message` values. The caller that assembles Model context converts selected Chat
and Session records into Messages.

CRUD implementations validate domain records before create or update and return
the persisted snapshot after a successful write. `chats.read` treats `idx` as an
inclusive fixed range, returns records in ascending `Chat.idx` order, and returns
`NotFound` when any index in that range is absent. It does not silently return a
partial range.

#### Short Compression Flow

The conceptual operation is:

```text
Memory.short.sessions.compress(
  tenant_id,
  app_id,
  agent_id,
  user_id,
  session_id,
  model,
  metadata,
  cancellation
)
```

Memory reads the current Session summary, selects the next eligible continuous
Chat range according to its own configured rule, invokes the supplied Model,
validates the compression result, and atomically updates `sessions.summary`
together with the Session JSON summary range. A failure or cancellation leaves
both values unchanged.

When no summary exists, the selected range starts at `idx.from = 1`. When a
summary already exists, the selected range starts at the current
`sessions.extra.idx.to + 1`. The implementation must select a continuous range
and must not skip an eligible Chat. Implementations serialize compression per
Session or perform an equivalent conditional write; if the stored summary range
changes before the write, stale output must not overwrite it.

#### Long Compression Flow

The conceptual operation is:

```text
Memory.long.profiles.compress(
  tenant_id,
  agent_id,
  user_id,
  type,
  session_id,
  model,
  metadata,
  cancellation
)
```

Memory reads the current Profile and selected Session Chats, invokes the
supplied Model, validates the replacement Profile content, and creates or
optimistically updates the Profile. The successful write updates `content`,
`extra`, and `version` together. When no durable information is found, `content`
remains unchanged while the Session progress and `version` still advance.

If no eligible Chat exists, this operation does not invoke the Model. It returns
the existing Profile unchanged, or returns `NotFound` when the owner has no
Profile.

The Model is supplied by Runtime from the current resolved Agent execution.
Memory does not bind a Provider and does not implement Provider retry or
streaming behavior. Memory is responsible for selecting persisted data, running
the compression flow, validating its result, writing it, and enforcing
optimistic locking.

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

- `InvalidInput`: an invalid owner type or another invalid operation argument.
- `NotFound`: a requested record does not exist or a requested fixed Chat range
  is incomplete.
- `Storage`: a persistence read or write failed.
- `Model`: `Memory.short.sessions.compress` or `Memory.long.profiles.compress`
  failed while invoking the supplied Model.
- `InvalidOutput`: the Model call completed but its compression result did not
  satisfy the required output contract.
- `Conflict`: a create operation conflicts with an existing record, a direct
  optimistic update sees a version mismatch, or compression exhausts its
  optimistic-lock retries.
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

A normal `profiles.read` returns `NotFound` when the Profile does not exist. A
missing Profile is not an error inside long compression when eligible Chats
exist; Memory uses empty previous content and creates the first Profile. If no
eligible Chat exists, long compression returns `NotFound` and creates nothing.

If the range selected internally by a compression implementation is invalid or
non-contiguous, the operation fails before calling the Model. Profile version
conflicts are handled inside Memory by rereading and rerunning compression;
`Conflict` is returned only after that retry limit is exhausted. Cancellation is
never retried and never advances summary or Profile progress.

## Knowledge

`Knowledge` manages externally imported knowledge independently from
conversation `Memory`:

```text
Knowledge
├── document
│   └── create / read / update / delete / search
├── vector
│   └── create / read / update / delete / search
└── hybrid
    └── search
```

- `document` owns original knowledge documents and text search.
- `vector` owns vector records and vector search. `model::Embedding` generates
  vectors but does not store or search them.
- `hybrid` owns backend-native hybrid retrieval. It is a capability under
  `Knowledge`, not a top-level Component.

An original document uses this Schema:

```rust
pub struct Document {
    pub id: u64,
    pub tenant_id: u64,
    pub app_id: u64,
    pub agent_id: u64,
    pub user_id: u64,
    pub scope: Scope,
    pub title: String,
    pub content: String,
    pub ext: Option<String>,
    pub url: Option<String>,
    pub status: Status,
    pub metadata: Metadata,
    pub version: u64,
    pub created_time: u64,
    pub updated_time: u64,
}
```

`scope` is numeric: `1 = user` requires `user_id > 0`; `2 = agent` requires
`user_id = 0`. It describes visibility ownership only. Future read, write, or
administrative permissions remain a separate concern.

`status` is always serialized and persisted as a number: `1 = pending`,
`2 = processing`, `3 = ready`, and `4 = failed`. The Rust `Status` variants are
typed constants for these numbers; the boundary never sends or stores the
strings `"pending"`, `"processing"`, `"ready"`, or `"failed"`. `title` is
required. `content` contains parsed text and may remain empty until parsing
completes; `url` references the original file, so at least one of `content` and
`url` must be present. A Document with `status = 3` must have non-empty
`content`. `ext` is the lowercase original file extension without a leading
dot, such as `pdf`, `docx`, or `tar.gz`.

One Document may produce multiple Wiki retrieval units. Each Wiki keeps the
source relation through `Wiki.doc_id = Document.id`; `Wiki.idx` and
`Wiki.spans` describe its order and location inside the Document. Token,
graph, and other retrieval-unit fields remain on Wiki rather than being copied
into Document.

`Document.id` and the Knowledge Schema's `Wiki.doc_id` are both `u64`. The
existing standalone RAG crate still uses its previous string `doc_id`; migrating
that crate is outside the current Schema and Component phase.

`Document` is a persistence capability. It accepts a complete Document and does
not perform upload, download, file parsing, content extraction, Wiki splitting,
Embedding, or workflow orchestration:

```text
document.create(Document) -> Document
document.read(Read) -> Document
document.update(Update) -> Document
document.delete(id)
document.search(dsl) -> Documents
```

The caller constructs the complete Document, including `id`, ownership, scope,
content, URL, numeric status, version, and timestamps. Document validates and
persists that object without deriving status or filling business fields. An
upper-layer flow may prepare that object before calling
`document.create(document)`, but Document does not know or execute those
preparation steps.

`Read` matches `id + tenant_id + app_id + agent_id + user_id`. A missing record
or ownership mismatch returns `NotFound`. `Update` identifies the record by
`id`, checks the supplied `version`, and returns `Conflict` when it is stale. It
may update `scope` and `user_id` together, allowing `user -> agent` and
`agent -> user` visibility changes. `tenant_id`, `app_id`, and `agent_id` are
not mutable. A successful update increments `version` and refreshes
`updated_time`.

`document.delete(id)` deletes only the original Document record. It does not
delete Wiki or Vector data and does not invoke another Component. If an
upper-layer business flow needs related data removed, it explicitly calls each
required capability and owns that orchestration and consistency policy.

All Knowledge searches receive one JSON object named `dsl`. Query expressions,
ownership and scope filters, pagination, sorting, and implementation-specific
options belong inside that object. The Component does not define fixed `query`,
`offset`, or `limit` fields and does not prescribe an Elasticsearch or database
DSL. Each implementation validates and interprets its own DSL.

Document search returns original Document records:

```rust
pub struct DocumentHit {
    pub document: Document,
    pub score: f64,
}

pub struct Documents {
    pub total: u64,
    pub hits: Vec<DocumentHit>,
}
```

Vector and Hybrid search share one Wiki hit shape:

```rust
pub struct Hit {
    pub wiki: Wiki,
    pub score: f64,
}

pub struct Wikis {
    pub total: u64,
    pub hits: Vec<Hit>,
}
```

`Documents` is returned only by `document.search`; `Wikis` is returned by
`vector.search` and `hybrid.search`. Neither is named `SearchResult` or
`Output`. Each hit's `score` is the final score used by the current search
operation.
The backend-specific hybrid query and scoring details are not exposed in the
unified Schema. `Wiki` never carries an embedding vector in a search response.
`Wiki.embedding` remains an optional field so the same structure can be used by
Knowledge storage and Vector operations. Vector and Hybrid search must clear it
to `None` before returning `Wikis`; when it is `None`, serialization omits the
field entirely.

`Knowledge.vector` uses the full Wiki structure for its CRUD contract:

```text
vector.create(Wiki) -> Wiki
vector.batch_create(Vec<Wiki>) -> Vec<Wiki>
vector.read(id) -> Wiki
vector.update(Wiki) -> Wiki
vector.delete(id)
vector.search(dsl) -> Wikis
```

`Wiki.id` is a required `u64`. `create`, `batch_create`, `read`, and `update`
may carry `Wiki.embedding`; Vector uses it for its stored vector data.
`batch_create` receives complete Wiki records and returns the records persisted
by that call. Its atomicity and partial-failure behavior are defined by the
concrete Vector implementation. `vector.search` is the same return boundary as
Hybrid search, so every returned Wiki must have `embedding = None`.

Vectorization is an independent operation. `document.create`,
`document.update`, and Wiki text storage do not invoke an Embedding Model and do
not call Vector. A caller, such as a scheduled business task, reads Documents,
splits them into Wikis, obtains embeddings through its chosen mechanism, and
explicitly calls `vector.create` or `vector.batch_create`. Vector itself does
not select or invoke an Embedding Model. Deleting Document, Wiki, or Vector data
is also explicit business orchestration; no Component performs cascading deletes.

`Wiki.validate()` requires a non-zero `id`, a non-zero `doc_id`, and non-empty
`content`. `Wiki.validate_vector()` additionally requires a non-empty embedding
whose values are all finite. Vector create, batch create, and update call the
latter; search results only need the base Wiki validation. The exact index layout
and storage implementation are not prescribed.

`Knowledge.hybrid` has no CRUD operations:

```text
hybrid.search(dsl) -> Wikis
```

Hybrid interprets its own JSON DSL and invokes the backend's native hybrid
retrieval capability, such as Elasticsearch hybrid search. It does not call
`document.search` or `vector.search`, and does not implement application-side
score fusion, deduplication, or ordering. It returns the same `Wikis` shape as
Vector search and clears `Wiki.embedding` before returning. The contract does
not prescribe a backend, query syntax, or scoring algorithm.
