//! Component acceptance backed by actual DeepSeek responses.
//! These complement deterministic contract/failure tests; they do not replace them.

use futures_util::{StreamExt, stream::Map};
use halo::agent::core;
use halo::agent::{
    self, Agent, Cancellation, Error, Input, Message, Messages, Payload,
    compression::Compressor as _,
    document::Document,
    evaluation::Evaluator as _,
    event::Event,
    memory::{self, extraction::Extractor as _},
    message::Role,
    model::{chat, content::part::Part},
    plan::Planner as _,
    provider::deepseek::DeepSeek,
    reason::Reasoner as _,
};
use serde_json::{Value, json};

fn deepseek_chat() -> impl chat::Chat {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real Chat")
}

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": value}).as_object().unwrap().clone(),
    });
    message
}

fn assistant_text(message: &Message) -> String {
    message
        .content
        .iter()
        .filter(|part| part.r#type == "text")
        .filter_map(|part| part.data.get("value").and_then(Value::as_str))
        .collect()
}

fn log_plan_response(event: Result<Event, Error>) -> Result<Event, Error> {
    if let Ok(Event::Complete { message, .. }) = &event {
        println!("LIVE_PLANNER_RAW_TEXT={}", assistant_text(message));
        println!("LIVE_PLANNER_RAW_VALUES={:?}", message.values);
    }
    event
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_graph_extraction_returns_entities_and_a_declared_relation() {
    let graph = core::document::graph::Graph::new(
        serde_json::from_value(json!({
            "entity_types": [
                {"name": "person", "description": "人物"},
                {"name": "company", "description": "公司"}
            ],
            "relation_types": [
                {"name": "works_for", "description": "在某公司任职", "from": ["person"], "to": ["company"]}
            ]
        }))
        .expect("valid graph extraction options"),
        deepseek_chat(),
    )
    .expect("construct Graph extraction");
    let output = graph
        .run(
            &[Document {
                content: "张三在甲公司工作。".into(),
                ..Document::default()
            }],
            &Cancellation::new(),
        )
        .await
        .expect("real model should extract the declared entities and relation");

    assert!(
        output
            .entities
            .iter()
            .any(|entity| { entity.r#type == "person" && entity.name.contains("张三") }),
        "real Graph response omitted the source person: {output:?}"
    );
    assert!(
        output
            .entities
            .iter()
            .any(|entity| { entity.r#type == "company" && entity.name.contains("甲公司") }),
        "real Graph response omitted the source company: {output:?}"
    );
    assert!(
        output.relations.iter().any(|relation| {
            relation.kind == "works_for"
                && output
                    .entities
                    .iter()
                    .any(|entity| entity.id == relation.from && entity.r#type == "person")
                && output
                    .entities
                    .iter()
                    .any(|entity| entity.id == relation.to && entity.r#type == "company")
        }),
        "real Graph response omitted the declared relation: {output:?}"
    );
    println!(
        "LIVE_GRAPH_OUTPUT={}",
        serde_json::to_string(&output).unwrap()
    );
}

struct InspectChat<C>(C);

impl<C: chat::Chat> chat::Chat for InspectChat<C> {
    type Stream = Map<C::Stream, fn(Result<Event, Error>) -> Result<Event, Error>>;

    async fn stream(&self, request: chat::Request) -> Result<Self::Stream, Error> {
        Ok(self
            .0
            .stream(request)
            .await?
            .map(log_plan_response as fn(Result<Event, Error>) -> Result<Event, Error>))
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_planner_returns_a_valid_complete_plan() {
    let planner = core::plan::Planner::new(InspectChat(deepseek_chat()), chat::Request::default())
        .expect("construct Planner");
    let input = Input {
        payload: vec![halo::agent::Payload {
            r#type: "text".into(),
            data: json!({"value":"规划两步完成任务：先查询北京天气，再根据天气给出是否适合露营的建议。请严格返回合法 JSON：version 必须是 JSON 整数 1（不是字符串），nodes 是以逻辑节点 ID 为 key、值仅含 name/objective 的对象，edges 使用 from/to。"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let plan = planner
        .plan(&input, None, None, &Cancellation::new())
        .await
        .expect("real model should produce a valid complete plan");

    assert!(
        !plan.nodes.is_empty(),
        "real Planner returned no logical nodes"
    );
    assert!(plan.validate().is_ok());
    println!(
        "LIVE_PLANNER_PLAN={}",
        serde_json::to_string(&plan).unwrap()
    );
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_evaluator_judges_the_final_agent_result() {
    let evaluator = core::evaluation::Evaluator::new(
        deepseek_chat(),
        chat::Request {
            messages: Messages::new([text(
                Role::System,
                "判断用户要求与最终回答是否一致。用户问“2 加 2 等于几”，最终回答是“2 加 2 等于 4”。这是正确回答，应判通过且不需要反馈。严格只输出一个 JSON 对象，格式为 {\"passed\":true,\"feedback\":null}；passed 必须是 boolean，feedback 必须是 null 或 Messages 数组。不要输出 Markdown、代码围栏、解释或其他字段。",
            )]),
            ..Default::default()
        },
    )
    .expect("construct Evaluator");
    let input = Input {
        payload: vec![halo::agent::Payload {
            r#type: "text".into(),
            data: json!({"value":"2 加 2 等于几？"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let result = Messages::new([text(Role::Assistant, "2 加 2 等于 4。")]);
    let evaluation = evaluator
        .evaluate(&input, &result, &Cancellation::new())
        .await
        .expect("real Evaluator should return a structured judgment");

    assert!(
        evaluation.passed,
        "real Evaluator rejected a correct result"
    );
    assert!(evaluation.feedback.is_none());
    println!(
        "LIVE_EVALUATION={}",
        serde_json::to_string(&evaluation).unwrap()
    );
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends two live model requests"]
async fn real_agent_completion_is_checked_by_a_real_evaluator() {
    let evaluator = core::evaluation::Evaluator::new(
        deepseek_chat(),
        chat::Request {
            messages: Messages::new([text(
                Role::System,
                "判断助手是否正确回答了用户问题。用户问 2+2 等于几；回答包含数字 4 即通过。返回 JSON: {\"passed\":boolean,\"feedback\":null}。",
            )]),
            ..Default::default()
        },
    )
    .expect("construct real final Evaluator");
    let agent = agent::chat()
        .model(deepseek_chat())
        .system_prompt("回答用户的算术问题，简短作答。")
        .evaluator(evaluator)
        .build()
        .expect("build Agent with real Evaluator");
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"2+2 等于几？"}),
            mime_type: None,
        }],
        ..Default::default()
    };

    let mut events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("start real Agent run");
    let mut completed = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.expect("Agent event")
            && message.role == Role::Assistant
        {
            completed = Some(message);
        }
    }
    let completed = completed.expect("Agent should emit an assistant Complete");
    let answer = assistant_text(&completed);
    assert!(
        answer.contains('4'),
        "real Agent answer did not contain the correct result: {answer:?}"
    );
    println!("LIVE_AGENT_EVALUATED_ANSWER={answer}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_reasoner_returns_the_provider_complete_message() {
    let reasoner = core::reason::Reasoner::new(
        deepseek_chat(),
        chat::Request {
            messages: Messages::new([text(
                Role::System,
                "用简体中文回答用户的问题，并给出必要的简短解释。",
            )]),
            ..Default::default()
        },
    )
    .expect("construct real Reasoner");
    let answer = reasoner
        .reason(
            Messages::new([text(Role::User, "地球为什么有四季？请简要解释。")]),
            &Cancellation::new(),
        )
        .await
        .expect("real Reasoner should return the model Complete");
    let complete = answer
        .0
        .first()
        .expect("Reasoner should return its complete Message");
    assert_eq!(complete.role, Role::Assistant);
    assert!(!assistant_text(complete).trim().is_empty());
    assert_eq!(
        complete
            .metadata
            .get("provider_name")
            .and_then(Value::as_str),
        Some("deepseek")
    );
    assert!(
        complete.usage.is_some(),
        "real Provider usage should survive"
    );
    println!("LIVE_REASONER_COMPLETE={complete:?}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_memory_extractor_returns_typed_durable_content() {
    let extractor =
        core::memory::extraction::Extractor::new(deepseek_chat(), chat::Request::default())
            .expect("construct long-memory Extractor");
    let conversation = Messages::new([
        text(
            Role::User,
            "请记住我的长期偏好：以后都用简体中文，并尽量简洁回答。",
        ),
        text(Role::Assistant, "好的，我会尽量用简体中文简洁回答。"),
    ]);
    let extracted = extractor
        .extract(&conversation, &Cancellation::new())
        .await
        .expect("real Extractor should return schema-valid memory");
    let values = &extracted.0[0].values;
    let preference = values
        .get("user")
        .and_then(Value::as_object)
        .and_then(|user| user.get("preference"))
        .and_then(Value::as_str)
        .expect("explicit long-term preference should be extracted");
    assert!(preference.contains("中文") || preference.contains("简洁"));
    println!(
        "LIVE_MEMORY_EXTRACTION={}",
        serde_json::to_string(values).unwrap()
    );
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_memory_extractor_merges_existing_content() {
    let extractor =
        core::memory::extraction::Extractor::new(deepseek_chat(), chat::Request::default())
            .expect("construct long-memory Extractor");
    let current = memory::Item {
        id: 1,
        r#type: memory::Type::Preference,
        user_id: 2,
        agent_id: None,
        session_id: None,
        from: 1,
        to: 2,
        content: "偏好简洁回答".into(),
        version: 1,
        created_time: 0,
        updated_time: 0,
        metadata: None,
    };
    let merged = extractor
        .merge(&current, "请使用简体中文回答", &Cancellation::new())
        .await
        .expect("real Extractor should return merged replacement text");

    assert!(!merged.trim().is_empty());
    assert!(merged.contains("简洁") || merged.contains("中文"));
    println!("LIVE_MEMORY_MERGE={merged}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_compressor_returns_the_model_complete_message() {
    let compressor = core::compression::Compressor::new(
        deepseek_chat(),
        chat::Request {
            messages: Messages::new([text(
                Role::System,
                "把给定对话压缩成简体中文的简短上下文，保留用户偏好与已确定事项，只返回摘要。摘要必须保留‘露营’和‘简体中文’这两个关键词。",
            )]),
            max_tokens: Some(256),
            ..Default::default()
        },
    )
    .expect("construct Compressor")
    .with_size(10_000);
    let history = Messages::new([
        text(Role::User, "我周末想去露营，出发前要确认天气。"),
        text(Role::Assistant, "记得查看天气、风力和营地开放情况。"),
        text(Role::User, "以后请用简体中文简洁回答。"),
    ]);
    let compressed = compressor
        .compress(history, &Cancellation::new())
        .await
        .expect("real Compressor should return the model Complete message");
    let complete = compressed
        .0
        .first()
        .expect("Compressor should return one complete Message");
    let summary = assistant_text(complete);
    assert!(!summary.trim().is_empty());
    assert!(summary.contains("露营") || summary.contains("天气"));
    println!("LIVE_COMPRESSED_MESSAGE={complete:?}");
}
