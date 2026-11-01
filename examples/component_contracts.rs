//! Small, offline examples for the low-level component contracts.
use halo::agent::core::sandbox::{Config as SandboxConfig, Local};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::agent::{
    event::Event,
    function::Function,
    knowledge::keyword,
    message::Role,
    model::{content::part::Part, embed, multimodal, tts},
    prompt::{Template, Variables},
    sandbox::{Request as SandboxRequest, Sandbox},
};
use serde_json::{Map, json};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let cancellation = Cancellation::new();

    // Function is a standalone async business operation. It does not know
    // about Workflow state or Agent execution identifiers.
    let increment = |value: u64, _: Cancellation| async move { Ok::<_, Error>(value + 1) };
    let value = Function::call(&increment, 41, &cancellation).await?;
    println!("function result = {value}");

    // Prompt formats Message values and expands a named Messages history.
    let mut question = Message::new(Role::User);
    question.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!("问题：{question}"))]),
    });
    let rendered = Template::new()
        .message(question)
        .format(&Variables::new().value("question", json!("北京天气怎么样")))?;
    println!("prompt messages = {rendered:?}");

    // Search requests are explicit: keyword, vector and hybrid have separate
    // request shapes. The backend is supplied by the application.
    let keyword_request = keyword::Request {
        query: "北京天气".into(),
        options: Map::new(),
    };
    keyword_request.validate()?;

    // Embed responses use the shared Message shape: vector in values and
    // provider/model/dimension in metadata.
    let embedding = embed::message(vec![0.1, 0.2, 0.3], "local-example", "embed-v1", Some(3))?;
    println!("embedding = {embedding:?}");

    // TTS and multimodal models keep their own request contracts while
    // sharing the same Event type. Validation is local and does not invent a
    // provider response or require a network call.
    let mut spoken = Message::new(Role::User);
    spoken.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!("请朗读这段文字"))]),
    });
    tts::Request {
        messages: Messages::new([spoken.clone()]),
        voice: Some("default".into()),
        format: Some("audio/wav".into()),
        speed: Some(1.0),
        stream: false,
        options: Map::new(),
    }
    .validate()?;

    let mut vision = Message::new(Role::User);
    vision.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!("描述这张图片"))]),
    });
    vision.content.push(Part {
        r#type: "image".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({"kind":"url", "value":"https://example.invalid/image.png", "mime_type":"image/png"}),
        )]),
    });
    multimodal::Request {
        messages: Messages::new([vision]),
        modalities: vec!["text".into()],
        stream: false,
        ..multimodal::Request::default()
    }
    .validate()?;

    // Sandbox is an explicit capability. This executes a real local command;
    // it is not an isolation boundary and should only receive trusted input.
    let sandbox = Local::new(SandboxConfig::default())?;
    let response = sandbox
        .execute(
            SandboxRequest {
                operation: "command".into(),
                input: json!({"args": ["printf", "sandbox-ok"]}),
                metadata: Map::new(),
            },
            &cancellation,
        )
        .await?;
    println!("sandbox response = {}", response.data);

    // Event is the observation contract shared by Model, Agent and Workflow.
    let mut complete = Message::new(Role::Assistant);
    complete.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!("组件示例完成"))]),
    });
    let event = Event::Complete {
        message: complete,
        usage: None,
        finish_reason: Some("stop".into()),
    };
    println!("event = {event:?}");
    Ok(())
}
