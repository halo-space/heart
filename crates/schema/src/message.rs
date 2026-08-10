use serde::{Deserialize, Serialize};

use crate::tool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    Image {
        url: String,
        detail: Option<ImageDetail>,
    },
    Audio {
        data: String,
        format: String,
    },
    File {
        data: Option<String>,
        id: Option<String>,
        filename: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audio {
    pub id: String,
    pub data: Option<String>,
    pub transcript: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    pub index: u32,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
        name: Option<String>,
    },
    Developer {
        content: String,
        name: Option<String>,
    },
    User {
        content: Vec<Content>,
        name: Option<String>,
    },
    Assistant {
        content: Option<String>,
        audio: Option<Audio>,
        embeddings: Vec<Embedding>,
        name: Option<String>,
        refusal: Option<String>,
        executes: Vec<tool::Execute>,
    },
    Tool {
        content: String,
        execute_id: String,
    },
}

impl Message {
    pub const fn role(&self) -> Role {
        match self {
            Self::System { .. } => Role::System,
            Self::Developer { .. } => Role::Developer,
            Self::User { .. } => Role::User,
            Self::Assistant { .. } => Role::Assistant,
            Self::Tool { .. } => Role::Tool,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_messages_report_the_assistant_role() {
        let message = Message::Assistant {
            content: Some("transcribed text".into()),
            audio: None,
            embeddings: Vec::new(),
            name: None,
            refusal: None,
            executes: Vec::new(),
        };

        assert_eq!(message.role(), Role::Assistant);
    }

    #[test]
    fn user_messages_support_file_and_text_content() {
        let message = Message::User {
            content: vec![
                Content::File {
                    data: Some("Base64...".into()),
                    id: None,
                    filename: Some("report.txt".into()),
                },
                Content::Text {
                    text: "帮我总结这个文件".into(),
                },
            ],
            name: None,
        };

        assert_eq!(message.role(), Role::User);
        assert_eq!(
            match message {
                Message::User { content, .. } => content.len(),
                _ => 0,
            },
            2
        );
    }

    #[test]
    fn embedding_preserves_provider_batch_index() {
        let embedding = Embedding {
            index: 1,
            embedding: vec![0.12, -0.03, 0.55],
        };

        assert_eq!(embedding.index, 1);
        assert_eq!(embedding.embedding, vec![0.12, -0.03, 0.55]);
    }
}
