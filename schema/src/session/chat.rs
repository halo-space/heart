use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::{Metadata, memory::Error, message::Message, model::Usage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Status {
    Running = 1,
    Completed = 2,
    Failed = 3,
    Stopped = 4,
}

impl Serialize for Status {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for Status {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::try_from(value).map_err(|()| D::Error::custom("invalid chat status"))
    }
}

impl TryFrom<u8> for Status {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Running),
            2 => Ok(Self::Completed),
            3 => Ok(Self::Failed),
            4 => Ok(Self::Stopped),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chat {
    pub id: u64,
    pub tenant_id: u64,
    pub app_id: u64,
    pub agent_id: u64,
    pub user_id: u64,
    pub session_id: u64,
    pub idx: u64,
    pub ref_id: Option<u64>,
    pub trust: i8,
    pub feedback: i8,
    pub models: Metadata,
    pub status: Status,
    pub input: Value,
    pub message: Option<Message>,
    pub usage: Option<Usage>,
    pub metadata: Metadata,
    pub created_time: u64,
    pub updated_time: u64,
}

impl Chat {
    pub fn validate(&self) -> Result<(), Error> {
        let valid_trust = (-1..=1).contains(&self.trust);
        let valid_feedback = (-1..=1).contains(&self.feedback);
        let valid_message = self
            .message
            .as_ref()
            .is_none_or(|message| matches!(message, Message::Assistant { .. }));

        if self.idx == 0 || !valid_trust || !valid_feedback || !valid_message {
            return Err(Error::InvalidInput);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn chat_keeps_input_and_final_message_separate() {
        let chat = Chat {
            id: 10,
            tenant_id: 1,
            app_id: 2,
            agent_id: 3,
            user_id: 4,
            session_id: 5,
            idx: 2,
            ref_id: Some(9),
            trust: -1,
            feedback: 0,
            models: Metadata::new(),
            status: Status::Completed,
            input: json!({ "text": "Regenerate" }),
            message: Some(Message::Assistant {
                content: Some("New result".into()),
                audio: None,
                embeddings: Vec::new(),
                name: None,
                refusal: None,
                executes: Vec::new(),
            }),
            usage: Some(Usage {
                input: Some(10),
                cached: None,
                output: Some(4),
                reasoning: None,
                total: Some(14),
                duration: None,
            }),
            metadata: Metadata::new(),
            created_time: 1_786_501_800_000,
            updated_time: 1_786_501_900_000,
        };

        let value = serde_json::to_value(chat).expect("chat must serialize");

        assert_eq!(value["idx"], 2);
        assert_eq!(value["ref_id"], 9);
        assert_eq!(value["trust"], -1);
        assert_eq!(value["feedback"], 0);
        assert_eq!(value["status"], 2);
        assert_eq!(value["message"]["role"], "assistant");
        assert_eq!(value["message"]["content"], "New result");
        assert_eq!(value["usage"]["total"], 14);

        let chat = serde_json::from_value::<Chat>(value).expect("chat must deserialize");
        assert_eq!(chat.validate(), Ok(()));
    }

    #[test]
    fn chat_rejects_non_assistant_message() {
        let mut chat = valid_chat();
        chat.message = Some(Message::Tool {
            content: "tool result".into(),
            execute_id: "execute_1".into(),
        });

        assert_eq!(chat.validate(), Err(Error::InvalidInput));
    }

    #[test]
    fn chat_rejects_invalid_trust_and_feedback() {
        let mut chat = valid_chat();
        chat.trust = 2;
        assert_eq!(chat.validate(), Err(Error::InvalidInput));

        chat.trust = 0;
        chat.feedback = -2;
        assert_eq!(chat.validate(), Err(Error::InvalidInput));
    }

    fn valid_chat() -> Chat {
        Chat {
            id: 10,
            tenant_id: 1,
            app_id: 2,
            agent_id: 3,
            user_id: 4,
            session_id: 5,
            idx: 1,
            ref_id: None,
            trust: 0,
            feedback: 0,
            models: Metadata::new(),
            status: Status::Running,
            input: json!({}),
            message: None,
            usage: None,
            metadata: Metadata::new(),
            created_time: 1,
            updated_time: 1,
        }
    }
}
