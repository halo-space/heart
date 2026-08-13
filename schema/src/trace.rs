use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::{Metadata, memory::Error, message::Message};

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
        Self::try_from(value).map_err(|()| D::Error::custom("invalid trace status"))
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
pub struct Trace {
    pub id: u64,
    pub tenant_id: u64,
    pub user_id: u64,
    pub agent_id: u64,
    pub session_id: u64,
    pub chat_id: u64,
    pub idx: u64,
    pub attempt: u32,
    pub key: String,
    pub status: Status,
    pub input: Value,
    pub message: Option<Message>,
    pub error: Option<Value>,
    pub metadata: Metadata,
    pub duration: u64,
    pub created_time: u64,
    pub updated_time: u64,
}

impl Trace {
    pub fn validate(&self) -> Result<(), Error> {
        let valid_message = self
            .message
            .as_ref()
            .is_none_or(|message| matches!(message, Message::Tool { .. }));

        if self.chat_id == 0
            || self.idx == 0
            || self.attempt == 0
            || self.key.is_empty()
            || !valid_message
        {
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
    fn trace_serializes_tool_result_and_retry_identity() {
        let trace = Trace {
            id: 2,
            tenant_id: 1,
            user_id: 3,
            agent_id: 4,
            session_id: 5,
            chat_id: 6,
            idx: 1,
            attempt: 2,
            key: "weather".into(),
            status: Status::Completed,
            input: json!({ "city": "天津" }),
            message: Some(Message::Tool {
                content: "{\"temperature\":31}".into(),
                execute_id: "execute_1".into(),
            }),
            error: None,
            metadata: Metadata::new(),
            duration: 640,
            created_time: 1_786_501_800_900,
            updated_time: 1_786_501_801_540,
        };

        let value = serde_json::to_value(&trace).expect("trace must serialize");

        assert_eq!(value["idx"], 1);
        assert_eq!(value["attempt"], 2);
        assert_eq!(value["key"], "weather");
        assert_eq!(value["status"], 2);
        assert_eq!(value["message"]["role"], "tool");
        assert!(value.get("usage").is_none());

        assert_eq!(trace.validate(), Ok(()));
    }

    #[test]
    fn trace_rejects_invalid_identity_and_message() {
        let mut trace = valid_trace();

        trace.attempt = 0;
        assert_eq!(trace.validate(), Err(Error::InvalidInput));

        trace = valid_trace();
        trace.message = Some(Message::Assistant {
            content: Some("not a tool result".into()),
            audio: None,
            embeddings: Vec::new(),
            name: None,
            refusal: None,
            executes: Vec::new(),
        });
        assert_eq!(trace.validate(), Err(Error::InvalidInput));
    }

    fn valid_trace() -> Trace {
        Trace {
            id: 1,
            tenant_id: 1,
            user_id: 2,
            agent_id: 3,
            session_id: 4,
            chat_id: 5,
            idx: 1,
            attempt: 1,
            key: "weather".into(),
            status: Status::Running,
            input: json!({}),
            message: None,
            error: None,
            metadata: Metadata::new(),
            duration: 0,
            created_time: 1,
            updated_time: 1,
        }
    }
}
