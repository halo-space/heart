use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Metadata, memory::Error, model::Usage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Running,
    Completed,
    Failed,
    Partial,
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chat {
    pub id: u64,
    pub tenant_id: u64,
    pub app_id: u64,
    pub agent_id: Option<u64>,
    pub team_id: Option<u64>,
    pub user_id: u64,
    pub session_id: u64,
    pub idx: u64,
    pub ref_id: Option<u64>,
    pub trust: i8,
    pub feedback: i8,
    pub models: Metadata,
    pub trace_id: Option<String>,
    pub status: Status,
    pub input: Value,
    pub output: Option<Value>,
    pub usage: Option<Usage>,
    pub metadata: Metadata,
    pub created_time: u64,
    pub updated_time: u64,
}

impl Chat {
    pub fn validate(&self) -> Result<(), Error> {
        let has_one_target = self.agent_id.is_some() ^ self.team_id.is_some();
        let valid_trust = (-1..=1).contains(&self.trust);
        let valid_feedback = (-1..=1).contains(&self.feedback);

        if self.idx == 0 || !has_one_target || !valid_trust || !valid_feedback {
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
    fn chat_keeps_input_and_final_output_separate() {
        let chat = Chat {
            id: 10,
            tenant_id: 1,
            app_id: 2,
            agent_id: Some(3),
            team_id: None,
            user_id: 4,
            session_id: 5,
            idx: 2,
            ref_id: Some(9),
            trust: -1,
            feedback: 0,
            models: Metadata::new(),
            trace_id: Some("trace_1".into()),
            status: Status::Completed,
            input: json!({ "text": "Regenerate" }),
            output: Some(json!({ "text": "New result" })),
            usage: None,
            metadata: Metadata::new(),
            created_time: 1_786_501_800_000,
            updated_time: 1_786_501_900_000,
        };

        let value = serde_json::to_value(chat).expect("chat must serialize");

        assert_eq!(value["idx"], 2);
        assert_eq!(value["ref_id"], 9);
        assert_eq!(value["trust"], -1);
        assert_eq!(value["feedback"], 0);
        assert_eq!(value["trace_id"], "trace_1");
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output"]["text"], "New result");

        let chat = serde_json::from_value::<Chat>(value).expect("chat must deserialize");
        assert_eq!(chat.validate(), Ok(()));
    }

    #[test]
    fn chat_requires_exactly_one_target() {
        let mut chat = valid_chat();
        chat.team_id = Some(4);

        assert_eq!(chat.validate(), Err(Error::InvalidInput));

        chat.agent_id = None;
        assert_eq!(chat.validate(), Ok(()));

        chat.team_id = None;
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
            agent_id: Some(3),
            team_id: None,
            user_id: 4,
            session_id: 5,
            idx: 1,
            ref_id: None,
            trust: 0,
            feedback: 0,
            models: Metadata::new(),
            trace_id: None,
            status: Status::Running,
            input: json!({}),
            output: None,
            usage: None,
            metadata: Metadata::new(),
            created_time: 1,
            updated_time: 1,
        }
    }
}
