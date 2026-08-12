use serde::{Deserialize, Serialize};

use crate::{Metadata, memory::Error};

pub mod chat;
pub mod profile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Idx {
    pub from: u64,
    pub to: u64,
}

impl Idx {
    pub const fn validate(&self) -> Result<(), Error> {
        if self.from == 0 || self.from > self.to {
            return Err(Error::InvalidInput);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Extra {
    pub idx: Option<Idx>,
    #[serde(flatten)]
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Active,
    Archived,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub id: u64,
    pub tenant_id: u64,
    pub app_id: u64,
    pub agent_id: u64,
    pub user_id: u64,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub status: Status,
    pub extra: Extra,
    pub created_time: u64,
    pub updated_time: u64,
}

impl Session {
    pub fn validate(&self) -> Result<(), Error> {
        match (&self.summary, self.extra.idx) {
            (Some(summary), Some(idx)) if !summary.is_empty() => idx.validate(),
            (None, None) => Ok(()),
            _ => Err(Error::InvalidInput),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn session_serializes_summary_range_inside_extra() {
        let session = Session {
            id: 1,
            tenant_id: 2,
            app_id: 3,
            agent_id: 4,
            user_id: 5,
            title: Some("Tianjin trip".into()),
            summary: Some("The user is planning a trip to Tianjin.".into()),
            status: Status::Active,
            extra: Extra {
                idx: Some(Idx { from: 1, to: 8 }),
                metadata: Metadata::new(),
            },
            created_time: 1_786_501_800_000,
            updated_time: 1_786_501_900_000,
        };

        let value = serde_json::to_value(session).expect("session must serialize");

        assert_eq!(value["status"], "active");
        assert_eq!(value["extra"]["idx"], json!({ "from": 1, "to": 8 }));

        let session = serde_json::from_value::<Session>(value).expect("session must deserialize");
        assert_eq!(session.validate(), Ok(()));
    }

    #[test]
    fn session_requires_summary_and_idx_together() {
        let session = Session {
            id: 1,
            tenant_id: 2,
            app_id: 3,
            agent_id: 4,
            user_id: 5,
            title: None,
            summary: Some("summary".into()),
            status: Status::Active,
            extra: Extra::default(),
            created_time: 1,
            updated_time: 1,
        };

        assert_eq!(session.validate(), Err(Error::InvalidInput));
    }

    #[test]
    fn idx_rejects_zero_and_reversed_ranges() {
        assert_eq!(Idx { from: 0, to: 1 }.validate(), Err(Error::InvalidInput));
        assert_eq!(Idx { from: 2, to: 1 }.validate(), Err(Error::InvalidInput));
    }
}
