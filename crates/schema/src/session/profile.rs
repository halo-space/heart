use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::memory::Error;

use super::Idx;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Type {
    User,
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    #[serde(flatten)]
    pub idx: Idx,
    pub update_time: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    pub id: u64,
    pub tenant_id: u64,
    pub agent_id: u64,
    pub user_id: u64,
    #[serde(rename = "type")]
    pub profile_type: Type,
    pub content: String,
    pub extra: BTreeMap<String, Progress>,
    pub version: u64,
    pub created_time: u64,
    pub updated_time: u64,
}

impl Profile {
    pub fn validate(&self) -> Result<(), Error> {
        let valid_owner = match self.profile_type {
            Type::User => self.user_id != 0,
            Type::Agent => self.user_id == 0,
        };

        if !valid_owner
            || self
                .extra
                .values()
                .any(|progress| progress.idx.validate().is_err())
        {
            return Err(Error::InvalidInput);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_extra_is_keyed_by_session_id() {
        let profile = Profile {
            id: 1,
            tenant_id: 2,
            agent_id: 3,
            user_id: 4,
            profile_type: Type::User,
            content: "The user prefers quiet hotels.".into(),
            extra: BTreeMap::from([(
                "session_1".into(),
                Progress {
                    idx: Idx { from: 1, to: 10 },
                    update_time: 1_786_501_800_000,
                },
            )]),
            version: 2,
            created_time: 1_786_501_700_000,
            updated_time: 1_786_501_800_000,
        };

        let value = serde_json::to_value(profile).expect("profile must serialize");

        assert_eq!(value["type"], "user");
        assert_eq!(value["extra"]["session_1"]["from"], 1);
        assert_eq!(value["extra"]["session_1"]["to"], 10);
        assert_eq!(
            value["extra"]["session_1"]["update_time"],
            1_786_501_800_000_u64
        );

        let profile = serde_json::from_value::<Profile>(value).expect("profile must deserialize");
        assert_eq!(profile.validate(), Ok(()));
    }

    #[test]
    fn profile_validates_owner_type() {
        let mut profile = profile(Type::Agent, 1);
        assert_eq!(profile.validate(), Err(Error::InvalidInput));

        profile.user_id = 0;
        assert_eq!(profile.validate(), Ok(()));

        profile.profile_type = Type::User;
        assert_eq!(profile.validate(), Err(Error::InvalidInput));
    }

    fn profile(profile_type: Type, user_id: u64) -> Profile {
        Profile {
            id: 1,
            tenant_id: 2,
            agent_id: 3,
            user_id,
            profile_type,
            content: String::new(),
            extra: BTreeMap::new(),
            version: 0,
            created_time: 1,
            updated_time: 1,
        }
    }
}
