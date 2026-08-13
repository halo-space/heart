use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Metadata;

use super::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Scope {
    User = 1,
    Agent = 2,
}

impl Serialize for Scope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for Scope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::try_from(value).map_err(|()| D::Error::custom("invalid document scope"))
    }
}

impl TryFrom<u8> for Scope {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::User),
            2 => Ok(Self::Agent),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Status {
    Pending = 1,
    Processing = 2,
    Ready = 3,
    Failed = 4,
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
        Self::try_from(value).map_err(|()| D::Error::custom("invalid document status"))
    }
}

impl TryFrom<u8> for Status {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Pending),
            2 => Ok(Self::Processing),
            3 => Ok(Self::Ready),
            4 => Ok(Self::Failed),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

impl Document {
    pub fn validate(&self) -> Result<(), Error> {
        if self.id == 0
            || self.version == 0
            || self.created_time == 0
            || self.updated_time < self.created_time
            || validate_fields(
                self.user_id,
                self.scope,
                &self.title,
                &self.content,
                self.ext.as_deref(),
                self.url.as_deref(),
                Some(self.status),
            )
            .is_err()
        {
            return Err(Error::InvalidInput);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Read {
    pub id: u64,
    pub tenant_id: u64,
    pub app_id: u64,
    pub agent_id: u64,
    pub user_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Update {
    pub id: u64,
    pub user_id: u64,
    pub scope: Scope,
    pub title: String,
    pub content: String,
    pub ext: Option<String>,
    pub url: Option<String>,
    pub status: Status,
    pub metadata: Metadata,
    pub version: u64,
}

impl Update {
    pub fn validate(&self) -> Result<(), Error> {
        if self.id == 0 || self.version == 0 {
            return Err(Error::InvalidInput);
        }

        validate_fields(
            self.user_id,
            self.scope,
            &self.title,
            &self.content,
            self.ext.as_deref(),
            self.url.as_deref(),
            Some(self.status),
        )
    }
}

fn validate_fields(
    user_id: u64,
    scope: Scope,
    title: &str,
    content: &str,
    ext: Option<&str>,
    url: Option<&str>,
    status: Option<Status>,
) -> Result<(), Error> {
    let valid_owner = match scope {
        Scope::User => user_id != 0,
        Scope::Agent => user_id == 0,
    };
    let has_content = !content.trim().is_empty();
    let has_url = url.is_some_and(|url| !url.trim().is_empty());
    let valid_ext = ext.is_none_or(|ext| {
        !ext.is_empty()
            && !ext.starts_with('.')
            && ext.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'.' | b'-' | b'_')
            })
    });

    if !valid_owner
        || title.trim().is_empty()
        || (!has_content && !has_url)
        || (status == Some(Status::Ready) && !has_content)
        || !valid_ext
    {
        return Err(Error::InvalidInput);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_serializes_numeric_scope_and_status() {
        let document = document(Scope::User, 4, Status::Ready, "Parsed text", None);

        let value = serde_json::to_value(&document).expect("document must serialize");

        assert_eq!(value["scope"], 1);
        assert_eq!(value["status"], 3);
        assert_eq!(document.validate(), Ok(()));
    }

    #[test]
    fn pending_document_may_only_reference_uploaded_file() {
        let document = document(
            Scope::Agent,
            0,
            Status::Pending,
            "",
            Some("storage://documents/guide.pdf"),
        );

        assert_eq!(document.validate(), Ok(()));
    }

    #[test]
    fn ready_document_requires_content() {
        let document = document(
            Scope::Agent,
            0,
            Status::Ready,
            "",
            Some("storage://documents/guide.pdf"),
        );

        assert_eq!(document.validate(), Err(Error::InvalidInput));
    }

    #[test]
    fn document_validates_scope_and_extension() {
        let mut document = document(Scope::Agent, 7, Status::Ready, "text", None);
        assert_eq!(document.validate(), Err(Error::InvalidInput));

        document.user_id = 0;
        document.ext = Some(".PDF".into());
        assert_eq!(document.validate(), Err(Error::InvalidInput));

        document.ext = Some("tar.gz".into());
        assert_eq!(document.validate(), Ok(()));
    }

    #[test]
    fn update_allows_scope_and_owner_to_change_together() {
        let input = Update {
            id: 1,
            user_id: 0,
            scope: Scope::Agent,
            title: "Shared guide".into(),
            content: "Parsed text".into(),
            ext: Some("pdf".into()),
            url: None,
            status: Status::Ready,
            metadata: Metadata::new(),
            version: 2,
        };

        assert_eq!(input.validate(), Ok(()));
    }

    fn document(
        scope: Scope,
        user_id: u64,
        status: Status,
        content: &str,
        url: Option<&str>,
    ) -> Document {
        Document {
            id: 1,
            tenant_id: 2,
            app_id: 3,
            agent_id: 4,
            user_id,
            scope,
            title: "Guide".into(),
            content: content.into(),
            ext: Some("pdf".into()),
            url: url.map(str::to_owned),
            status,
            metadata: Metadata::new(),
            version: 1,
            created_time: 1,
            updated_time: 1,
        }
    }
}
