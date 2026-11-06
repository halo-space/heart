//! Adapter boundary for the replaceable DingTalk Stream SDK.

pub(crate) use dingtalk_stream::{
    AsyncChatbotHandler, ChatbotMessage, ChatbotReplier, Credential, DingTalkStreamClient,
    messages::frames::MessageBody,
};

#[cfg(test)]
pub(crate) use dingtalk_stream::messages::frames::Headers;
