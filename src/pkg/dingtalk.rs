//! Adapter boundary for the replaceable DingTalk Stream SDK.

pub(crate) use dingtalk_stream::{
    AsyncChatbotHandler, ChatbotMessage, ChatbotReplier, Credential, DingTalkStreamClient,
    messages::frames::MessageBody,
};
