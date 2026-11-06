//! Adapter boundary for the replaceable Feishu/Lark Rust SDK.

pub(crate) use larksuite_oapi_sdk_rs::LarkError;
pub(crate) use larksuite_oapi_sdk_rs::{
    EventDispatcher, LarkClient, RequestOption,
    events::im::{MessageSender, P2MessageReceiveV1},
    ws::WsClient,
};

#[cfg(test)]
pub(crate) use larksuite_oapi_sdk_rs::events::im::Message;
