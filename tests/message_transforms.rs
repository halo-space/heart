//! Custom compression contracts do not require or fabricate a model result.
//! Concrete model-backed compression and reasoning are exercised by live
//! tests in `component_real_acceptance`.

use futures_executor::block_on;
use halo::agent::compression::Compressor;
use halo::agent::message::Role;
use halo::agent::{Cancellation, Error, Message, Messages};
use serde_json::json;

fn message(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message
        .content
        .push(halo::agent::model::content::part::Part {
            r#type: "text".into(),
            data: serde_json::from_value(json!({"value":value})).unwrap(),
        });
    message
}

#[test]
fn custom_compressors_share_size_contract_and_return_their_input() {
    struct Custom {
        size: Option<u64>,
    }
    impl Compressor for Custom {
        fn size(&self) -> Option<u64> {
            self.size
        }

        async fn compress(&self, input: Messages, _: &Cancellation) -> Result<Messages, Error> {
            Ok(input)
        }
    }

    let compressor = Custom { size: Some(10_000) };
    assert_eq!(compressor.size(), Some(10_000));
    let input = Messages::new([message(Role::User, "input")]);
    assert_eq!(
        block_on(compressor.compress(input.clone(), &Cancellation::new())).unwrap(),
        input
    );

    struct Standalone;
    impl Compressor for Standalone {
        async fn compress(&self, input: Messages, _: &Cancellation) -> Result<Messages, Error> {
            Ok(input)
        }
    }
    assert_eq!(Standalone.size(), None);
    assert_eq!(
        block_on(Standalone.compress(input.clone(), &Cancellation::new())).unwrap(),
        input
    );
}
