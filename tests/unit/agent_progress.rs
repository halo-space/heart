use super::*;

struct Model;
impl Chat for Model {
    type Stream = futures_util::stream::Empty<Result<Event, Error>>;

    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        Ok(futures_util::stream::empty())
    }
}

#[test]
fn event_progress_does_not_inline_the_complete_execution_machine() {
    fn bounded<F: Future>(_: impl FnOnce(ReActState<Model>) -> F) {
        assert!(
            std::mem::size_of::<F>() < 16 * 1024,
            "event progress Future: {} bytes",
            std::mem::size_of::<F>()
        );
    }
    bounded(next_react_progress::<Model>);
    bounded(next_react_event::<Model>);
}

#[tokio::test]
async fn ready_cancel_handoff_ignores_only_the_tokio_budget_yield() {
    std::future::poll_fn(|cx| {
        while tokio::task::coop::has_budget_remaining() {
            let (send, receive) = tokio::sync::oneshot::channel();
            send.send(()).unwrap();
            assert!(matches!(
                std::pin::pin!(receive).poll(cx),
                Poll::Ready(Ok(()))
            ));
        }
        let relay = crate::core::agent::subagent::Relay::default();
        let (send, receive) = tokio::sync::oneshot::channel();
        send.send(7_u64).unwrap();
        let mut receive = std::pin::pin!(receive);
        let mut consumed = false;
        let (ready, events) = crate::core::agent::subagent::cancel_ready(
            cx,
            &relay,
            |cx| {
                if consumed {
                    return Poll::Ready(None);
                }
                match receive.as_mut().poll(cx) {
                    Poll::Ready(Ok(value)) => {
                        consumed = true;
                        Poll::Ready(Some(value))
                    }
                    Poll::Ready(Err(error)) => panic!("{error}"),
                    Poll::Pending => Poll::Pending,
                }
            },
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(ready, [7]);
        assert!(events.is_empty());
        assert!(
            !tokio::task::coop::has_budget_remaining(),
            "original budget is restored"
        );
        let mut polls = 0;
        let (ready, _) = crate::core::agent::subagent::cancel_ready::<u64>(
            cx,
            &relay,
            |_| {
                polls += 1;
                assert_eq!(polls, 1);
                Poll::Pending
            },
            |_| Ok(()),
        )
        .unwrap();
        assert!(ready.is_empty());
        Poll::Ready(())
    })
    .await;
}

#[test]
fn plan_progress_keeps_the_owned_task_off_nested_poll_stacks() {
    assert!(std::mem::size_of::<execution::TaskProgress<Model>>() < 1024);
}
