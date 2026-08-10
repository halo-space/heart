pub mod model;
pub mod tool;

pub type Cancellation = tokio_util::sync::CancellationToken;

#[cfg(test)]
mod tests {
    use super::Cancellation;

    #[test]
    fn cancellation_propagates_to_descendants_only() {
        let node_a = Cancellation::new();
        let node_c = node_a.child_token();
        let node_d = node_c.child_token();
        let sibling = node_a.child_token();

        node_c.cancel();

        assert!(!node_a.is_cancelled());
        assert!(node_c.is_cancelled());
        assert!(node_d.is_cancelled());
        assert!(!sibling.is_cancelled());
    }

    #[test]
    fn cancelling_parent_cancels_the_whole_subtree() {
        let node_a = Cancellation::new();
        let node_c = node_a.child_token();
        let node_d = node_c.child_token();

        node_a.cancel();

        assert!(node_a.is_cancelled());
        assert!(node_c.is_cancelled());
        assert!(node_d.is_cancelled());
    }
}
