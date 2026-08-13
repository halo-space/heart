pub mod long;
pub mod short;

pub trait Memory: Send + Sync {
    fn short(&self) -> &dyn short::Short;

    fn long(&self) -> &dyn long::Long;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_contract_is_object_safe() {
        fn accepts_memory(_: &dyn Memory) {}
        fn accepts_short(_: &dyn short::Short) {}
        fn accepts_chats(_: &dyn short::Chats) {}
        fn accepts_sessions(_: &dyn short::Sessions) {}
        fn accepts_traces(_: &dyn short::Traces) {}
        fn accepts_long(_: &dyn long::Long) {}
        fn accepts_profiles(_: &dyn long::Profiles) {}

        let _ = accepts_memory;
        let _ = accepts_short;
        let _ = accepts_chats;
        let _ = accepts_sessions;
        let _ = accepts_traces;
        let _ = accepts_long;
        let _ = accepts_profiles;
    }
}
