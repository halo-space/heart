// Compile the attribute in the defining library too, not only downstream
// examples. This is a test-only assertion, with no node or runtime created.
#[crate::node]
async fn macro_internal_contract(
    input: crate::Messages,
    _: crate::Cancellation,
) -> Result<crate::Messages, crate::Error> {
    Ok(input)
}
