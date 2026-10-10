//! Private compiled Node Schema support. Validators are execution dependencies,
//! not Runtime state or another public input/output protocol.

use serde_json::Value;

use crate::Error;

pub(crate) fn compile(schema: &Value) -> Result<jsonschema::Validator, Error> {
    jsonschema::meta::validate(schema).map_err(|error| {
        Error::new("INVALID_SCHEMA", "node I/O schema is invalid").with_details(error.to_string())
    })?;
    // Cargo disables network/file resolution. In-document references are
    // supported, but compiling a Node must never fetch an external schema.
    jsonschema::options()
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| {
            Error::new("INVALID_SCHEMA", "node I/O schema cannot be compiled")
                .with_details(error.to_string())
        })
}

pub(crate) fn validate(
    validator: &jsonschema::Validator,
    value: &Value,
    code: &str,
    message: &str,
) -> Result<(), Error> {
    validator
        .validate(value)
        .map_err(|error| Error::new(code, message).with_details(error.to_string()))
}
