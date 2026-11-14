//! Private field adapter shared by execution paths. This module does not
//! define an input protocol, own state, infer bindings or merge Messages.
use serde_json::{Map, Value};

use crate::{Error, Messages};

/// Each tuple is (target field, actual source node ID, literal source field).
/// Callers select successful executions before handing their Messages here.
pub(crate) fn fields<'a>(
    bindings: impl IntoIterator<Item = (&'a str, &'a str, &'a str)>,
    mut source: impl FnMut(&str) -> Result<&'a Messages, Error>,
) -> Result<Map<String, Value>, Error> {
    let mut values = Map::new();
    for (target, node_id, field) in bindings {
        if target.is_empty() || node_id.is_empty() || field.is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "input binding contains an empty field or node ID",
            ));
        }
        if values.contains_key(target) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                format!("duplicate input field {target}"),
            ));
        }
        let value = source(node_id)?
            .as_slice()
            .iter()
            .rev()
            .find_map(|message| message.values.get(field))
            .cloned()
            .ok_or_else(|| {
                Error::new(
                    "INVALID_ARGUMENTS",
                    format!("missing field {field} from node {node_id}"),
                )
            })?;
        values.insert(target.to_owned(), value);
    }
    Ok(values)
}
