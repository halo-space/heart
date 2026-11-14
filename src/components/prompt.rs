use std::collections::BTreeMap;

use serde_json::Value;

use crate::components::message::{Message, Messages};

pub type Error = crate::Error;

#[derive(Clone, Debug)]
enum Item {
    Message(Box<Message>),
    Messages(String),
}

#[derive(Clone, Debug, Default)]
pub struct Template {
    items: Vec<Item>,
}

#[derive(Clone, Debug, Default)]
pub struct Variables {
    values: BTreeMap<String, Value>,
    messages: BTreeMap<String, Messages>,
}

impl Template {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn message(mut self, message: Message) -> Self {
        self.items.push(Item::Message(Box::new(message)));
        self
    }

    pub fn messages(mut self, name: impl Into<String>) -> Self {
        self.items.push(Item::Messages(name.into()));
        self
    }

    pub fn format(&self, variables: &Variables) -> Result<Messages, Error> {
        let mut output = Messages::default();
        for item in &self.items {
            match item {
                Item::Message(message) => output.push(render_message(message, variables)?),
                Item::Messages(name) => {
                    let history = variables.messages.get(name).ok_or_else(|| {
                        Error::new(
                            "INVALID_ARGUMENTS",
                            format!("missing messages variable: {name}"),
                        )
                    })?;
                    output.0.extend(history.0.iter().cloned());
                }
            }
        }
        Ok(output)
    }
}

impl Variables {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn value(mut self, name: impl Into<String>, value: Value) -> Self {
        self.values.insert(name.into(), value);
        self
    }

    pub fn messages(mut self, name: impl Into<String>, messages: Messages) -> Self {
        self.messages.insert(name.into(), messages);
        self
    }
}

fn render_message(message: &Message, variables: &Variables) -> Result<Message, Error> {
    let mut message = message.clone();
    for part in &mut message.content {
        if let Some(value) = part.data.get_mut("value") {
            if let Some(text) = value.as_str() {
                *value = Value::String(render_text(text, variables)?);
            } else if value.is_number() || value.is_boolean() {
                *value = Value::String(render_text(&value.to_string(), variables)?);
            } else {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "prompt part value must be a string, number, or boolean",
                ));
            }
        }
    }
    Ok(message)
}

fn render_text(text: &str, variables: &Variables) -> Result<String, Error> {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '{' if index + 1 < chars.len() && chars[index + 1] == '{' => {
                output.push('{');
                index += 2;
            }
            '}' if index + 1 < chars.len() && chars[index + 1] == '}' => {
                output.push('}');
                index += 2;
            }
            '{' => {
                let start = index + 1;
                let Some(relative_end) = chars[start..].iter().position(|char| *char == '}') else {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "unclosed prompt placeholder",
                    ));
                };
                let end = start + relative_end;
                let name: String = chars[start..end].iter().collect();
                if is_variable_name(&name) {
                    let value = variables.values.get(&name).ok_or_else(|| {
                        Error::new(
                            "INVALID_ARGUMENTS",
                            format!("missing prompt variable: {name}"),
                        )
                    })?;
                    output.push_str(&value_to_text(value)?);
                } else {
                    output.extend(chars[index..=end].iter());
                }
                index = end + 1;
            }
            '}' => {
                return Err(Error::new("INVALID_ARGUMENTS", "unmatched prompt brace"));
            }
            character => {
                output.push(character);
                index += 1;
            }
        }
    }
    Ok(output)
}

fn is_variable_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first == '_' || first.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn value_to_text(value: &Value) -> Result<String, Error> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Null | Value::Object(_) | Value::Array(_) => Err(Error::new(
            "INVALID_ARGUMENTS",
            "prompt variable must be a string, number, or boolean",
        )),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_components_prompt_tests.rs"]
mod tests;
