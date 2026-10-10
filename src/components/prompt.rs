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
        if let Some(object) = part.data.as_object_mut()
            && let Some(value) = object.get_mut("value")
        {
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
mod tests {
    use super::*;
    use crate::components::message::Role;
    use crate::components::model::content::Part;
    use serde_json::Map;
    use serde_json::json;

    fn message(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![Part {
                r#type: "text".into(),
                data: json!({"value": text}),
            }],
            values: Map::new(),
            metadata: Map::new(),
            usage: None,
        }
    }

    #[test]
    fn renders_text_and_expands_messages_in_order() {
        let template = Template::new()
            .message(message("问题：{question}"))
            .messages("history");
        let history = Messages::new([message("历史")]);
        let result = template
            .format(
                &Variables::new()
                    .value("question", json!("天气"))
                    .messages("history", history),
            )
            .unwrap();
        assert_eq!(result.0.len(), 2);
        assert_eq!(result.0[0].content[0].data["value"], "问题：天气");
        assert_eq!(result.0[1].content[0].data["value"], "历史");
    }

    #[test]
    fn rejects_missing_and_non_scalar_variables_without_mutating_template() {
        let template = Template::new().message(message("{question}"));
        assert_eq!(
            template.format(&Variables::new()).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
        let variables = Variables::new().value("question", json!({"text": "天气"}));
        assert_eq!(
            template.format(&variables).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
        let template = Template::new().message(Message {
            role: Role::User,
            content: vec![Part {
                r#type: "text".into(),
                data: json!({"value": ["not", "text"]}),
            }],
            values: Map::new(),
            metadata: Map::new(),
            usage: None,
        });
        assert_eq!(
            template.format(&Variables::new()).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }

    #[test]
    fn supports_escaped_braces_and_repeated_variables() {
        let template = Template::new().message(message("{{x}} {name} {name}"));
        let result = template
            .format(&Variables::new().value("name", json!("A")))
            .unwrap();
        assert_eq!(result.0[0].content[0].data["value"], "{x} A A");
    }
}
