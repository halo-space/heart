//! Heart MCP implementation backed by the replaceable `pkg::mcp` adapter.

use serde_json::{Map, Value, json};

use crate::{
    Cancellation, Error, Message,
    mcp::{Definition, Server as Contract},
    message::Role,
    model::content::part::{Part, object as part_data},
    pkg,
    tool::{Call, ToolDefinition},
};

/// MCP Server implementation. Transport and SDK details are isolated in
/// `pkg::mcp`; this type only maps the adapter to Heart's public contract.
pub struct Server {
    definition: Definition,
    client: pkg::mcp::Client,
    tools: Vec<ToolDefinition>,
}

impl Server {
    pub async fn http(
        definition: Definition,
        endpoint: &str,
        headers: Map<String, Value>,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        validate_definition(&definition)?;
        let client = pkg::mcp::Client::http(endpoint, headers, cancellation).await?;
        let tools = discover_tools(&client, cancellation).await?;
        Ok(Self {
            definition,
            client,
            tools,
        })
    }

    pub async fn stdio(
        definition: Definition,
        command: tokio::process::Command,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        validate_definition(&definition)?;
        let client = pkg::mcp::Client::stdio(command, cancellation).await?;
        let tools = discover_tools(&client, cancellation).await?;
        Ok(Self {
            definition,
            client,
            tools,
        })
    }

    pub async fn close(self) -> Result<(), Error> {
        self.client.close().await
    }
}

impl Contract for Server {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    async fn call(&self, call: Call, cancellation: &Cancellation) -> Result<Message, Error> {
        if call.call_id.trim().is_empty() || call.name.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "MCP call id and name are required",
            ));
        }
        let result = self
            .client
            .call(&call.name, call.arguments, cancellation)
            .await?;
        if result.is_error {
            if let Some(value) = &result.structured_content
                && let Ok(error) = serde_json::from_value::<Error>(value.clone())
            {
                return Err(error);
            }
            let text = result
                .content
                .iter()
                .filter_map(|part| match part {
                    pkg::mcp::Content::Text(text) => Some(text.as_str()),
                    pkg::mcp::Content::Other => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            return Err(Error::new(
                "TOOL_ERROR",
                if text.trim().is_empty() {
                    "MCP tool execution failed".to_owned()
                } else {
                    text
                },
            ));
        }
        let mut message = Message::new(Role::Tool);
        message
            .metadata
            .insert("call_id".into(), json!(call.call_id));
        message.metadata.insert("name".into(), json!(call.name));
        if let Some(value) = result.structured_content {
            message.values = match value {
                Value::Object(values) => values,
                value => Map::from_iter([("value".into(), value)]),
            };
        }
        for part in result.content {
            if let pkg::mcp::Content::Text(text) = part {
                message.content.push(Part {
                    r#type: "text".into(),
                    data: part_data(json!({"value": text})),
                });
            } else {
                return Err(Error::new(
                    "UNSUPPORTED",
                    "MCP non-text tool content requires the V3 media contract",
                ));
            }
        }
        Ok(message)
    }
}

async fn discover_tools(
    client: &pkg::mcp::Client,
    cancellation: &Cancellation,
) -> Result<Vec<ToolDefinition>, Error> {
    client
        .list_tools(cancellation)
        .await?
        .into_iter()
        .map(|tool| {
            let name = tool.name;
            let definition = ToolDefinition {
                description: tool
                    .description
                    .filter(|text| !text.trim().is_empty())
                    .unwrap_or_else(|| name.clone()),
                name,
                parameters: tool.input_schema,
            };
            definition
                .validate()
                .map_err(|_| Error::new("INVALID_RESPONSE", "invalid MCP tool schema"))?;
            Ok(definition)
        })
        .collect()
}

fn validate_definition(definition: &Definition) -> Result<(), Error> {
    if definition.name.trim().is_empty() {
        Err(Error::new(
            "INVALID_ARGUMENTS",
            "MCP server name is required",
        ))
    } else {
        Ok(())
    }
}
