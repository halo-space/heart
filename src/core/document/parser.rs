use std::collections::HashSet;

use pulldown_cmark::{Event, Options as MarkdownOptions, Tag};
use serde_json::{Map, Value, json};

use super::check_cancelled;
use crate::{
    Cancellation,
    document::{self, Document, Error},
};

#[derive(Clone, Copy, Debug)]
enum Method {
    General,
    One,
    Qa,
    Table,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Text,
    Markdown,
    Csv,
    Json,
}

/// A concrete Parser. Configuration is kept private, not a second public Config.
#[derive(Clone, Debug)]
pub struct Parser {
    method: Method,
    format: Format,
    options: Map<String, Value>,
}

pub fn new(method: &str, config: Value) -> Result<Parser, Error> {
    let config = config
        .as_object()
        .ok_or_else(|| invalid("config", "parser config must be an object"))?;
    allowed_keys(config, &["format", "engine", "options"])?;
    let method = match method {
        "general" | "naive" => Method::General,
        "one" => Method::One,
        "qa" => Method::Qa,
        "table" => Method::Table,
        _ => {
            return Err(Error::new(
                "UNSUPPORTED",
                "parser method is not implemented",
            ));
        }
    };
    let format = match field(config, "format")? {
        "text" | "txt" => Format::Text,
        "markdown" | "md" => Format::Markdown,
        "csv" => Format::Csv,
        "json" => Format::Json,
        _ => {
            return Err(Error::new(
                "UNSUPPORTED",
                "parser format is not implemented",
            ));
        }
    };
    let engine = field(config, "engine")?;
    let compatible = matches!(
        (format, engine),
        (Format::Text, "text")
            | (Format::Markdown, "markdown")
            | (Format::Csv, "csv")
            | (Format::Json, "json")
    );
    if !compatible {
        return Err(Error::new(
            "UNSUPPORTED",
            "parser format and engine are incompatible",
        ));
    }
    if matches!(method, Method::Qa | Method::Table) && !matches!(format, Format::Csv | Format::Json)
    {
        return Err(Error::new(
            "UNSUPPORTED",
            "this domain parser requires CSV or JSON",
        ));
    }
    let options = match config.get("options") {
        None => Map::new(),
        Some(Value::Object(options)) => options.clone(),
        _ => return Err(invalid("options", "parser options must be an object")),
    };
    let mut allowed = vec!["trim", "normalize_newlines", "remove_empty"];
    if format == Format::Csv {
        allowed.push("delimiter");
    }
    if matches!(method, Method::Qa) {
        allowed.extend(["question", "answer"]);
    }
    if format == Format::Json && matches!(method, Method::General) {
        allowed.push("content");
    }
    allowed_keys(&options, &allowed)?;
    for key in ["trim", "normalize_newlines", "remove_empty"] {
        if options.get(key).is_some_and(|value| !value.is_boolean()) {
            return Err(invalid(key, "option must be boolean"));
        }
    }
    for key in ["question", "answer", "content"] {
        if options
            .get(key)
            .is_some_and(|value| value.as_str().is_none_or(|value| value.trim().is_empty()))
        {
            return Err(invalid(key, "field name must be nonempty text"));
        }
    }
    if let Some(delimiter) = options.get("delimiter") {
        let delimiter = delimiter
            .as_str()
            .ok_or_else(|| invalid("delimiter", "delimiter must be text"))?;
        if delimiter.len() != 1 || matches!(delimiter.as_bytes()[0], b'"' | b'\r' | b'\n' | 0) {
            return Err(invalid(
                "delimiter",
                "CSV delimiter must be one non-quote, non-newline ASCII byte",
            ));
        }
    }
    if matches!(method, Method::Qa)
        && string_option(&options, "question", "question")
            == string_option(&options, "answer", "answer")
    {
        return Err(invalid(
            "question",
            "question and answer fields must differ",
        ));
    }
    Ok(Parser {
        method,
        format,
        options,
    })
}

fn invalid(field: &str, message: &str) -> Error {
    Error::new("INVALID_ARGUMENTS", message).with_details(field)
}
fn parse_error(field: &str, message: &str) -> Error {
    Error::new("PARSE_ERROR", message).with_details(field)
}
fn field<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a str, Error> {
    object
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid(name, "parser field must be nonempty text"))
}
fn string_option<'a>(options: &'a Map<String, Value>, key: &str, default: &'a str) -> &'a str {
    options.get(key).and_then(Value::as_str).unwrap_or(default)
}
fn allowed_keys(options: &Map<String, Value>, allowed: &[&str]) -> Result<(), Error> {
    if let Some(key) = options.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(key, "unknown parser configuration field"));
    }
    Ok(())
}

impl Parser {
    fn enabled(&self, key: &str, default: bool) -> bool {
        self.options
            .get(key)
            .and_then(Value::as_bool)
            .unwrap_or(default)
    }
    fn clean(&self, text: &str) -> String {
        let text = if self.enabled("trim", false) {
            text.trim()
        } else {
            text
        };
        if self.enabled("normalize_newlines", false) {
            text.replace("\r\n", "\n").replace('\r', "\n")
        } else {
            text.to_owned()
        }
    }
    fn document(&self, content: &str, role: Option<&str>) -> Document {
        let mut metadata = Map::from_iter([("method".into(), json!("text"))]);
        if let Some(role) = role {
            metadata.insert("layout".into(), json!({"role":role}));
        }
        Document {
            content: self.clean(content),
            metadata,
            ..Document::default()
        }
    }
    fn qa(&self, question: &str, answer: &str) -> Document {
        let question = self.clean(question);
        let answer = self.clean(answer);
        let mut document = self.document(&format!("{question}\n{answer}"), None);
        document
            .metadata
            .insert("qa".into(), json!({"question":question,"answer":answer}));
        document
    }

    fn text(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        // Paragraph recognition is parsing, not length/token chunking. Original
        // separators inside a paragraph are retained unless explicitly normalized.
        let mut result = Vec::new();
        let mut start = None;
        let mut offset = 0;
        let mut paragraph = 0u64;
        for line in text.split_inclusive('\n') {
            check_cancelled(cancellation)?;
            if line.trim().is_empty() {
                if let Some(start) = start.take() {
                    paragraph += 1;
                    let mut document = self.document(&text[start..offset], Some("paragraph"));
                    document.metadata.insert(
                        "location".into(),
                        json!({"paragraph":{"from":paragraph,"to":paragraph}}),
                    );
                    result.push(document);
                }
            } else {
                start.get_or_insert(offset);
            }
            offset += line.len();
        }
        if let Some(start) = start {
            paragraph += 1;
            let mut document = self.document(&text[start..], Some("paragraph"));
            document.metadata.insert(
                "location".into(),
                json!({"paragraph":{"from":paragraph,"to":paragraph}}),
            );
            result.push(document);
        }
        Ok(result)
    }

    fn markdown(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        let mut result = Vec::new();
        let mut depth = 0usize;
        let mut start = 0;
        let mut role = "paragraph";
        let mut options = MarkdownOptions::empty();
        options.insert(
            MarkdownOptions::ENABLE_TABLES
                | MarkdownOptions::ENABLE_STRIKETHROUGH
                | MarkdownOptions::ENABLE_TASKLISTS,
        );
        for (event, range) in pulldown_cmark::Parser::new_ext(text, options).into_offset_iter() {
            check_cancelled(cancellation)?;
            match event {
                Event::Start(tag) => {
                    if depth == 0 {
                        start = range.start;
                        role = match tag {
                            Tag::Heading { .. } => "heading",
                            Tag::CodeBlock(_) => "code",
                            Tag::List(_) => "list",
                            Tag::Table(_) => "table",
                            _ => "paragraph",
                        };
                    }
                    depth += 1;
                }
                Event::End(_) => {
                    depth -= 1;
                    if depth == 0 {
                        result.push(self.document(&text[start..range.end], Some(role)));
                    }
                }
                Event::Rule if depth == 0 => result.push(self.document(&text[range], None)),
                _ => {}
            }
        }
        // Content retains source markup. No rendered offsets are mislabeled as
        // original paragraph/character locations and no raw HTML is executed.
        Ok(result)
    }

    fn csv(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        let delimiter = string_option(&self.options, "delimiter", ",").as_bytes()[0];
        validate_csv_quotes(text.as_bytes(), delimiter, cancellation)?;
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .flexible(false)
            .from_reader(text.as_bytes());
        let headers = reader
            .headers()
            .map_err(|_| parse_error("header", "CSV header cannot be parsed"))?
            .clone();
        let mut names = HashSet::new();
        if headers
            .iter()
            .any(|name| name.trim().is_empty() || !names.insert(name))
        {
            return Err(parse_error(
                "header",
                "CSV columns must have unique nonempty names",
            ));
        }
        let question = string_option(&self.options, "question", "question");
        let answer = string_option(&self.options, "answer", "answer");
        let qa_columns = if matches!(self.method, Method::Qa) {
            Some((
                headers
                    .iter()
                    .position(|name| name == question)
                    .ok_or_else(|| parse_error("question", "CSV question column is missing"))?,
                headers
                    .iter()
                    .position(|name| name == answer)
                    .ok_or_else(|| parse_error("answer", "CSV answer column is missing"))?,
            ))
        } else {
            None
        };
        let mut result = Vec::new();
        for (index, record) in reader.records().enumerate() {
            check_cancelled(cancellation)?;
            let record = record.map_err(|_| {
                parse_error(&format!("row={}", index + 1), "CSV row cannot be parsed")
            })?;
            if let Some((question, answer)) = qa_columns {
                result.push(self.qa(&record[question], &record[answer]));
            } else {
                let fields: Map<String, Value> = headers
                    .iter()
                    .zip(record.iter())
                    .map(|(key, value)| (key.into(), json!(value)))
                    .collect();
                let content = headers
                    .iter()
                    .zip(record.iter())
                    .map(|(key, value)| format!("{key}: {value}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let mut document = self.document(&content, Some("row"));
                document
                    .metadata
                    .insert("table".into(), json!({"fields":fields}));
                result.push(document);
            }
        }
        Ok(result)
    }

    fn json(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        let value: Value =
            serde_json::from_str(text).map_err(|_| parse_error("data", "invalid JSON input"))?;
        let rows = match value {
            Value::Array(rows) => rows,
            value => vec![value],
        };
        let mut result = Vec::new();
        for (index, row) in rows.into_iter().enumerate() {
            check_cancelled(cancellation)?;
            let context = format!("row={}", index + 1);
            let document = match self.method {
                Method::Qa => {
                    let object = row
                        .as_object()
                        .ok_or_else(|| parse_error(&context, "QA row must be an object"))?;
                    let question =
                        scalar(object.get(string_option(&self.options, "question", "question")))
                            .ok_or_else(|| {
                                parse_error(&context, "QA question must be scalar content")
                            })?;
                    let answer =
                        scalar(object.get(string_option(&self.options, "answer", "answer")))
                            .ok_or_else(|| {
                                parse_error(&context, "QA answer must be scalar content")
                            })?;
                    self.qa(&question, &answer)
                }
                Method::Table => {
                    let object = row
                        .as_object()
                        .ok_or_else(|| parse_error(&context, "table row must be an object"))?;
                    let mut document = self.document(&row.to_string(), Some("row"));
                    document
                        .metadata
                        .insert("table".into(), json!({"fields":object}));
                    document
                }
                _ => {
                    let content = match &row {
                        Value::String(content) => Some(content.as_str()),
                        Value::Object(object) => object
                            .get(string_option(&self.options, "content", "content"))
                            .and_then(Value::as_str),
                        _ => None,
                    }
                    .ok_or_else(|| parse_error(&context, "JSON row needs text content"))?;
                    self.document(content, None)
                }
            };
            result.push(document);
        }
        Ok(result)
    }
}

fn scalar(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(value) => Some(value.clone()),
        value @ (Value::Number(_) | Value::Bool(_)) => Some(value.to_string()),
        _ => None,
    }
}

// The csv reader deliberately accepts some malformed quotes. Reject truncated
// or inconsistent quoted fields before treating its permissive output as facts.
fn validate_csv_quotes(
    data: &[u8],
    delimiter: u8,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    #[derive(Clone, Copy)]
    enum State {
        Start,
        Plain,
        Quoted,
        Closed,
    }
    let mut state = State::Start;
    for (index, byte) in data.iter().copied().enumerate() {
        if index % 4096 == 0 {
            check_cancelled(cancellation)?;
        }
        state = match (state, byte) {
            (State::Quoted, b'"') => State::Closed,
            (State::Quoted, _) => State::Quoted,
            (State::Closed, b'"') => State::Quoted,
            (_, byte) if byte == delimiter || matches!(byte, b'\r' | b'\n') => State::Start,
            (State::Closed, _) | (State::Plain, b'"') => {
                return Err(parse_error("data", "malformed CSV quoting"));
            }
            (State::Start, b'"') => State::Quoted,
            _ => State::Plain,
        };
    }
    if matches!(state, State::Quoted) {
        return Err(parse_error("data", "unterminated CSV quoted field"));
    }
    Ok(())
}

impl document::Parser for Parser {
    async fn parse(
        &self,
        data: &[u8],
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        check_cancelled(cancellation)?;
        let data = data.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(data);
        let text = std::str::from_utf8(data)
            .map_err(|_| parse_error("data", "input is not UTF-8 text"))?;
        let mut result = if matches!(self.method, Method::One) {
            // `one` preserves the complete source, but still validates its format.
            if self.format == Format::Json {
                let _: Value = serde_json::from_str(text)
                    .map_err(|_| parse_error("data", "invalid JSON input"))?;
            }
            if self.format == Format::Csv {
                self.csv(text, cancellation)?;
            }
            vec![self.document(text, None)]
        } else {
            match self.format {
                Format::Text => self.text(text, cancellation)?,
                Format::Markdown => self.markdown(text, cancellation)?,
                Format::Csv => self.csv(text, cancellation)?,
                Format::Json => self.json(text, cancellation)?,
            }
        };
        if self.enabled("remove_empty", true) {
            result.retain(|document| !document.content.trim().is_empty());
        }
        check_cancelled(cancellation)?;
        Ok(result)
    }
}
