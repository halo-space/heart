use std::collections::{BTreeMap, HashSet};

use pulldown_cmark::{Event, Options as MarkdownOptions, Tag};
use serde_json::{Map, Value, json};

use crate::cancellation::check;
use crate::{
    Cancellation,
    document::{self, Document, Error},
};

mod liteparse;

#[derive(Clone, Copy, Debug)]
enum Method {
    General,
    One,
    Laws,
    Paper,
    Book,
    Manual,
    Presentation,
    Email,
    Tag,
    Resume,
    Picture,
    Qa,
    Table,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Text,
    Markdown,
    Csv,
    Json,
    Email,
    Binary(&'static str),
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
        "laws" => Method::Laws,
        "paper" => Method::Paper,
        "book" => Method::Book,
        "manual" => Method::Manual,
        "presentation" => Method::Presentation,
        "email" => Method::Email,
        "tag" => Method::Tag,
        "resume" => Method::Resume,
        "picture" => Method::Picture,
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
        "pdf" => Format::Binary("pdf"),
        "doc" => Format::Binary("doc"),
        "docx" => Format::Binary("docx"),
        "docm" => Format::Binary("docm"),
        "dot" => Format::Binary("dot"),
        "dotm" => Format::Binary("dotm"),
        "dotx" => Format::Binary("dotx"),
        "xlsx" => Format::Binary("xlsx"),
        "xls" => Format::Binary("xls"),
        "xlsm" => Format::Binary("xlsm"),
        "xlsb" => Format::Binary("xlsb"),
        "pptx" => Format::Binary("pptx"),
        "ppt" => Format::Binary("ppt"),
        "pptm" => Format::Binary("pptm"),
        "pot" => Format::Binary("pot"),
        "potm" => Format::Binary("potm"),
        "potx" => Format::Binary("potx"),
        "odt" => Format::Binary("odt"),
        "ott" => Format::Binary("ott"),
        "ods" => Format::Binary("ods"),
        "ots" => Format::Binary("ots"),
        "odp" => Format::Binary("odp"),
        "otp" => Format::Binary("otp"),
        "rtf" => Format::Binary("rtf"),
        "pages" => Format::Binary("pages"),
        "key" => Format::Binary("key"),
        "numbers" => Format::Binary("numbers"),
        "tsv" => Format::Binary("tsv"),
        "png" => Format::Binary("png"),
        "jpg" | "jpeg" => Format::Binary("jpg"),
        "tif" | "tiff" => Format::Binary("tiff"),
        "gif" => Format::Binary("gif"),
        "bmp" => Format::Binary("bmp"),
        "webp" => Format::Binary("webp"),
        "svg" => Format::Binary("svg"),
        "eml" => Format::Email,
        "msg" => {
            return Err(Error::new(
                "UNSUPPORTED",
                "the email parser currently supports EML, not MSG",
            ));
        }
        _ => {
            return Err(Error::new(
                "UNSUPPORTED",
                "parser format is not implemented",
            ));
        }
    };
    let engine = if !config.contains_key("engine") {
        match format {
            Format::Binary(_) => "liteparse",
            Format::Email => "email",
            _ => return Err(invalid("engine", "parser field must be nonempty text")),
        }
    } else {
        field(config, "engine")?
    };
    let compatible = matches!(
        (format, engine),
        (Format::Text, "text")
            | (Format::Markdown, "markdown")
            | (Format::Csv, "csv")
            | (Format::Json, "json")
            | (Format::Binary(_), "liteparse")
            | (Format::Email, "email")
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
    if matches!(method, Method::Laws) && matches!(format, Format::Csv | Format::Json) {
        return Err(Error::new(
            "UNSUPPORTED",
            "the laws parser requires text, markdown, or document bytes",
        ));
    }
    if matches!(method, Method::Presentation)
        && !matches!(format, Format::Binary("pdf" | "ppt" | "pptx"))
    {
        return Err(Error::new(
            "UNSUPPORTED",
            "the presentation parser requires PDF or PowerPoint bytes",
        ));
    }
    if matches!(method, Method::Paper | Method::Book | Method::Manual)
        && matches!(format, Format::Csv | Format::Json)
    {
        return Err(Error::new(
            "UNSUPPORTED",
            "this structure parser requires text, markdown, or document bytes",
        ));
    }
    if matches!(method, Method::Resume) && matches!(format, Format::Csv | Format::Json) {
        return Err(Error::new(
            "UNSUPPORTED",
            "the resume parser requires text, markdown, or document bytes",
        ));
    }
    if matches!(method, Method::Picture)
        && !matches!(
            format,
            Format::Binary("png" | "jpg" | "tiff" | "gif" | "bmp" | "webp" | "svg")
        )
    {
        return Err(Error::new(
            "UNSUPPORTED",
            "the picture parser requires image bytes",
        ));
    }
    if matches!(method, Method::Email) && !matches!(format, Format::Email) {
        return Err(Error::new(
            "UNSUPPORTED",
            "the email parser requires EML input",
        ));
    }
    if matches!(method, Method::Tag) && !matches!(format, Format::Text | Format::Csv | Format::Json)
    {
        return Err(Error::new(
            "UNSUPPORTED",
            "the tag parser requires text, CSV, or JSON input",
        ));
    }
    if matches!(format, Format::Email) && !matches!(method, Method::Email | Method::One) {
        return Err(Error::new(
            "UNSUPPORTED",
            "EML input requires the email or one parser",
        ));
    }
    let options = match config.get("options") {
        None => Map::new(),
        Some(Value::Object(options)) => options.clone(),
        _ => return Err(invalid("options", "parser options must be an object")),
    };
    let mut allowed = vec!["trim", "normalize_newlines", "remove_empty"];
    if matches!(format, Format::Binary(_)) {
        allowed.extend([
            "ocr",
            "ocr_language",
            "ocr_server_url",
            "max_pages",
            "target_pages",
            "output",
        ]);
    }
    if format == Format::Csv {
        allowed.push("delimiter");
    }
    if matches!(format, Format::Email) {
        allowed.push("fields");
    }
    if matches!(method, Method::Qa) {
        allowed.extend(["question", "answer"]);
    }
    if format == Format::Json && matches!(method, Method::General | Method::Tag) {
        allowed.push("content");
    }
    if matches!(method, Method::Tag) {
        allowed.push("tags");
    }
    allowed_keys(&options, &allowed)?;
    for key in ["trim", "normalize_newlines", "remove_empty"] {
        if options.get(key).is_some_and(|value| !value.is_boolean()) {
            return Err(invalid(key, "option must be boolean"));
        }
    }
    if matches!(format, Format::Binary(_)) {
        liteparse::validate(&options)?;
    }
    if matches!(format, Format::Email) {
        validate_email_fields(&options)?;
    }
    for key in ["question", "answer", "content"] {
        if options
            .get(key)
            .is_some_and(|value| value.as_str().is_none_or(|value| value.trim().is_empty()))
        {
            return Err(invalid(key, "field name must be nonempty text"));
        }
    }
    if options
        .get("tags")
        .is_some_and(|value| value.as_str().is_none_or(|value| value.trim().is_empty()))
    {
        return Err(invalid("tags", "field name must be nonempty text"));
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

const EMAIL_FIELDS: &[&str] = &[
    "from", "to", "cc", "bcc", "date", "subject", "body", "metadata",
];

fn validate_email_fields(options: &Map<String, Value>) -> Result<(), Error> {
    let Some(value) = options.get("fields") else {
        return Ok(());
    };
    let fields = value
        .as_array()
        .ok_or_else(|| invalid("fields", "email fields must be an array"))?;
    for field in fields {
        let field = field
            .as_str()
            .ok_or_else(|| invalid("fields", "email field names must be text"))?;
        if !EMAIL_FIELDS.contains(&field) {
            return Err(invalid("fields", "unsupported email field"));
        }
    }
    Ok(())
}

fn split_email_parts(text: &str) -> (&str, &str) {
    if let Some(index) = text.find("\r\n\r\n") {
        (&text[..index], &text[index + 4..])
    } else if let Some(index) = text.find("\n\n") {
        (&text[..index], &text[index + 2..])
    } else {
        (text, "")
    }
}

fn email_headers(text: &str) -> BTreeMap<String, String> {
    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    let mut current = None::<String>;
    for raw in text.lines() {
        let line = raw.trim_end_matches('\r');
        if line.starts_with([' ', '\t']) {
            if let Some(key) = current.as_ref()
                && let Some(value) = headers.get_mut(key)
            {
                value.push(' ');
                value.push_str(line.trim());
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        if key.is_empty() {
            continue;
        }
        headers.insert(key.clone(), value.trim().to_owned());
        current = Some(key);
    }
    headers
}

fn split_tags(value: &str) -> Vec<String> {
    value
        .split([',', ';', '|'])
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(str::to_owned)
        .collect()
}

fn is_law_heading(line: &str) -> bool {
    let line = line.trim();
    if line.is_empty() {
        return false;
    }
    if line.starts_with('第')
        && line
            .chars()
            .any(|character| matches!(character, '章' | '节' | '条'))
    {
        return true;
    }
    let lower = line.to_ascii_lowercase();
    for prefix in ["article ", "section ", "chapter "] {
        if let Some(rest) = lower.strip_prefix(prefix)
            && rest
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_digit())
        {
            return true;
        }
    }
    let digit_end = line
        .char_indices()
        .take_while(|(_, character)| character.is_ascii_digit())
        .map(|(index, character)| index + character.len_utf8())
        .last();
    digit_end.is_some_and(|end| matches!(line[end..].chars().next(), Some('、' | '.' | '．')))
}

fn is_section_heading(line: &str) -> bool {
    let line = line.trim();
    if line.is_empty() {
        return false;
    }
    if line.starts_with('#') {
        return true;
    }
    let lower = line.to_ascii_lowercase();
    if [
        "abstract",
        "introduction",
        "references",
        "appendix",
        "chapter ",
        "section ",
        "part ",
    ]
    .iter()
    .any(|prefix| lower == *prefix || lower.starts_with(prefix))
    {
        return true;
    }
    let digit_end = line
        .char_indices()
        .take_while(|(_, character)| character.is_ascii_digit() || *character == '.')
        .map(|(index, character)| index + character.len_utf8())
        .last();
    digit_end.is_some_and(|end| {
        end > 0
            && line[end..]
                .chars()
                .next()
                .is_some_and(|character| character.is_whitespace())
    })
}

fn is_resume_heading(line: &str) -> bool {
    let line = line.trim();
    if line.is_empty() {
        return false;
    }
    if line.starts_with('#') || is_section_heading(line) {
        return true;
    }
    [
        "个人信息",
        "基本信息",
        "联系方式",
        "教育经历",
        "教育背景",
        "工作经历",
        "工作经验",
        "项目经历",
        "项目经验",
        "专业技能",
        "技能",
        "获奖经历",
        "自我评价",
        "个人总结",
        "education",
        "experience",
        "employment",
        "projects",
        "skills",
        "summary",
        "contact",
    ]
    .iter()
    .any(|heading| line.eq_ignore_ascii_case(heading))
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
            check(cancellation)?;
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
            check(cancellation)?;
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

    /// Split legal text at explicit article or section headings. This is
    /// domain parsing, not length-based chunking: original content is kept
    /// verbatim and a missing heading leaves the input as one Document.
    fn laws(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut clause = false;
        for line in text.split_inclusive('\n') {
            check(cancellation)?;
            if is_law_heading(line) {
                if !current.trim().is_empty() {
                    result.push(self.law_document(&current, clause));
                    current.clear();
                }
                clause = true;
            }
            current.push_str(line);
        }
        if !current.trim().is_empty() {
            result.push(self.law_document(&current, clause));
        }
        if result.is_empty() && !text.is_empty() {
            result.push(self.law_document(text, false));
        }
        Ok(result)
    }

    fn sections(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut section = false;
        for line in text.split_inclusive('\n') {
            check(cancellation)?;
            if is_section_heading(line) {
                if !current.trim().is_empty() {
                    result.push(self.section_document(&current, section));
                    current.clear();
                }
                section = true;
            }
            current.push_str(line);
        }
        if !current.trim().is_empty() {
            result.push(self.section_document(&current, section));
        }
        if result.is_empty() && !text.is_empty() {
            result.push(self.section_document(text, false));
        }
        Ok(result)
    }

    fn section_document(&self, content: &str, section: bool) -> Document {
        self.document(content, section.then_some("section"))
    }

    fn resume_sections(
        &self,
        text: &str,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut section = false;
        for line in text.split_inclusive('\n') {
            check(cancellation)?;
            if is_resume_heading(line) {
                if !current.trim().is_empty() {
                    result.push(self.resume_document(&current, section));
                    current.clear();
                }
                section = true;
            }
            current.push_str(line);
        }
        if !current.trim().is_empty() {
            result.push(self.resume_document(&current, section));
        }
        if result.is_empty() && !text.is_empty() {
            result.push(self.resume_document(text, false));
        }
        Ok(result)
    }

    fn resume_document(&self, content: &str, section: bool) -> Document {
        self.document(content, section.then_some("resume_section"))
    }

    /// Parse the text-facing RFC 5322 representation of an EML message. MIME
    /// attachments and MSG/OLE decoding remain outside this first slice; the
    /// original body is preserved rather than silently dropping an unknown
    /// multipart part.
    fn email(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        check(cancellation)?;
        let (header_text, body) = split_email_parts(text);
        let headers = email_headers(header_text);
        let fields = self
            .options
            .get("fields")
            .and_then(Value::as_array)
            .filter(|fields| !fields.is_empty())
            .map(|fields| fields.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_else(|| EMAIL_FIELDS.to_vec());

        let mut structured = Map::new();
        let mut visible = String::new();
        for field in fields {
            check(cancellation)?;
            if field == "body" {
                structured.insert(field.into(), json!(body));
                if !body.is_empty() {
                    visible.push_str(body);
                    if !body.ends_with('\n') {
                        visible.push('\n');
                    }
                }
            } else if field == "metadata" {
                structured.insert(
                    field.into(),
                    Value::Object(
                        headers
                            .iter()
                            .filter(|(key, _)| !EMAIL_FIELDS.contains(&key.as_str()))
                            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                            .collect(),
                    ),
                );
            } else if let Some(value) = headers.get(field) {
                structured.insert(field.into(), json!(value));
                visible.push_str(field);
                visible.push_str(": ");
                visible.push_str(value);
                visible.push('\n');
            }
        }
        let mut document = self.document(&visible, Some("email"));
        document
            .metadata
            .insert("email".into(), Value::Object(structured));
        Ok(vec![document])
    }

    fn tag_document(&self, content: &str, tags: Vec<String>) -> Document {
        let mut document = self.document(content, Some("tag"));
        document
            .metadata
            .insert("tag".into(), json!({"tags": tags}));
        document
    }

    fn tag_text(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        let mut result = Vec::new();
        for line in text.lines() {
            check(cancellation)?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (content, tags) = line.split_once('\t').unwrap_or((line, ""));
            result.push(self.tag_document(content.trim(), split_tags(tags)));
        }
        Ok(result)
    }

    fn tag_csv(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
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
        let content_name = string_option(&self.options, "content", "content");
        let tags_name = string_option(&self.options, "tags", "tags");
        let content_index = headers
            .iter()
            .position(|name| name == content_name)
            .ok_or_else(|| parse_error("content", "CSV content column is missing"))?;
        let tags_index = headers
            .iter()
            .position(|name| name == tags_name)
            .ok_or_else(|| parse_error("tags", "CSV tags column is missing"))?;
        let mut result = Vec::new();
        for (index, record) in reader.records().enumerate() {
            check(cancellation)?;
            let record = record.map_err(|_| {
                parse_error(&format!("row={}", index + 1), "CSV row cannot be parsed")
            })?;
            let content = record[content_index].trim();
            if content.is_empty() {
                continue;
            }
            result.push(self.tag_document(content, split_tags(&record[tags_index])));
        }
        Ok(result)
    }

    fn tag_json(&self, text: &str, cancellation: &Cancellation) -> Result<Vec<Document>, Error> {
        let value: Value =
            serde_json::from_str(text).map_err(|_| parse_error("data", "invalid JSON input"))?;
        let rows = match value {
            Value::Array(rows) => rows,
            value => vec![value],
        };
        let content_name = string_option(&self.options, "content", "content");
        let tags_name = string_option(&self.options, "tags", "tags");
        let mut result = Vec::new();
        for (index, row) in rows.into_iter().enumerate() {
            check(cancellation)?;
            let object = row.as_object().ok_or_else(|| {
                parse_error(&format!("row={}", index + 1), "tag row must be an object")
            })?;
            let content = scalar(object.get(content_name)).ok_or_else(|| {
                parse_error(&format!("row={}", index + 1), "tag row needs text content")
            })?;
            let tags = match object.get(tags_name) {
                Some(Value::Array(values)) => values
                    .iter()
                    .filter_map(Value::as_str)
                    .flat_map(split_tags)
                    .collect(),
                Some(Value::String(value)) => split_tags(value),
                Some(_) => {
                    return Err(parse_error(
                        &format!("row={}", index + 1),
                        "tag row tags must be text or an array",
                    ));
                }
                None => Vec::new(),
            };
            result.push(self.tag_document(&content, tags));
        }
        Ok(result)
    }

    fn law_document(&self, content: &str, clause: bool) -> Document {
        self.document(content, clause.then_some("clause"))
    }

    fn split_law_pages(
        &self,
        pages: Vec<Document>,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        let mut result = Vec::new();
        for page in pages {
            check(cancellation)?;
            let parts = self.laws(&page.content, cancellation)?;
            if parts.len() <= 1 {
                result.push(page);
                continue;
            }
            for mut part in parts {
                // LiteParse may combine a text layer and OCR on one page, so
                // the domain splitter cannot claim `method = text` here.
                part.metadata.remove("method");
                // The page location is a reliable fact from LiteParse and
                // remains valid for every clause extracted from that page.
                if let Some(location) = page.metadata.get("location") {
                    part.metadata.insert("location".into(), location.clone());
                }
                part.metadata.insert("parser".into(), json!("liteparse"));
                result.push(part);
            }
        }
        Ok(result)
    }

    fn split_section_pages(
        &self,
        pages: Vec<Document>,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        let mut result = Vec::new();
        for page in pages {
            check(cancellation)?;
            let parts = self.sections(&page.content, cancellation)?;
            if parts.len() <= 1 {
                result.push(page);
                continue;
            }
            for mut part in parts {
                // A binary page can contain both extracted text and OCR;
                // keep the page fact but do not infer a text provenance.
                part.metadata.remove("method");
                if let Some(location) = page.metadata.get("location") {
                    part.metadata.insert("location".into(), location.clone());
                }
                part.metadata.insert("parser".into(), json!("liteparse"));
                result.push(part);
            }
        }
        Ok(result)
    }

    fn split_resume_pages(
        &self,
        pages: Vec<Document>,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        let mut result = Vec::new();
        for page in pages {
            check(cancellation)?;
            let parts = self.resume_sections(&page.content, cancellation)?;
            if parts.len() <= 1 {
                result.push(page);
                continue;
            }
            for mut part in parts {
                // Page location is reliable; text/OCR provenance is not
                // known at this layer, so no method is synthesized.
                part.metadata.remove("method");
                if let Some(location) = page.metadata.get("location") {
                    part.metadata.insert("location".into(), location.clone());
                }
                part.metadata.insert("parser".into(), json!("liteparse"));
                result.push(part);
            }
        }
        Ok(result)
    }

    fn mark_slides(
        &self,
        mut pages: Vec<Document>,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        for page in &mut pages {
            check(cancellation)?;
            page.metadata
                .insert("layout".into(), json!({"role":"slide"}));
        }
        Ok(pages)
    }

    fn mark_pictures(
        &self,
        mut documents: Vec<Document>,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        for document in &mut documents {
            check(cancellation)?;
            document.metadata.remove("location");
            document
                .metadata
                .insert("layout".into(), json!({"role":"image"}));
        }
        Ok(documents)
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
            check(cancellation)?;
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
            check(cancellation)?;
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
            check(cancellation)?;
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
    const EXTS: &'static [&'static str] = &[
        "text", "txt", "markdown", "md", "csv", "json", "pdf", "doc", "docx", "docm", "dot",
        "dotm", "dotx", "xlsx", "xls", "xlsm", "xlsb", "pptx", "ppt", "pptm", "pot", "potm",
        "potx", "odt", "ott", "ods", "ots", "odp", "otp", "rtf", "pages", "key", "numbers", "tsv",
        "png", "jpg", "jpeg", "tif", "tiff", "gif", "bmp", "webp", "svg", "eml",
    ];

    async fn parse(
        &self,
        data: &[u8],
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        check(cancellation)?;
        if let Format::Binary(extension) = self.format {
            let documents =
                liteparse::parse(data, extension, self.method, &self.options, cancellation).await?;
            if matches!(self.method, Method::Laws) {
                return self.split_law_pages(documents, cancellation);
            }
            if matches!(self.method, Method::Paper | Method::Book | Method::Manual) {
                return self.split_section_pages(documents, cancellation);
            }
            if matches!(self.method, Method::Resume) {
                return self.split_resume_pages(documents, cancellation);
            }
            if matches!(self.method, Method::Picture) {
                return self.mark_pictures(documents, cancellation);
            }
            if matches!(self.method, Method::Presentation) {
                return self.mark_slides(documents, cancellation);
            }
            return Ok(documents);
        }
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
                Format::Text if matches!(self.method, Method::Laws) => {
                    self.laws(text, cancellation)?
                }
                Format::Text if matches!(self.method, Method::Tag) => {
                    self.tag_text(text, cancellation)?
                }
                Format::Text
                    if matches!(self.method, Method::Paper | Method::Book | Method::Manual) =>
                {
                    self.sections(text, cancellation)?
                }
                Format::Text if matches!(self.method, Method::Resume) => {
                    self.resume_sections(text, cancellation)?
                }
                Format::Text => self.text(text, cancellation)?,
                Format::Markdown if matches!(self.method, Method::Laws) => {
                    self.laws(text, cancellation)?
                }
                Format::Markdown
                    if matches!(self.method, Method::Paper | Method::Book | Method::Manual) =>
                {
                    self.sections(text, cancellation)?
                }
                Format::Markdown if matches!(self.method, Method::Resume) => {
                    self.resume_sections(text, cancellation)?
                }
                Format::Markdown => self.markdown(text, cancellation)?,
                Format::Email => self.email(text, cancellation)?,
                Format::Csv if matches!(self.method, Method::Tag) => {
                    self.tag_csv(text, cancellation)?
                }
                Format::Csv => self.csv(text, cancellation)?,
                Format::Json if matches!(self.method, Method::Tag) => {
                    self.tag_json(text, cancellation)?
                }
                Format::Json => self.json(text, cancellation)?,
                Format::Binary(_) => unreachable!("binary formats return before UTF-8 decoding"),
            }
        };
        if self.enabled("remove_empty", true) {
            result.retain(|document| !document.content.trim().is_empty());
        }
        check(cancellation)?;
        Ok(result)
    }
}
