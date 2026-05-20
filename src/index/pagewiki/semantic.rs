//! `Semantic`：使用 LLM 将文本切分为语义 PageWiki。
//!
//! 详见 `openspec/changes/rag-page-wiki/design.md` 第 6、10 节。

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use serde_json::Map;

use crate::index::pagewiki::base::Base;
use crate::index::pagewiki::spans::resolve_spans;
use crate::index::pagewiki::types::{Error, Evidence, PageWiki, Span};
use std::future::Future;
use std::pin::Pin;

/// 默认 prompt 模板。占位符：`{{text}}` / `{{scenario}}` / `{{metadata}}`。
pub const DEFAULT_PROMPT_TEMPLATE: &str = r#"你是一个文档结构化助手。请把下面的【文档】拆分成若干语义连贯的知识页。

要求：
1. 只输出 JSON 对象，格式严格为 {"pages": [...]}，不要使用 markdown 代码块。
2. 每个 page 包含：
   - header (string, 可选，段落标题)
   - content (string, 必填，字符数须在 450-550 之间)
   - keywords (array of string, 可选)
   - questions (array of string, 可选，可能命中本段的问题)
   - tags (array of string, 可选)
   - attributes (object, 可选，业务自定义属性)
   - graph (object, 可选，{node_type, neighbors, properties})
   - evidence (array of {start_text, end_text, start_line, end_line})：
     用于在原文中精确定位本页范围，start_line/end_line 为 1-based 行号。
3. 不要编造原文中不存在的事实。
4. 不要在 page 中输出 id / doc_id / version / scenario / idx，这些由下游填入。

场景：{{scenario}}
元数据：{{metadata}}

【文档】
{{text}}
"#;

/// LLM 响应的 page 结构（私有）。
#[derive(Debug, serde::Deserialize)]
struct LlmPage {
    #[serde(default)]
    header: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    questions: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    attributes: Map<String, serde_json::Value>,
    #[serde(default)]
    graph: crate::index::pagewiki::types::Graph,
    #[serde(default)]
    evidence: Vec<Evidence>,
}

/// LLM 响应的顶层结构（私有）。
#[derive(Debug, serde::Deserialize)]
struct LlmResponse {
    #[serde(default)]
    pages: Vec<LlmPage>,
}

/// LLM 驱动的 PageWiki 切分器。
pub struct Semantic {
    client: Client<OpenAIConfig>,
    /// 模型名称。
    model: String,
    /// prompt 模板（支持 `with_prompt` 覆盖）。
    prompt_template: String,
}

impl Semantic {
    /// 构造 [`Semantic`]。
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        let cfg = OpenAIConfig::default()
            .with_api_base(base_url)
            .with_api_key(api_key);
        Self {
            client: Client::with_config(cfg),
            model: model_name.into(),
            prompt_template: DEFAULT_PROMPT_TEMPLATE.to_string(),
        }
    }

    /// 替换 prompt 模板（完整替换，不拼接）。
    pub fn with_prompt(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    /// 渲染 prompt，替换三个占位符。
    fn render_prompt(&self, text: &str, scenario: &str, metadata: &str) -> String {
        self.prompt_template
            .replace("{{text}}", text)
            .replace("{{scenario}}", scenario)
            .replace("{{metadata}}", metadata)
    }
}

impl Base for Semantic {
    fn cut<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PageWiki>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let span = tracing::debug_span!(
                "semantic.cut",
                model = %self.model,
                text_len = text.chars().count(),
            );
            drop(span);

            // 渲染 prompt（场景与元数据由调用方在 trait 之外注入，这里以空串兜底）。
            let prompt = self.render_prompt(text, "", "");

            // 构造 LLM 请求。
            let msg = ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()
                .map_err(|e| Error::LlmRequest(e.to_string()))?;

            let req = CreateChatCompletionRequestArgs::default()
                .model(&self.model)
                .messages([msg.into()])
                .build()
                .map_err(|e| Error::LlmRequest(e.to_string()))?;

            // 发起 HTTP 请求。
            let resp = self
                .client
                .chat()
                .create(req)
                .await
                .map_err(|e| Error::LlmRequest(e.to_string()))?;

            // 提取文本内容。
            let raw = resp
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();

            // 解析 JSON。
            let llm_resp: LlmResponse =
                serde_json::from_str(&raw).map_err(|e| Error::LlmParse(e.to_string()))?;

            let mut out = Vec::with_capacity(llm_resp.pages.len());
            for page in llm_resp.pages {
                // 长度校验：content 字符数须在 [450, 550]。
                let actual = page.content.chars().count();
                if !(450..=550).contains(&actual) {
                    return Err(Error::ContentLength {
                        actual,
                        min: 450,
                        max: 550,
                    });
                }

                // 反算 spans。
                let spans: Vec<Span> = if page.evidence.is_empty() {
                    Vec::new()
                } else {
                    resolve_spans(text, &page.evidence)?
                };

                let pw = PageWiki {
                    header: page.header,
                    content: page.content,
                    keywords: page.keywords,
                    questions: page.questions,
                    tags: page.tags,
                    attributes: page.attributes,
                    graph: page.graph,
                    spans,
                    ..Default::default()
                };
                out.push(pw);
            }

            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_prompt_replaces_placeholders() {
        let s = Semantic::new("http://localhost", "key", "m");
        let rendered = s.render_prompt("BODY", "general", "{}");
        assert!(rendered.contains("BODY"));
        assert!(rendered.contains("general"));
        assert!(!rendered.contains("{{text}}"));
        assert!(!rendered.contains("{{scenario}}"));
        assert!(!rendered.contains("{{metadata}}"));
    }

    #[test]
    fn with_prompt_replaces_template() {
        let s = Semantic::new("http://localhost", "key", "m").with_prompt("custom {{text}} end");
        let rendered = s.render_prompt("X", "", "");
        assert_eq!(rendered, "custom X end");
    }

    #[test]
    fn default_template_has_placeholder() {
        assert!(DEFAULT_PROMPT_TEMPLATE.contains("{{text}}"));
    }
}
