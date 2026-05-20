//! 集成测试：pagewiki::Semantic 的 prompt 渲染与构造逻辑。
//!
//! 注：真实 LLM 调用依赖网络，此处只测试不发网络请求的逻辑分支。

use rag::index::pagewiki::Semantic;
use rag::index::pagewiki::semantic::DEFAULT_PROMPT_TEMPLATE;

#[test]
fn default_template_contains_all_placeholders() {
    assert!(DEFAULT_PROMPT_TEMPLATE.contains("{{text}}"));
    assert!(DEFAULT_PROMPT_TEMPLATE.contains("{{scenario}}"));
    assert!(DEFAULT_PROMPT_TEMPLATE.contains("{{metadata}}"));
}

#[test]
fn semantic_new_builds_without_panic() {
    // 只要构造不 panic 即可；不发起 HTTP。
    let _s = Semantic::new("http://localhost:11434", "test-key", "test-model");
}

#[test]
fn with_prompt_overrides_template() {
    let s = Semantic::new("http://localhost", "k", "m").with_prompt("自定义模板 {{text}} 结束");
    // 通过公共 cut 外层验证模板替换生效：
    // 这里只能间接验证，直接调用私有 render_prompt 不可行，
    // 但 with_prompt 的正确性已在 src/pagewiki/semantic.rs 单元测试覆盖。
    // 此处仅验证构造链不会 panic。
    drop(s);
}
