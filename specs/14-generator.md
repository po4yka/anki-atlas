# Spec: crate `generator`

## Source Reference
Python: `packages/generator/agents/` (models.py, generator.py, enhancer.py, validator.py) + `packages/card/apf/` (generator.py, converter.py, validator.py, linter.py, renderer.py)

## Purpose
Card generation pipeline: LLM-driven agents for generating, enhancing, splitting, and validating flashcards from structured content, plus APF (Anki Prompt Format) v2.1 HTML rendering, conversion, linting, and validation. The generator agent takes Q/A pairs and produces structured cards via LLM structured output. The enhancer improves cards and suggests splits. Pre/post validators check content quality. The APF submodule handles deterministic HTML rendering from card specs, markdown-to-HTML conversion, HTML sanitization, and format linting.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
llm = { path = "../llm" }
async-trait = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
thiserror = "2"
tracing = "0.1"
regex = "1"
html-escape = "0.2"
ammonia = "4"                     # HTML sanitization (Rust equivalent of nh3)

[dev-dependencies]
mockall = "0.13"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Public API

### Error (`src/error.rs`)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GeneratorError {
    #[error("card generation failed: {message}")]
    Generation { message: String, model: Option<String> },

    #[error("card validation failed: {message}")]
    Validation { message: String },

    #[error("card enhancement failed: {message}")]
    Enhancement { message: String, model: Option<String> },

    #[error("APF format error: {0}")]
    Apf(String),

    #[error("HTML conversion error: {0}")]
    HtmlConversion(String),

    #[error("LLM error: {0}")]
    Llm(#[from] llm::LlmError),
}
```

### Models (`src/models.rs`)

```rust
use serde::{Deserialize, Serialize};

/// A single generated flashcard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCard {
    pub card_index: u32,
    pub slug: String,
    pub lang: String,
    pub apf_html: String,
    pub confidence: f32,           // 0.0 - 1.0
    pub content_hash: String,      // first 16 hex chars of SHA-256 of apf_html
}

/// Result of a card generation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub cards: Vec<GeneratedCard>,
    pub total_cards: usize,
    pub model_used: String,
    pub generation_time_secs: f64,
    pub warnings: Vec<String>,
}

/// Dependencies passed to generation agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationDeps {
    pub note_title: String,
    pub topic: String,
    pub language_tags: Vec<String>,
    pub source_file: String,
}

/// A single planned card from a split decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitPlan {
    pub card_number: u32,
    pub concept: String,
    pub question: String,
    pub answer_summary: String,
}

/// Result of a card splitting analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitDecision {
    pub should_split: bool,
    pub card_count: u32,
    pub plans: Vec<SplitPlan>,
    pub reasoning: String,
}

/// Severity level for validation issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
}

/// A single validation issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub message: String,
    pub location: Option<String>,
}

/// Result of validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues.iter().filter(|i| i.severity == Severity::Error).collect()
    }

    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues.iter().filter(|i| i.severity == Severity::Warning).collect()
    }
}
```

### Generator Agent Trait (`src/agents/mod.rs`)

```rust
use async_trait::async_trait;
use crate::error::GeneratorError;
use crate::models::*;

/// Trait for card generation from Q/A pairs.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait GeneratorAgent: Send + Sync {
    async fn generate(
        &self,
        deps: &GenerationDeps,
        qa_pairs: &[(String, String)],
    ) -> Result<GenerationResult, GeneratorError>;
}

/// Trait for card enhancement and split suggestions.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait EnhancerAgent: Send + Sync {
    /// Enhance a single card.
    async fn enhance(
        &self,
        card: &GeneratedCard,
        deps: &GenerationDeps,
    ) -> Result<GeneratedCard, GeneratorError>;

    /// Analyze content and suggest whether to split into multiple cards.
    async fn suggest_split(
        &self,
        content: &str,
        deps: &GenerationDeps,
    ) -> Result<SplitDecision, GeneratorError>;
}

/// Trait for content validation.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait ValidatorAgent: Send + Sync {
    async fn validate(
        &self,
        content: &str,
        deps: &GenerationDeps,
    ) -> Result<ValidationResult, GeneratorError>;
}
```

### LLM-backed Agent Implementations (`src/agents/llm_generator.rs`, etc.)

```rust
use std::sync::Arc;
use llm::LlmProvider;

/// LLM-backed generator agent.
pub struct LlmGeneratorAgent {
    provider: Arc<dyn LlmProvider>,
    model_name: String,
    temperature: f32,          // default 0.3
}

impl LlmGeneratorAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self;
}

/// LLM-backed enhancer agent.
pub struct LlmEnhancerAgent {
    provider: Arc<dyn LlmProvider>,
    model_name: String,
    temperature: f32,          // default 0.3
}

impl LlmEnhancerAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self;
}

/// LLM-backed pre-validator agent.
pub struct LlmPreValidatorAgent {
    provider: Arc<dyn LlmProvider>,
    model_name: String,
    temperature: f32,          // default 0.0
}

/// LLM-backed post-validator agent.
pub struct LlmPostValidatorAgent {
    provider: Arc<dyn LlmProvider>,
    model_name: String,
    temperature: f32,          // default 0.0
}
```

### APF Renderer (`src/apf/renderer.rs`)

```rust
use serde::{Deserialize, Serialize};

pub const PROMPT_VERSION: &str = "apf-v2.1";

/// Input spec for rendering a card to APF HTML.
/// Uses generic fields so callers can construct freely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardSpec {
    pub card_index: u32,
    pub slug: String,
    pub slug_base: Option<String>,
    pub lang: String,
    pub card_type: String,         // "Simple", "Missing", "Draw"
    pub tags: Vec<String>,
    pub guid: String,
    pub source_path: Option<String>,
    pub source_anchor: Option<String>,
    pub title: String,
    pub key_point_code: Option<String>,
    pub key_point_code_lang: Option<String>,
    pub key_point_notes: Vec<String>,
    pub other_notes: Option<String>,
    pub extra: Option<String>,
}

/// Render a CardSpec to APF v2.1 HTML string.
pub fn render(spec: &CardSpec) -> String;

/// Render multiple cards separated by CARD_SEPARATOR markers.
pub fn render_batch(specs: &[CardSpec]) -> String;
```

### APF Linter (`src/apf/linter.rs`)

```rust
use serde::{Deserialize, Serialize};

pub const MAX_LINE_WIDTH: usize = 88;
pub const MIN_TAGS: usize = 3;
pub const MAX_TAGS: usize = 6;

/// Result of APF linting/validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl LintResult {
    pub fn is_valid(&self) -> bool { self.errors.is_empty() }
}

/// Validate APF card format against specification.
pub fn validate_apf(apf_html: &str, slug: Option<&str>) -> LintResult;
```

### APF Converter (`src/apf/converter.rs`)

```rust
/// Convert Markdown content to HTML.
/// Uses a basic markdown-to-HTML converter with code highlighting support.
pub fn markdown_to_html(md_content: &str, sanitize: bool) -> String;

/// Sanitize HTML using ammonia with Anki-safe tag/attribute allowlist.
pub fn sanitize_html(html: &str) -> String;

/// Convert a single APF field from Markdown to HTML.
pub fn convert_apf_field(field_content: &str) -> String;

/// Highlight code with language class annotations.
pub fn highlight_code(code: &str, language: Option<&str>) -> String;
```

### APF Validator (`src/apf/validator.rs`)

```rust
/// Validate HTML structure used in APF cards.
/// Returns list of validation error messages.
pub fn validate_card_html(apf_html: &str) -> Vec<String>;

/// Result of Markdown validation.
#[derive(Debug, Clone)]
pub struct MarkdownValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Validate Markdown structure (balanced fences, formatting markers).
pub fn validate_markdown(content: &str) -> MarkdownValidationResult;

/// Validate APF document with Markdown content.
pub fn validate_apf_markdown(apf_content: &str) -> MarkdownValidationResult;
```

### Module root (`src/lib.rs`)

```rust
pub mod agents;
pub mod apf;
pub mod error;
pub mod models;

pub use error::GeneratorError;
pub use models::*;
```

## Internal Details

### LLM Generator Agent
- Constructs system/user prompts for card generation.
- Sends prompt via `LlmProvider::generate_json` with `temperature=0.3`.
- Parses JSON response into `Vec<GeneratedCard>`, computing `content_hash` as SHA-256[:16] of `apf_html`.
- Measures wall-clock time via `std::time::Instant`.

### LLM Enhancer Agent
- `enhance`: sends card HTML + context to LLM, expects `enhanced_front`, `improvements`, `confidence` in response. Returns original card if no improvements suggested.
- `suggest_split`: sends content to LLM, expects `should_split`, `card_count`, `plans[]`, `reasoning`. Parses into `SplitDecision`.

### APF Linter Checks
- Required sentinels: `PROMPT_VERSION`, `BEGIN_CARDS`, `END_CARDS`.
- Final line must be `END_OF_CARDS`.
- Card header format: `<!-- Card N | slug: slug | CardType: Type | Tags: tag1 tag2 -->`.
- Tags: 3-6 count, no whitespace, alphanumeric/underscore/hyphen, lowercase convention.
- Manifest JSON must match header slug, contain required fields.
- Cloze density validation for Missing card type.
- Duplicate slug detection across cards.

### HTML Validation
- Detect backtick code fences (should be `<pre><code>`).
- Detect markdown bold/italic (should be `<strong>`/`<em>`).
- Validate `<pre>` contains nested `<code>`.
- Flag inline `<code>` outside `<pre>`.

### Allowed HTML Tags (sanitizer)
- `p, br, strong, b, em, i, u, s, code, pre, ul, ol, li, table, thead, tbody, tr, th, td, blockquote, h1-h6, a, img, div, span, sup, sub, hr, figure, figcaption`

## Acceptance Criteria
- [ ] `GeneratedCard` content_hash is SHA-256[:16] of apf_html
- [ ] `GenerationResult` tracks generation time and warnings
- [ ] `ValidationResult::is_valid` returns false when any Error-severity issue exists
- [ ] `ValidationResult::errors()` and `warnings()` filter by severity
- [ ] `SplitDecision` correctly represents split/no-split scenarios
- [ ] `MockGeneratorAgent`, `MockEnhancerAgent`, `MockValidatorAgent` compile
- [ ] `LlmGeneratorAgent::generate` produces cards from Q/A pairs via mock provider
- [ ] `LlmEnhancerAgent::enhance` returns original card when no improvements
- [ ] `LlmEnhancerAgent::suggest_split` parses split plans from LLM response
- [ ] `render` produces valid APF v2.1 HTML with all required sentinels
- [ ] `render` escapes code in `<pre><code>` blocks
- [ ] `render_batch` separates multiple cards with CARD_SEPARATOR
- [ ] `validate_apf` detects missing sentinels, invalid headers, tag count violations
- [ ] `validate_apf` validates manifest JSON consistency with header
- [ ] `validate_apf` detects duplicate slugs
- [ ] `validate_apf` validates cloze density for Missing type
- [ ] `validate_card_html` detects backtick fences, markdown bold/italic
- [ ] `validate_card_html` detects `<pre>` without nested `<code>`
- [ ] `validate_markdown` detects unclosed code fences
- [ ] `validate_markdown` detects unbalanced formatting markers
- [ ] `markdown_to_html` converts basic markdown (bold, italic, code, lists)
- [ ] `sanitize_html` strips disallowed tags but preserves allowed ones
- [ ] `highlight_code` wraps code in `<pre><code class="language-X">`
- [ ] All types are `Send + Sync` (compile-time assertion)
- [ ] `make check` equivalent passes (clippy, fmt, test)
