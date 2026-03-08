# Spec: crate `validation`

## Source Reference

Python: `packages/validation/` (pipeline.py, validators.py, quality.py)

## Purpose

Card content validation pipeline: defines severity levels, validation issues, and a composable pipeline that chains multiple validators. Ships with four built-in validators (HTML, content, format, tag) and a heuristic quality scorer across five dimensions (clarity, atomicity, testability, memorability, accuracy). All types are pure/stateless -- no I/O, no async.

## Dependencies

```toml
[dependencies]
common = { path = "../common" }
card = { path = "../card" }
serde = { version = "1", features = ["derive"] }
thiserror = "2"
regex = "1"
once_cell = "1"

[dev-dependencies]
# No extra test deps needed -- all validators are pure functions.
```

## Public API

### Pipeline (`src/pipeline.rs`)

```rust
use serde::{Deserialize, Serialize};

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

/// A single validation issue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub message: String,
    /// Optional location hint (e.g. "front", "back", "tags").
    pub location: String,
}

impl ValidationIssue {
    pub fn error(message: impl Into<String>, location: impl Into<String>) -> Self;
    pub fn warning(message: impl Into<String>, location: impl Into<String>) -> Self;
    pub fn info(message: impl Into<String>, location: impl Into<String>) -> Self;
}

/// Aggregated validation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// True when no ERROR-level issues exist.
    pub fn is_valid(&self) -> bool;

    /// Filter to ERROR-level issues only.
    pub fn errors(&self) -> Vec<&ValidationIssue>;

    /// Filter to WARNING-level issues only.
    pub fn warnings(&self) -> Vec<&ValidationIssue>;

    /// Merge multiple results into one.
    pub fn merge(results: &[ValidationResult]) -> ValidationResult;

    /// Empty (passing) result.
    pub fn ok() -> ValidationResult;
}

/// Trait for validators. All implementations must be Send + Sync.
pub trait Validator: Send + Sync {
    fn validate(&self, front: &str, back: &str, tags: &[String]) -> ValidationResult;
}

/// Pipeline that chains validators in sequence and merges results.
pub struct ValidationPipeline {
    validators: Vec<Box<dyn Validator>>,
}

impl ValidationPipeline {
    pub fn new(validators: Vec<Box<dyn Validator>>) -> Self;

    /// Run all validators and return merged result.
    pub fn run(&self, front: &str, back: &str, tags: &[String]) -> ValidationResult;
}
```

### Validators (`src/validators.rs`)

```rust
/// Check card content quality: empty fields, min/max length, unmatched code fences.
pub struct ContentValidator {
    pub min_length: usize, // default 10
    pub max_length: usize, // default 5000
}

impl ContentValidator {
    pub fn new() -> Self;
}

impl Validator for ContentValidator { /* ... */ }

/// Check APF format compliance: trailing whitespace, consecutive blank lines.
pub struct FormatValidator;

impl FormatValidator {
    pub fn new() -> Self;
}

impl Validator for FormatValidator { /* ... */ }

/// Validate HTML: balanced tags (stack-based), forbidden elements.
///
/// Forbidden tags: script, style, iframe, object, applet.
/// Void elements (br, img, hr, etc.) are skipped in balance checking.
pub struct HtmlValidator;

impl HtmlValidator {
    pub fn new() -> Self;
}

impl Validator for HtmlValidator { /* ... */ }

/// Validate tags against conventions: no empty tags, no invalid chars,
/// max 20 tags, no duplicates.
///
/// Valid tag characters: [a-zA-Z0-9_:/-].
pub struct TagValidator {
    pub max_tags: usize, // default 20
}

impl TagValidator {
    pub fn new() -> Self;
}

impl Validator for TagValidator { /* ... */ }
```

### Quality Scorer (`src/quality.rs`)

```rust
use serde::Serialize;

/// Five-dimension quality assessment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QualityScore {
    pub clarity: f64,
    pub atomicity: f64,
    pub testability: f64,
    pub memorability: f64,
    pub accuracy: f64,
}

impl QualityScore {
    /// Average of all five dimensions.
    pub fn overall(&self) -> f64;
}

/// Score a card using heuristic checks. Each dimension is 0.0-1.0.
///
/// Dimensions:
/// - clarity: penalizes vague openers ("explain", "describe"), yes/no questions, missing "?"
/// - atomicity: penalizes long questions (>20 words), multi-concept splits ("and"/"or")
/// - testability: penalizes extremely long answers (>100 words), empty answers
/// - memorability: penalizes long enumerations (>4 bullet items), long answers (>150 words)
/// - accuracy: penalizes missing question mark, empty front/back
pub fn assess_quality(front: &str, back: &str) -> QualityScore;
```

## Internal Details

### HTML tag balancing algorithm
1. Regex scan for `<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*?>`.
2. Skip void elements: `area, base, br, col, embed, hr, img, input, link, meta, param, source, track, wbr`.
3. If tag is in forbidden set (`script, style, iframe, object, applet`), emit ERROR and skip.
4. Maintain a stack of open tag names.
5. On opening tag: push to stack.
6. On closing tag: if stack top matches, pop. Otherwise emit ERROR ("unexpected closing tag").
7. After scanning, emit ERROR for each unclosed tag remaining on the stack.

### Content validation rules
| Condition | Severity | Message |
|-----------|----------|---------|
| Front is empty (after trim) | ERROR | "Front side is empty" |
| Front < 10 chars | WARNING | "Front side is very short" |
| Back is empty | ERROR | "Back side is empty" |
| Back < 10 chars | WARNING | "Back side is very short" |
| Front > 5000 chars | WARNING | "Front side exceeds 5000 chars" |
| Back > 5000 chars | WARNING | "Back side exceeds 5000 chars" |
| Odd count of ``` in front or back | ERROR | "Unmatched code fence" |

### Format validation rules
- Trailing whitespace on any line: WARNING per field.
- Three or more consecutive newlines (`\n\n\n`): WARNING per field.

### Tag validation rules
- Empty tag string: ERROR.
- Tag contains chars outside `[a-zA-Z0-9_:/-]`: WARNING.
- More than 20 tags: WARNING.
- Duplicate tags: WARNING.

### Quality scoring

**Clarity** (start at 1.0):
- -0.4 if front matches vague pattern: `^explain\s+`, `^describe\s+`, `^tell\s+(me\s+)?about\s+`, `^discuss\s+`, `^elaborate\s+(on\s+)?` (case-insensitive)
- -0.3 if front starts with yes/no starter: "is ", "are ", "does ", "do ", "can ", "will ", "has ", "have ", "was ", "were ", "did ", "could ", "would ", "should "
- -0.2 if front contains no "?"
- Floor at 0.0.

**Atomicity** (start at 1.0):
- -0.4 if word count > 30, else -0.2 if > 20
- -0.4 if 2+ occurrences of `\b(and|or)\b`, else -0.1 if 1
- Floor at 0.0.

**Testability** (start at 1.0):
- 0.0 if back is empty
- -0.5 if word count > 200, else -0.3 if > 100
- Floor at 0.0.

**Memorability** (start at 1.0):
- -0.5 if enumeration items > 7 (regex: `^\s*(?:[-*]|\d+[.)])\s`), else -0.2 if > 4
- -0.3 if word count > 150
- Floor at 0.0.

**Accuracy** (start at 1.0):
- -0.5 if front is empty
- -0.5 if back is empty
- -0.2 if front contains no "?"
- Floor at 0.0.

## Acceptance Criteria

- [ ] `cargo test -p validation` passes
- [ ] `cargo clippy -p validation -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `ValidationResult::ok()` has empty issues and `is_valid() == true`
- [ ] `ValidationResult::merge` concatenates issues from all inputs
- [ ] `ValidationResult::is_valid` returns false iff any issue has `Severity::Error`
- [ ] `ContentValidator` emits ERROR for empty front, empty back
- [ ] `ContentValidator` emits ERROR for unmatched code fences
- [ ] `ContentValidator` emits WARNING for short content (< 10 chars)
- [ ] `FormatValidator` detects trailing whitespace and consecutive blank lines
- [ ] `HtmlValidator` detects unclosed `<div>`, unexpected `</span>`, forbidden `<script>`
- [ ] `HtmlValidator` allows void elements (`<br>`, `<img>`) without closing tags
- [ ] `TagValidator` emits ERROR for empty tags, WARNING for invalid chars
- [ ] `TagValidator` emits WARNING for > 20 tags and duplicate tags
- [ ] `ValidationPipeline` runs all validators and merges results
- [ ] `ValidationPipeline` with empty validator list returns `ok()`
- [ ] `assess_quality` returns overall 1.0 for a well-formed Q&A card
- [ ] `assess_quality` penalizes "Explain X" questions (clarity < 1.0)
- [ ] `assess_quality` returns 0.0 testability for empty back
- [ ] `assess_quality` penalizes long enumerations (memorability < 1.0)
- [ ] `QualityScore::overall` is the mean of all five dimensions
- [ ] All scoring dimensions are clamped to `[0.0, 1.0]`
