use validation::{Severity, ValidationIssue, ValidationPipeline, ValidationResult, Validator};

// --- Severity ---

#[test]
fn severity_display_error() {
    assert_eq!(Severity::Error.to_string(), "error");
}

#[test]
fn severity_display_warning() {
    assert_eq!(Severity::Warning.to_string(), "warning");
}

#[test]
fn severity_display_info() {
    assert_eq!(Severity::Info.to_string(), "info");
}

#[test]
fn severity_ordering() {
    assert!(Severity::Error < Severity::Warning);
    assert!(Severity::Warning < Severity::Info);
}

#[test]
fn severity_serde_roundtrip() {
    let json = serde_json::to_string(&Severity::Warning).unwrap();
    assert_eq!(json, "\"warning\"");
    let parsed: Severity = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, Severity::Warning);
}

// --- ValidationIssue ---

#[test]
fn issue_error_constructor() {
    let issue = ValidationIssue::error("something wrong", "front");
    assert_eq!(issue.severity, Severity::Error);
    assert_eq!(issue.message, "something wrong");
    assert_eq!(issue.location, "front");
}

#[test]
fn issue_warning_constructor() {
    let issue = ValidationIssue::warning("minor thing", "back");
    assert_eq!(issue.severity, Severity::Warning);
    assert_eq!(issue.message, "minor thing");
    assert_eq!(issue.location, "back");
}

#[test]
fn issue_info_constructor() {
    let issue = ValidationIssue::info("note", "tags");
    assert_eq!(issue.severity, Severity::Info);
    assert_eq!(issue.message, "note");
    assert_eq!(issue.location, "tags");
}

// --- ValidationResult ---

#[test]
fn ok_result_is_valid() {
    let result = ValidationResult::ok();
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn result_with_only_warnings_is_valid() {
    let result = ValidationResult {
        issues: vec![ValidationIssue::warning("warn", "front")],
    };
    assert!(result.is_valid());
}

#[test]
fn result_with_error_is_not_valid() {
    let result = ValidationResult {
        issues: vec![ValidationIssue::error("err", "front")],
    };
    assert!(!result.is_valid());
}

#[test]
fn result_with_info_only_is_valid() {
    let result = ValidationResult {
        issues: vec![ValidationIssue::info("info", "front")],
    };
    assert!(result.is_valid());
}

#[test]
fn errors_filter() {
    let result = ValidationResult {
        issues: vec![
            ValidationIssue::error("e1", "front"),
            ValidationIssue::warning("w1", "back"),
            ValidationIssue::error("e2", "back"),
        ],
    };
    let errors = result.errors();
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].message, "e1");
    assert_eq!(errors[1].message, "e2");
}

#[test]
fn warnings_filter() {
    let result = ValidationResult {
        issues: vec![
            ValidationIssue::error("e1", "front"),
            ValidationIssue::warning("w1", "back"),
            ValidationIssue::warning("w2", "front"),
        ],
    };
    let warnings = result.warnings();
    assert_eq!(warnings.len(), 2);
    assert_eq!(warnings[0].message, "w1");
    assert_eq!(warnings[1].message, "w2");
}

#[test]
fn merge_empty_list() {
    let merged = ValidationResult::merge(&[]);
    assert!(merged.is_valid());
    assert!(merged.issues.is_empty());
}

#[test]
fn merge_concatenates_issues() {
    let r1 = ValidationResult {
        issues: vec![ValidationIssue::error("e1", "front")],
    };
    let r2 = ValidationResult {
        issues: vec![ValidationIssue::warning("w1", "back")],
    };
    let merged = ValidationResult::merge(&[r1, r2]);
    assert_eq!(merged.issues.len(), 2);
    assert_eq!(merged.issues[0].message, "e1");
    assert_eq!(merged.issues[1].message, "w1");
}

#[test]
fn merge_preserves_validity() {
    let r1 = ValidationResult::ok();
    let r2 = ValidationResult {
        issues: vec![ValidationIssue::error("e1", "front")],
    };
    let merged = ValidationResult::merge(&[r1, r2]);
    assert!(!merged.is_valid());
}

// --- ValidationPipeline ---

struct AlwaysOkValidator;
impl Validator for AlwaysOkValidator {
    fn validate(&self, _front: &str, _back: &str, _tags: &[String]) -> ValidationResult {
        ValidationResult::ok()
    }
}

struct AlwaysErrorValidator {
    msg: String,
}
impl Validator for AlwaysErrorValidator {
    fn validate(&self, _front: &str, _back: &str, _tags: &[String]) -> ValidationResult {
        ValidationResult {
            issues: vec![ValidationIssue::error(&self.msg, "test")],
        }
    }
}

#[test]
fn pipeline_empty_validators_returns_ok() {
    let pipeline = ValidationPipeline::new(vec![]);
    let result = pipeline.run("front", "back", &[]);
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn pipeline_single_ok_validator() {
    let pipeline = ValidationPipeline::new(vec![Box::new(AlwaysOkValidator)]);
    let result = pipeline.run("front", "back", &[]);
    assert!(result.is_valid());
}

#[test]
fn pipeline_single_error_validator() {
    let pipeline = ValidationPipeline::new(vec![Box::new(AlwaysErrorValidator {
        msg: "bad".to_string(),
    })]);
    let result = pipeline.run("front", "back", &[]);
    assert!(!result.is_valid());
    assert_eq!(result.issues.len(), 1);
}

#[test]
fn pipeline_merges_multiple_validators() {
    let pipeline = ValidationPipeline::new(vec![
        Box::new(AlwaysErrorValidator {
            msg: "err1".to_string(),
        }),
        Box::new(AlwaysErrorValidator {
            msg: "err2".to_string(),
        }),
        Box::new(AlwaysOkValidator),
    ]);
    let result = pipeline.run("front", "back", &[]);
    assert_eq!(result.issues.len(), 2);
    assert_eq!(result.issues[0].message, "err1");
    assert_eq!(result.issues[1].message, "err2");
}

// --- Send + Sync ---

#[test]
fn types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Severity>();
    assert_send_sync::<ValidationIssue>();
    assert_send_sync::<ValidationResult>();
    assert_send_sync::<ValidationPipeline>();
}
