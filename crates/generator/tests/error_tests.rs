use generator::GeneratorError;
use llm::LlmError;
use std::error::Error;

#[test]
fn generation_error_display() {
    let err = GeneratorError::Generation {
        message: "timeout".to_string(),
        model: Some("gpt-4".to_string()),
    };
    assert!(err.to_string().contains("timeout"));
    assert!(err.to_string().contains("generation"));
}

#[test]
fn generation_error_without_model() {
    let err = GeneratorError::Generation {
        message: "bad input".to_string(),
        model: None,
    };
    assert!(err.to_string().contains("bad input"));
}

#[test]
fn validation_error_display() {
    let err = GeneratorError::Validation {
        message: "empty content".to_string(),
    };
    assert!(err.to_string().contains("validation"));
    assert!(err.to_string().contains("empty content"));
}

#[test]
fn enhancement_error_display() {
    let err = GeneratorError::Enhancement {
        message: "rate limited".to_string(),
        model: Some("claude-3".to_string()),
    };
    assert!(err.to_string().contains("enhancement"));
    assert!(err.to_string().contains("rate limited"));
}

#[test]
fn apf_error_display() {
    let err = GeneratorError::Apf("missing sentinel".to_string());
    assert!(err.to_string().contains("APF"));
    assert!(err.to_string().contains("missing sentinel"));
}

#[test]
fn html_conversion_error_display() {
    let err = GeneratorError::HtmlConversion("unclosed tag".to_string());
    assert!(err.to_string().contains("HTML"));
    assert!(err.to_string().contains("unclosed tag"));
}

#[test]
fn llm_error_converts_to_generator_error() {
    let llm_err = LlmError::Connection("refused".to_string());
    let gen_err: GeneratorError = llm_err.into();
    assert!(gen_err.to_string().contains("connection"));
}

common::assert_send_sync!(GeneratorError);

#[test]
fn generator_error_implements_error_trait() {
    let err = GeneratorError::Validation {
        message: "test".to_string(),
    };
    // Should implement std::error::Error
    let _: &dyn Error = &err;
}
