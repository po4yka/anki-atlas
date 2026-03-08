use super::*;
use std::error::Error;

#[test]
fn provider_error_display() {
    let err = LlmError::Provider {
        message: "rate limited".to_string(),
        source: None,
    };
    assert_eq!(err.to_string(), "provider error: rate limited");
}

#[test]
fn provider_error_with_source() {
    let source = std::io::Error::new(std::io::ErrorKind::Other, "inner");
    let err = LlmError::Provider {
        message: "wrapped".to_string(),
        source: Some(Box::new(source)),
    };
    assert!(err.to_string().contains("wrapped"));
}

#[test]
fn invalid_json_error_display() {
    let err = LlmError::InvalidJson {
        message: "expected object".to_string(),
        response_text: "not json".to_string(),
    };
    assert_eq!(err.to_string(), "invalid JSON response: expected object");
}

#[test]
fn invalid_json_preserves_response_text() {
    let err = LlmError::InvalidJson {
        message: "bad".to_string(),
        response_text: "raw response here".to_string(),
    };
    if let LlmError::InvalidJson { response_text, .. } = &err {
        assert_eq!(response_text, "raw response here");
    } else {
        panic!("wrong variant");
    }
}

#[test]
fn exhausted_error_display() {
    let err = LlmError::Exhausted {
        message: "timeout".to_string(),
        attempts: 3,
    };
    assert_eq!(err.to_string(), "request failed after 3 retries: timeout");
}

#[test]
fn http_error_display() {
    let err = LlmError::Http {
        status: 429,
        body: "too many requests".to_string(),
    };
    assert_eq!(err.to_string(), "HTTP error 429: too many requests");
}

#[test]
fn connection_error_display() {
    let err = LlmError::Connection("refused".to_string());
    assert_eq!(err.to_string(), "connection error: refused");
}

#[test]
fn error_is_std_error() {
    let err = LlmError::Connection("test".to_string());
    let _: &dyn Error = &err;
}

#[test]
fn error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LlmError>();
}

#[test]
fn error_debug_format() {
    let err = LlmError::Http {
        status: 500,
        body: "internal".to_string(),
    };
    let debug = format!("{:?}", err);
    assert!(debug.contains("Http"));
    assert!(debug.contains("500"));
}
