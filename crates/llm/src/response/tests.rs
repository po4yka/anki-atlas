use super::*;

fn make_response(prompt: Option<u32>, completion: Option<u32>) -> LlmResponse {
    LlmResponse {
        text: "hello".to_string(),
        model: "test-model".to_string(),
        prompt_tokens: prompt,
        completion_tokens: completion,
        finish_reason: None,
        raw: HashMap::new(),
    }
}

#[test]
fn total_tokens_both_present() {
    let resp = make_response(Some(10), Some(20));
    assert_eq!(resp.total_tokens(), Some(30));
}

#[test]
fn total_tokens_only_prompt() {
    let resp = make_response(Some(10), None);
    assert_eq!(resp.total_tokens(), None);
}

#[test]
fn total_tokens_only_completion() {
    let resp = make_response(None, Some(20));
    assert_eq!(resp.total_tokens(), None);
}

#[test]
fn total_tokens_neither_present() {
    let resp = make_response(None, None);
    assert_eq!(resp.total_tokens(), None);
}

#[test]
fn total_tokens_zero_values() {
    let resp = make_response(Some(0), Some(0));
    assert_eq!(resp.total_tokens(), Some(0));
}

#[test]
fn response_serialization_roundtrip() {
    let resp = LlmResponse {
        text: "test output".to_string(),
        model: "gpt-4".to_string(),
        prompt_tokens: Some(100),
        completion_tokens: Some(50),
        finish_reason: Some("stop".to_string()),
        raw: {
            let mut m = HashMap::new();
            m.insert("id".to_string(), serde_json::json!("chatcmpl-123"));
            m
        },
    };
    let json = serde_json::to_string(&resp).unwrap();
    let deserialized: LlmResponse = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.text, resp.text);
    assert_eq!(deserialized.model, resp.model);
    assert_eq!(deserialized.prompt_tokens, resp.prompt_tokens);
    assert_eq!(deserialized.completion_tokens, resp.completion_tokens);
    assert_eq!(deserialized.finish_reason, resp.finish_reason);
    assert_eq!(deserialized.raw.len(), 1);
}

#[test]
fn response_clone() {
    let resp = make_response(Some(5), Some(10));
    let cloned = resp.clone();
    assert_eq!(cloned.text, resp.text);
    assert_eq!(cloned.total_tokens(), resp.total_tokens());
}

#[test]
fn response_debug() {
    let resp = make_response(Some(1), Some(2));
    let debug = format!("{:?}", resp);
    assert!(debug.contains("LlmResponse"));
    assert!(debug.contains("test-model"));
}
