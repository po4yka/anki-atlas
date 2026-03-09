use super::*;
use std::collections::HashMap;

#[test]
fn serialize_deserialize_roundtrip() {
    let envelope = JobEnvelope {
        job_id: "job-123".to_string(),
        task_name: "job_sync".to_string(),
        payload: HashMap::from([("key".to_string(), serde_json::json!("value"))]),
    };

    let json = serde_json::to_string(&envelope).expect("serialize");
    let deserialized: JobEnvelope = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.job_id, "job-123");
    assert_eq!(deserialized.task_name, "job_sync");
    assert_eq!(deserialized.payload.get("key"), Some(&serde_json::json!("value")));
}

#[test]
fn serialize_empty_payload() {
    let envelope = JobEnvelope {
        job_id: "job-456".to_string(),
        task_name: "job_index".to_string(),
        payload: HashMap::new(),
    };

    let json = serde_json::to_string(&envelope).expect("serialize");
    let deserialized: JobEnvelope = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.job_id, "job-456");
    assert!(deserialized.payload.is_empty());
}

#[test]
fn deserialize_from_json_string() {
    let json = r#"{"job_id":"abc","task_name":"job_sync","payload":{"deck":"Default"}}"#;
    let envelope: JobEnvelope = serde_json::from_str(json).expect("deserialize");

    assert_eq!(envelope.job_id, "abc");
    assert_eq!(envelope.task_name, "job_sync");
    assert_eq!(envelope.payload.get("deck"), Some(&serde_json::json!("Default")));
}

#[test]
fn deserialize_missing_field_fails() {
    let json = r#"{"job_id":"abc","task_name":"job_sync"}"#;
    let result = serde_json::from_str::<JobEnvelope>(json);
    assert!(result.is_err());
}

#[test]
fn serialize_complex_payload() {
    let envelope = JobEnvelope {
        job_id: "job-789".to_string(),
        task_name: "job_sync".to_string(),
        payload: HashMap::from([
            ("count".to_string(), serde_json::json!(42)),
            ("nested".to_string(), serde_json::json!({"a": 1})),
            ("list".to_string(), serde_json::json!([1, 2, 3])),
        ]),
    };

    let json = serde_json::to_string(&envelope).expect("serialize");
    let deserialized: JobEnvelope = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.payload.get("count"), Some(&serde_json::json!(42)));
    assert_eq!(deserialized.payload.get("nested"), Some(&serde_json::json!({"a": 1})));
    assert_eq!(deserialized.payload.get("list"), Some(&serde_json::json!([1, 2, 3])));
}

#[test]
fn envelope_is_clone() {
    let envelope = JobEnvelope {
        job_id: "job-clone".to_string(),
        task_name: "job_index".to_string(),
        payload: HashMap::new(),
    };
    let cloned = envelope.clone();
    assert_eq!(cloned.job_id, envelope.job_id);
}

#[test]
fn envelope_is_debug() {
    let envelope = JobEnvelope {
        job_id: "job-debug".to_string(),
        task_name: "job_sync".to_string(),
        payload: HashMap::new(),
    };
    let debug = format!("{:?}", envelope);
    assert!(debug.contains("job-debug"));
}
