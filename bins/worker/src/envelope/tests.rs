use super::*;
use jobs::{IndexJobPayload, JobPayload, JobType, SyncJobPayload};

#[test]
fn serialize_deserialize_roundtrip() {
    let envelope = JobEnvelope {
        job_id: "job-123".to_string(),
        job_type: JobType::Sync,
        payload: JobPayload::Sync(SyncJobPayload {
            source: "/tmp/collection.anki2".to_string(),
            run_migrations: true,
            index: false,
            force_reindex: true,
        }),
    };

    let json = serde_json::to_string(&envelope).expect("serialize");
    let deserialized: JobEnvelope = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.job_id, "job-123");
    assert_eq!(deserialized.job_type, JobType::Sync);
    assert_eq!(deserialized.payload, envelope.payload);
}

#[test]
fn serialize_index_payload() {
    let envelope = JobEnvelope {
        job_id: "job-456".to_string(),
        job_type: JobType::Index,
        payload: JobPayload::Index(IndexJobPayload {
            force_reindex: false,
        }),
    };

    let json = serde_json::to_string(&envelope).expect("serialize");
    let deserialized: JobEnvelope = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.job_id, "job-456");
    assert_eq!(deserialized.payload, envelope.payload);
}

#[test]
fn deserialize_from_json_string() {
    let json = r#"{"job_id":"abc","job_type":"sync","payload":{"kind":"sync","value":{"source":"/tmp/source.anki2","run_migrations":true,"index":true,"force_reindex":false}}}"#;
    let envelope: JobEnvelope = serde_json::from_str(json).expect("deserialize");

    assert_eq!(envelope.job_id, "abc");
    assert_eq!(envelope.job_type, JobType::Sync);
    assert_eq!(
        envelope.payload,
        JobPayload::Sync(SyncJobPayload {
            source: "/tmp/source.anki2".to_string(),
            run_migrations: true,
            index: true,
            force_reindex: false,
        })
    );
}

#[test]
fn deserialize_missing_field_fails() {
    let json = r#"{"job_id":"abc","job_type":"sync"}"#;
    let result = serde_json::from_str::<JobEnvelope>(json);
    assert!(result.is_err());
}

#[test]
fn serialize_sync_payload_preserves_flags() {
    let envelope = JobEnvelope {
        job_id: "job-789".to_string(),
        job_type: JobType::Sync,
        payload: JobPayload::Sync(SyncJobPayload {
            source: "/tmp/another.anki2".to_string(),
            run_migrations: false,
            index: true,
            force_reindex: true,
        }),
    };

    let json = serde_json::to_string(&envelope).expect("serialize");
    let deserialized: JobEnvelope = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.payload, envelope.payload);
}

#[test]
fn envelope_is_clone() {
    let envelope = JobEnvelope {
        job_id: "job-clone".to_string(),
        job_type: JobType::Index,
        payload: JobPayload::Index(IndexJobPayload {
            force_reindex: false,
        }),
    };
    let cloned = envelope.clone();
    assert_eq!(cloned.job_id, envelope.job_id);
}

#[test]
fn envelope_is_debug() {
    let envelope = JobEnvelope {
        job_id: "job-debug".to_string(),
        job_type: JobType::Sync,
        payload: JobPayload::Sync(SyncJobPayload {
            source: "/tmp/debug.anki2".to_string(),
            run_migrations: true,
            index: true,
            force_reindex: false,
        }),
    };
    let debug = format!("{:?}", envelope);
    assert!(debug.contains("job-debug"));
}

#[test]
fn deserialize_rejects_job_record_shaped_payload() {
    let json =
        r#"{"job_id":"rec-1","job_type":"index","payload":{},"status":"queued","progress":0.0}"#;
    let result = serde_json::from_str::<JobEnvelope>(json);
    assert!(result.is_err());
}
