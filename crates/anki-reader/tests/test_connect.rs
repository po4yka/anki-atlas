use anki_reader::connect::{AnkiConnectClient, ANKI_CONNECT_URL, ANKI_CONNECT_VERSION, DEFAULT_TIMEOUT_SECS};
use serde_json::json;
use std::collections::HashMap;
use wiremock::matchers::{body_json, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

// --- Constants ---

#[test]
fn constants_have_expected_values() {
    assert_eq!(ANKI_CONNECT_URL, "http://localhost:8765");
    assert_eq!(ANKI_CONNECT_VERSION, 6);
    assert_eq!(DEFAULT_TIMEOUT_SECS, 30);
}

// --- Construction ---

#[test]
fn client_default_uses_standard_url() {
    let client = AnkiConnectClient::default();
    // Just verifying construction doesn't panic
    let _ = client;
}

#[test]
fn client_custom_url() {
    let client = AnkiConnectClient::new("http://custom:1234", 10);
    let _ = client;
}

// --- invoke ---

#[tokio::test]
async fn invoke_sends_correct_payload() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(body_json(json!({
            "action": "version",
            "version": 6
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": 6,
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let result = client.invoke("version", None).await.unwrap();
    assert_eq!(result, json!(6));
}

#[tokio::test]
async fn invoke_sends_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(body_json(json!({
            "action": "findNotes",
            "version": 6,
            "params": {"query": "deck:Default"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": [1, 2, 3],
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let result = client
        .invoke("findNotes", Some(json!({"query": "deck:Default"})))
        .await
        .unwrap();
    assert_eq!(result, json!([1, 2, 3]));
}

#[tokio::test]
async fn invoke_returns_error_on_anki_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": null,
            "error": "some error"
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let result = client.invoke("badAction", None).await;
    assert!(result.is_err());
}

// --- ping ---

#[tokio::test]
async fn ping_returns_true_when_connected() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": 6,
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    assert!(client.ping().await);
}

#[tokio::test]
async fn ping_returns_false_when_refused() {
    // Use a port that's not listening
    let client = AnkiConnectClient::new("http://127.0.0.1:19999", 1);
    assert!(!client.ping().await);
}

// --- version ---

#[tokio::test]
async fn version_returns_number() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": 6,
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    assert_eq!(client.version().await, Some(6));
}

// --- deck_names ---

#[tokio::test]
async fn deck_names_returns_list() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": ["Default", "Japanese"],
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let names = client.deck_names().await.unwrap();
    assert_eq!(names, vec!["Default", "Japanese"]);
}

// --- create_deck ---

#[tokio::test]
async fn create_deck_returns_id() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": 12345,
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let id = client.create_deck("NewDeck").await.unwrap();
    assert_eq!(id, 12345);
}

// --- find_notes ---

#[tokio::test]
async fn find_notes_returns_ids() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": [100, 101, 102],
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let ids = client.find_notes("deck:Default").await.unwrap();
    assert_eq!(ids, vec![100, 101, 102]);
}

// --- add_note ---

#[tokio::test]
async fn add_note_returns_id_on_success() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": 999,
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let mut fields = HashMap::new();
    fields.insert("Front".into(), "Q".into());
    fields.insert("Back".into(), "A".into());
    let id = client
        .add_note("Default", "Basic", &fields, &["tag1".into()], false)
        .await
        .unwrap();
    assert_eq!(id, Some(999));
}

#[tokio::test]
async fn add_note_returns_none_on_duplicate() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": null,
            "error": "cannot create note because it is a duplicate"
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let mut fields = HashMap::new();
    fields.insert("Front".into(), "Q".into());
    let id = client
        .add_note("Default", "Basic", &fields, &[], false)
        .await
        .unwrap();
    assert_eq!(id, None);
}

// --- delete_notes ---

#[tokio::test]
async fn delete_notes_empty_is_noop() {
    let client = AnkiConnectClient::new("http://127.0.0.1:19999", 1);
    // Empty list should not make any HTTP calls, so even with bad URL it succeeds
    let result = client.delete_notes(&[]).await;
    assert!(result.is_ok());
}

// --- tags ---

#[tokio::test]
async fn get_tags_returns_list() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": ["vocab", "grammar", "kanji"],
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let tags = client.get_tags().await.unwrap();
    assert_eq!(tags, vec!["vocab", "grammar", "kanji"]);
}

#[tokio::test]
async fn add_tags_empty_is_noop() {
    let client = AnkiConnectClient::new("http://127.0.0.1:19999", 1);
    let result = client.add_tags(&[], "tag").await;
    assert!(result.is_ok());
}

// --- model_names ---

#[tokio::test]
async fn model_names_returns_list() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": ["Basic", "Cloze"],
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let names = client.model_names().await.unwrap();
    assert_eq!(names, vec!["Basic", "Cloze"]);
}

// --- sync ---

#[tokio::test]
async fn sync_succeeds() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": null,
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    client.sync().await.unwrap();
}

// --- Send + Sync ---

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn client_is_send_and_sync() {
    assert_send::<AnkiConnectClient>();
    assert_sync::<AnkiConnectClient>();
}
