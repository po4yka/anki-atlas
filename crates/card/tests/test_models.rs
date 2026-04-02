use card::models::{
    Card, CardManifest, CardValidationError, CognitiveLoad, SyncAction, SyncActionType,
    VALID_NOTE_TYPES,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn valid_manifest() -> CardManifest {
    CardManifest::new(
        "kotlin-coroutines-0-en".into(),
        "kotlin-coroutines-0".into(),
        "en".into(),
        "notes/kotlin.md".into(),
        "Coroutines".into(),
        "note-001".into(),
        "Kotlin Basics".into(),
        0,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("valid manifest should succeed")
}

fn valid_card() -> Card {
    let manifest = valid_manifest();
    Card::new(
        "kotlin-coroutines-0-en".into(),
        "en".into(),
        "<div>This is a valid APF HTML content for testing</div>".into(),
        manifest,
        "Basic".into(),
        vec!["kotlin".into(), "async".into()],
        None,
    )
    .expect("valid card should succeed")
}

// ===========================================================================
// CardManifest tests
// ===========================================================================

#[test]
fn manifest_new_valid() {
    let m = valid_manifest();
    assert_eq!(m.slug, "kotlin-coroutines-0-en");
    assert_eq!(m.lang, "en");
    assert_eq!(m.card_index, 0);
    assert!(m.guid.is_none());
    assert!(m.hash6.is_none());
}

#[test]
fn manifest_rejects_empty_slug() {
    let result = CardManifest::new(
        "".into(),
        "base".into(),
        "en".into(),
        "path.md".into(),
        "anchor".into(),
        "note-001".into(),
        "Title".into(),
        0,
        None,
        None,
        None,
        None,
        None,
    );
    let err = result.unwrap_err();
    assert!(err.messages.iter().any(|m| m.contains("slug")));
}

#[test]
fn manifest_rejects_empty_required_fields() {
    let result = CardManifest::new(
        "".into(), // slug empty
        "".into(), // slug_base empty
        "en".into(),
        "".into(), // source_path empty
        "".into(), // source_anchor empty
        "".into(), // note_id empty
        "".into(), // note_title empty
        0,
        None,
        None,
        None,
        None,
        None,
    );
    let err = result.unwrap_err();
    // Should collect ALL errors, not just the first
    assert!(
        err.messages.len() >= 5,
        "expected at least 5 validation errors, got {}: {:?}",
        err.messages.len(),
        err.messages,
    );
}

#[test]
fn manifest_rejects_invalid_lang_length() {
    let result = CardManifest::new(
        "slug-test-0-eng".into(),
        "slug-test-0".into(),
        "eng".into(), // 3 chars, should be 2
        "path.md".into(),
        "anchor".into(),
        "note-001".into(),
        "Title".into(),
        0,
        None,
        None,
        None,
        None,
        None,
    );
    let err = result.unwrap_err();
    assert!(err.messages.iter().any(|m| m.contains("lang")));
}

#[test]
fn manifest_rejects_invalid_lang_value() {
    let result = CardManifest::new(
        "slug-test-0-xx".into(),
        "slug-test-0".into(),
        "xx".into(), // not a valid language
        "path.md".into(),
        "anchor".into(),
        "note-001".into(),
        "Title".into(),
        0,
        None,
        None,
        None,
        None,
        None,
    );
    let err = result.unwrap_err();
    assert!(err.messages.iter().any(|m| m.contains("lang")));
}

#[test]
fn manifest_rejects_invalid_hash6() {
    let result = CardManifest::new(
        "test-slug-0-en".into(),
        "test-slug-0".into(),
        "en".into(),
        "path.md".into(),
        "anchor".into(),
        "note-001".into(),
        "Title".into(),
        0,
        None,
        Some("xyz".into()), // not 6 hex chars
        None,
        None,
        None,
    );
    let err = result.unwrap_err();
    assert!(err.messages.iter().any(|m| m.contains("hash6")));
}

#[test]
fn manifest_accepts_valid_hash6() {
    let m = CardManifest::new(
        "test-slug-0-en".into(),
        "test-slug-0".into(),
        "en".into(),
        "path.md".into(),
        "anchor".into(),
        "note-001".into(),
        "Title".into(),
        0,
        None,
        Some("a1b2c3".into()),
        None,
        None,
        None,
    )
    .expect("valid hash6 should succeed");
    assert_eq!(m.hash6.as_deref(), Some("a1b2c3"));
}

#[test]
fn manifest_rejects_difficulty_out_of_range() {
    let result = CardManifest::new(
        "test-slug-0-en".into(),
        "test-slug-0".into(),
        "en".into(),
        "path.md".into(),
        "anchor".into(),
        "note-001".into(),
        "Title".into(),
        0,
        None,
        None,
        None,
        Some(1.5), // > 1.0
        None,
    );
    let err = result.unwrap_err();
    assert!(err.messages.iter().any(|m| m.contains("difficulty")));
}

#[test]
fn manifest_accepts_valid_difficulty() {
    let m = CardManifest::new(
        "test-slug-0-en".into(),
        "test-slug-0".into(),
        "en".into(),
        "path.md".into(),
        "anchor".into(),
        "note-001".into(),
        "Title".into(),
        0,
        None,
        None,
        None,
        Some(0.5),
        Some(CognitiveLoad::Intermediate),
    )
    .expect("valid difficulty should succeed");
    assert_eq!(m.difficulty, Some(0.5));
    assert_eq!(m.cognitive_load, Some(CognitiveLoad::Intermediate));
}

#[test]
fn manifest_anchor_url() {
    let m = valid_manifest();
    let url = m.anchor_url();
    assert_eq!(url, "[[notes/kotlin.md#Coroutines]]");
}

#[test]
fn manifest_is_linked_to_note() {
    let m = valid_manifest();
    assert!(m.is_linked_to_note());
}

#[test]
fn manifest_is_not_linked_when_note_id_empty() {
    let mut m = valid_manifest();
    m.note_id = String::new();
    assert!(!m.is_linked_to_note());
}

#[test]
fn manifest_with_guid() {
    let m = valid_manifest();
    let m2 = m.with_guid("guid-123".into());
    assert_eq!(m2.guid.as_deref(), Some("guid-123"));
    assert_eq!(m2.slug, m.slug); // rest unchanged
}

#[test]
fn manifest_with_hash() {
    let m = valid_manifest();
    let m2 = m.with_hash("abcdef".into());
    assert_eq!(m2.hash6.as_deref(), Some("abcdef"));
}

#[test]
fn manifest_with_obsidian_uri() {
    let m = valid_manifest();
    let m2 = m.with_obsidian_uri("obsidian://open?vault=test".into());
    assert_eq!(
        m2.obsidian_uri.as_deref(),
        Some("obsidian://open?vault=test")
    );
}

#[test]
fn manifest_with_fsrs_metadata() {
    let m = valid_manifest();
    let m2 = m.with_fsrs_metadata(0.7, CognitiveLoad::Advanced);
    assert_eq!(m2.difficulty, Some(0.7));
    assert_eq!(m2.cognitive_load, Some(CognitiveLoad::Advanced));
}

// ===========================================================================
// Card tests
// ===========================================================================

#[test]
fn card_new_valid() {
    let card = valid_card();
    assert_eq!(card.slug, "kotlin-coroutines-0-en");
    assert_eq!(card.language, "en");
    assert_eq!(card.note_type, "Basic");
    assert!(card.anki_guid.is_none());
}

#[test]
fn card_rejects_empty_slug() {
    let manifest = valid_manifest();
    let result = Card::new(
        "".into(),
        "en".into(),
        "<div>This is valid HTML content for test</div>".into(),
        manifest,
        "Basic".into(),
        vec![],
        None,
    );
    assert!(result.is_err());
}

#[test]
fn card_rejects_short_slug() {
    let manifest = valid_manifest();
    let result = Card::new(
        "ab".into(), // < 3 chars
        "en".into(),
        "<div>This is valid HTML content for test</div>".into(),
        manifest,
        "Basic".into(),
        vec![],
        None,
    );
    assert!(result.is_err());
}

#[test]
fn card_rejects_invalid_language() {
    let manifest = valid_manifest();
    let result = Card::new(
        "kotlin-coroutines-0-en".into(),
        "xx".into(), // invalid lang
        "<div>This is valid HTML content for test</div>".into(),
        manifest,
        "Basic".into(),
        vec![],
        None,
    );
    assert!(result.is_err());
}

#[test]
fn card_rejects_short_apf_html() {
    let manifest = valid_manifest();
    let result = Card::new(
        "kotlin-coroutines-0-en".into(),
        "en".into(),
        "short".into(), // < 10 chars
        manifest,
        "Basic".into(),
        vec![],
        None,
    );
    assert!(result.is_err());
}

#[test]
fn card_rejects_invalid_note_type() {
    let manifest = valid_manifest();
    let result = Card::new(
        "kotlin-coroutines-0-en".into(),
        "en".into(),
        "<div>This is valid HTML content for test</div>".into(),
        manifest,
        "InvalidType".into(),
        vec![],
        None,
    );
    assert!(result.is_err());
}

#[test]
fn card_rejects_mismatched_manifest_lang() {
    let manifest = CardManifest::new(
        "kotlin-coroutines-0-ru".into(),
        "kotlin-coroutines-0".into(),
        "ru".into(), // manifest says ru
        "notes/kotlin.md".into(),
        "Coroutines".into(),
        "note-001".into(),
        "Kotlin Basics".into(),
        0,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let result = Card::new(
        "kotlin-coroutines-0-ru".into(),
        "en".into(), // card says en -- mismatch!
        "<div>This is valid HTML content for test</div>".into(),
        manifest,
        "Basic".into(),
        vec![],
        None,
    );
    let err = result.unwrap_err();
    assert!(
        err.messages
            .iter()
            .any(|m| m.contains("lang") || m.contains("language"))
    );
}

#[test]
fn card_rejects_mismatched_manifest_slug() {
    let manifest = valid_manifest(); // slug = "kotlin-coroutines-0-en"
    let result = Card::new(
        "different-slug-0-en".into(), // mismatch
        "en".into(),
        "<div>This is valid HTML content for test</div>".into(),
        manifest,
        "Basic".into(),
        vec![],
        None,
    );
    let err = result.unwrap_err();
    assert!(err.messages.iter().any(|m| m.contains("slug")));
}

#[test]
fn card_rejects_empty_tag() {
    let manifest = valid_manifest();
    let result = Card::new(
        "kotlin-coroutines-0-en".into(),
        "en".into(),
        "<div>This is valid HTML content for test</div>".into(),
        manifest,
        "Basic".into(),
        vec!["valid".into(), "".into()], // empty tag
        None,
    );
    assert!(result.is_err());
}

#[test]
fn card_is_new_when_no_guid() {
    let card = valid_card();
    assert!(card.is_new());
}

#[test]
fn card_is_not_new_when_guid_set() {
    let card = valid_card().with_guid("guid-123".into()).unwrap();
    assert!(!card.is_new());
}

#[test]
fn card_content_hash_is_deterministic() {
    let card = valid_card();
    let hash1 = card.content_hash();
    let hash2 = card.content_hash();
    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 6);
}

#[test]
fn card_content_hash_changes_with_content() {
    let card1 = valid_card();
    let card2 = card1
        .update_content("<div>Completely different HTML content here</div>".into())
        .unwrap();
    assert_ne!(card1.content_hash(), card2.content_hash());
}

#[test]
fn card_content_hash_changes_with_tags() {
    let card1 = valid_card();
    let card2 = card1.with_tags(vec!["different-tag".into()]);
    assert_ne!(card1.content_hash(), card2.content_hash());
}

#[test]
fn card_content_hash_tag_order_independent() {
    let manifest1 = valid_manifest();
    let manifest2 = valid_manifest();
    let card1 = Card::new(
        "kotlin-coroutines-0-en".into(),
        "en".into(),
        "<div>This is a valid APF HTML content for testing</div>".into(),
        manifest1,
        "Basic".into(),
        vec!["alpha".into(), "beta".into()],
        None,
    )
    .unwrap();
    let card2 = Card::new(
        "kotlin-coroutines-0-en".into(),
        "en".into(),
        "<div>This is a valid APF HTML content for testing</div>".into(),
        manifest2,
        "Basic".into(),
        vec!["beta".into(), "alpha".into()],
        None,
    )
    .unwrap();
    assert_eq!(card1.content_hash(), card2.content_hash());
}

#[test]
fn card_with_guid_updates_both_card_and_manifest() {
    let card = valid_card();
    let card2 = card.with_guid("guid-abc".into()).unwrap();
    assert_eq!(card2.anki_guid.as_deref(), Some("guid-abc"));
    assert_eq!(card2.manifest.guid.as_deref(), Some("guid-abc"));
}

#[test]
fn card_update_content() {
    let card = valid_card();
    let new_html = "<div>Updated HTML content for the card test</div>";
    let card2 = card.update_content(new_html.into()).unwrap();
    assert_eq!(card2.apf_html, new_html);
    assert_ne!(card.content_hash(), card2.content_hash());
}

#[test]
fn card_update_content_rejects_short_html() {
    let card = valid_card();
    let result = card.update_content("short".into());
    assert!(result.is_err());
}

#[test]
fn card_with_tags() {
    let card = valid_card();
    let card2 = card.with_tags(vec!["new-tag".into()]);
    assert_eq!(card2.tags, vec!["new-tag"]);
    assert_eq!(card2.slug, card.slug); // rest unchanged
}

// ===========================================================================
// SyncAction tests
// ===========================================================================

#[test]
fn sync_action_create_valid() {
    let card = valid_card();
    let action = SyncAction::new(SyncActionType::Create, card, None, None).unwrap();
    assert_eq!(action.action_type, SyncActionType::Create);
    assert!(action.anki_guid.is_none());
}

#[test]
fn sync_action_update_requires_guid() {
    let card = valid_card();
    let result = SyncAction::new(SyncActionType::Update, card, None, None);
    let err = result.unwrap_err();
    assert!(err.messages.iter().any(|m| m.contains("guid")));
}

#[test]
fn sync_action_delete_requires_guid() {
    let card = valid_card();
    let result = SyncAction::new(SyncActionType::Delete, card, None, None);
    let err = result.unwrap_err();
    assert!(err.messages.iter().any(|m| m.contains("guid")));
}

#[test]
fn sync_action_update_with_guid_valid() {
    let card = valid_card();
    let action = SyncAction::new(
        SyncActionType::Update,
        card,
        Some("guid-xyz".into()),
        Some("content changed".into()),
    )
    .unwrap();
    assert_eq!(action.anki_guid.as_deref(), Some("guid-xyz"));
    assert_eq!(action.reason.as_deref(), Some("content changed"));
}

#[test]
fn sync_action_is_destructive() {
    let card = valid_card();
    let create = SyncAction::new(SyncActionType::Create, card.clone(), None, None).unwrap();
    let skip = SyncAction::new(SyncActionType::Skip, card.clone(), None, None).unwrap();
    let update = SyncAction::new(
        SyncActionType::Update,
        card.clone(),
        Some("guid".into()),
        None,
    )
    .unwrap();
    let delete = SyncAction::new(SyncActionType::Delete, card, Some("guid".into()), None).unwrap();

    assert!(!create.is_destructive());
    assert!(!skip.is_destructive());
    assert!(update.is_destructive());
    assert!(delete.is_destructive());
}

#[test]
fn sync_action_requires_confirmation() {
    let card = valid_card();
    let create = SyncAction::new(SyncActionType::Create, card.clone(), None, None).unwrap();
    let update = SyncAction::new(
        SyncActionType::Update,
        card.clone(),
        Some("guid".into()),
        None,
    )
    .unwrap();
    let delete = SyncAction::new(SyncActionType::Delete, card, Some("guid".into()), None).unwrap();

    assert!(!create.requires_confirmation());
    assert!(!update.requires_confirmation());
    assert!(delete.requires_confirmation());
}

#[test]
fn sync_action_describe() {
    let card = valid_card();
    let action = SyncAction::new(
        SyncActionType::Update,
        card,
        Some("guid-1".into()),
        Some("content changed".into()),
    )
    .unwrap();
    let desc = action.describe();
    assert!(desc.contains("UPDATE"));
    assert!(desc.contains("kotlin-coroutines-0-en"));
    assert!(desc.contains("content changed"));
}

#[test]
fn sync_action_describe_no_reason() {
    let card = valid_card();
    let action = SyncAction::new(SyncActionType::Create, card, None, None).unwrap();
    let desc = action.describe();
    assert!(desc.contains("CREATE"));
    assert!(desc.contains("kotlin-coroutines-0-en"));
}

// ===========================================================================
// VALID_NOTE_TYPES constant
// ===========================================================================

#[test]
fn valid_note_types_contains_expected() {
    assert!(VALID_NOTE_TYPES.contains(&"APF::Simple"));
    assert!(VALID_NOTE_TYPES.contains(&"APF::Cloze"));
    assert!(VALID_NOTE_TYPES.contains(&"Basic"));
    assert!(VALID_NOTE_TYPES.contains(&"Cloze"));
    assert_eq!(VALID_NOTE_TYPES.len(), 4);
}

// ===========================================================================
// Send + Sync
// ===========================================================================

common::assert_send_sync!(
    CardManifest,
    Card,
    SyncAction,
    SyncActionType,
    CognitiveLoad,
    CardValidationError,
);
