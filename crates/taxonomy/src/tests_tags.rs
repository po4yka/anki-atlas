use crate::tags::*;

// === TagPrefix ===

#[test]
fn tag_prefix_as_str_android() {
    assert_eq!(TagPrefix::Android.as_str(), "android");
}

#[test]
fn tag_prefix_as_str_kotlin() {
    assert_eq!(TagPrefix::Kotlin.as_str(), "kotlin");
}

#[test]
fn tag_prefix_as_str_cs() {
    assert_eq!(TagPrefix::Cs.as_str(), "cs");
}

#[test]
fn tag_prefix_as_str_all_variants() {
    let pairs = [
        (TagPrefix::Android, "android"),
        (TagPrefix::Kotlin, "kotlin"),
        (TagPrefix::Cs, "cs"),
        (TagPrefix::Topic, "topic"),
        (TagPrefix::Difficulty, "difficulty"),
        (TagPrefix::Lang, "lang"),
        (TagPrefix::Source, "source"),
        (TagPrefix::Context, "context"),
        (TagPrefix::Bias, "bias"),
        (TagPrefix::Testing, "testing"),
        (TagPrefix::Architecture, "architecture"),
        (TagPrefix::Performance, "performance"),
        (TagPrefix::Platform, "platform"),
        (TagPrefix::Security, "security"),
        (TagPrefix::Networking, "networking"),
    ];
    for (prefix, expected) in pairs {
        assert_eq!(prefix.as_str(), expected, "failed for {:?}", prefix);
    }
}

// === VALID_PREFIXES ===

#[test]
fn valid_prefixes_count() {
    assert_eq!(VALID_PREFIXES.len(), 15);
}

#[test]
fn valid_prefixes_contains_android() {
    assert!(VALID_PREFIXES.contains(&"android"));
}

#[test]
fn valid_prefixes_contains_all() {
    for prefix in [
        "android",
        "kotlin",
        "cs",
        "topic",
        "difficulty",
        "lang",
        "source",
        "context",
        "bias",
        "testing",
        "architecture",
        "performance",
        "platform",
        "security",
        "networking",
    ] {
        assert!(VALID_PREFIXES.contains(&prefix), "missing prefix: {prefix}");
    }
}

// === META_TAG_PREFIXES ===

#[test]
fn meta_tag_prefixes_count() {
    assert_eq!(META_TAG_PREFIXES.len(), 4);
}

#[test]
fn meta_tag_prefixes_contents() {
    assert!(META_TAG_PREFIXES.contains(&"difficulty::"));
    assert!(META_TAG_PREFIXES.contains(&"lang::"));
    assert!(META_TAG_PREFIXES.contains(&"source::"));
    assert!(META_TAG_PREFIXES.contains(&"context::"));
}

// === META_TAGS ===

#[test]
fn meta_tags_contains_atomic() {
    assert_eq!(META_TAGS.len(), 1);
    assert!(META_TAGS.contains(&"atomic"));
}

// === lookup_tag ===

#[test]
fn lookup_tag_coroutines() {
    assert_eq!(lookup_tag("coroutines"), Some("kotlin_coroutines"));
}

#[test]
fn lookup_tag_kotlin_coroutines_variant() {
    assert_eq!(lookup_tag("Kotlin::Coroutines"), Some("kotlin_coroutines"));
}

#[test]
fn lookup_tag_flow() {
    assert_eq!(lookup_tag("flow"), Some("kotlin_flow"));
}

#[test]
fn lookup_tag_cognitive_bias() {
    assert_eq!(lookup_tag("cognitive-bias"), Some("cognitive_bias"));
}

#[test]
fn lookup_tag_bias() {
    assert_eq!(lookup_tag("bias"), Some("cognitive_bias"));
}

#[test]
fn lookup_tag_unknown() {
    assert_eq!(lookup_tag("nonexistent-tag-xyz"), None);
}

#[test]
fn lookup_tag_case_sensitive() {
    // "coroutines" exists but "Coroutines" (without prefix) may not
    // The mapping is case-sensitive as authored
    assert_eq!(lookup_tag("coroutines"), Some("kotlin_coroutines"));
}

// === TAG_MAPPING_LEN ===

#[test]
fn tag_mapping_len_matches_python() {
    assert_eq!(TAG_MAPPING_LEN, 464);
}

// === Topic tag sets ===

#[test]
fn kotlin_topic_tags_count() {
    assert_eq!(KOTLIN_TOPIC_TAGS.len(), 25);
}

#[test]
fn kotlin_topic_tags_contains_coroutines() {
    assert!(KOTLIN_TOPIC_TAGS.contains(&"kotlin_coroutines"));
}

#[test]
fn kotlin_topic_tags_contains_all() {
    let expected = [
        "kotlin_coroutines",
        "kotlin_flow",
        "kotlin_channels",
        "kotlin_dispatchers",
        "kotlin_collections",
        "kotlin_classes",
        "kotlin_types",
        "kotlin_functions",
        "kotlin_oop",
        "kotlin_initialization",
        "kotlin_delegation",
        "kotlin_syntax",
        "kotlin_equality",
        "kotlin_concurrency",
        "kotlin_java_interop",
        "kotlin_android",
        "kotlin_general",
        "kotlin_functional",
        "kotlin_builders",
        "kotlin_annotations",
        "kotlin_error_handling",
        "kotlin_performance",
        "kotlin_internals",
        "kotlin_memory",
        "kotlin_patterns",
    ];
    for tag in expected {
        assert!(KOTLIN_TOPIC_TAGS.contains(&tag), "missing: {tag}");
    }
}

#[test]
fn android_topic_tags_count() {
    assert_eq!(ANDROID_TOPIC_TAGS.len(), 27);
}

#[test]
fn android_topic_tags_contains_compose() {
    assert!(ANDROID_TOPIC_TAGS.contains(&"android_compose"));
}

#[test]
fn compsci_topic_tags_count() {
    assert_eq!(COMPSCI_TOPIC_TAGS.len(), 16);
}

#[test]
fn compsci_topic_tags_contains_algorithms() {
    assert!(COMPSCI_TOPIC_TAGS.contains(&"cs_algorithms"));
}

#[test]
fn cognitive_bias_topic_tags_count() {
    assert_eq!(COGNITIVE_BIAS_TOPIC_TAGS.len(), 11);
}

#[test]
fn cognitive_bias_topic_tags_contains_cognitive_bias() {
    assert!(COGNITIVE_BIAS_TOPIC_TAGS.contains(&"cognitive_bias"));
}

// === is_known_topic_tag ===

#[test]
fn is_known_topic_tag_android_compose() {
    assert!(is_known_topic_tag("android_compose"));
}

#[test]
fn is_known_topic_tag_cs_algorithms() {
    assert!(is_known_topic_tag("cs_algorithms"));
}

#[test]
fn is_known_topic_tag_kotlin_coroutines() {
    assert!(is_known_topic_tag("kotlin_coroutines"));
}

#[test]
fn is_known_topic_tag_cognitive_bias() {
    assert!(is_known_topic_tag("cognitive_bias"));
}

#[test]
fn is_known_topic_tag_unknown() {
    assert!(!is_known_topic_tag("unknown_thing"));
}

// === VALID_DIFFICULTIES ===

#[test]
fn valid_difficulties_count() {
    assert_eq!(VALID_DIFFICULTIES.len(), 3);
}

#[test]
fn valid_difficulties_contents() {
    assert!(VALID_DIFFICULTIES.contains(&"difficulty::easy"));
    assert!(VALID_DIFFICULTIES.contains(&"difficulty::medium"));
    assert!(VALID_DIFFICULTIES.contains(&"difficulty::hard"));
}

// === VALID_LANGS ===

#[test]
fn valid_langs_count() {
    assert_eq!(VALID_LANGS.len(), 2);
}

#[test]
fn valid_langs_contents() {
    assert!(VALID_LANGS.contains(&"lang::en"));
    assert!(VALID_LANGS.contains(&"lang::ru"));
}

// === TOPIC_PREFIXES ===

#[test]
fn topic_prefixes_count() {
    assert_eq!(TOPIC_PREFIXES.len(), 4);
}

#[test]
fn topic_prefixes_contents() {
    assert!(TOPIC_PREFIXES.contains(&"kotlin_"));
    assert!(TOPIC_PREFIXES.contains(&"android_"));
    assert!(TOPIC_PREFIXES.contains(&"cs_"));
    assert!(TOPIC_PREFIXES.contains(&"bias_"));
}

// === Send + Sync ===

#[test]
fn tag_prefix_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TagPrefix>();
}

// === Sample lookup coverage ===

#[test]
fn lookup_android_entries() {
    assert_eq!(lookup_tag("activity-lifecycle"), Some("android_lifecycle"));
    assert_eq!(lookup_tag("compose"), Some("android_compose"));
    assert_eq!(lookup_tag("recyclerview"), Some("android_layouts"));
}

#[test]
fn lookup_cs_entries() {
    assert_eq!(lookup_tag("algorithms"), Some("cs_algorithms"));
    assert_eq!(lookup_tag("design-patterns"), Some("cs_patterns"));
    assert_eq!(lookup_tag("dynamic-programming"), Some("cs_algorithms"));
}

#[test]
fn lookup_bias_entries() {
    assert_eq!(lookup_tag("dunning-kruger"), Some("bias_self"));
    assert_eq!(lookup_tag("sunk-cost"), Some("bias_decisions"));
    assert_eq!(lookup_tag("halo-effect"), Some("bias_perception"));
}
