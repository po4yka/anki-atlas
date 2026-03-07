"""Tests for packages.taxonomy: tags, normalization, validation, suggestions."""

from __future__ import annotations

from packages.taxonomy import TAG_MAPPING, VALID_PREFIXES, TagPrefix, normalize_tag, validate_tag
from packages.taxonomy.normalize import (
    is_meta_tag,
    is_topic_tag,
    normalize_tags,
    suggest_tag,
)
from packages.taxonomy.tags import (
    ALL_TOPIC_TAGS,
    ANDROID_TOPIC_TAGS,
    COGNITIVE_BIAS_TOPIC_TAGS,
    COMPSCI_TOPIC_TAGS,
    KOTLIN_TOPIC_TAGS,
    META_TAGS,
    VALID_DIFFICULTIES,
    VALID_LANGS,
)

# ---------------------------------------------------------------------------
# TagPrefix enum
# ---------------------------------------------------------------------------


class TestTagPrefix:
    def test_primary_prefixes_exist(self) -> None:
        assert TagPrefix.ANDROID == "android"
        assert TagPrefix.KOTLIN == "kotlin"
        assert TagPrefix.CS == "cs"
        assert TagPrefix.TOPIC == "topic"
        assert TagPrefix.DIFFICULTY == "difficulty"
        assert TagPrefix.LANG == "lang"

    def test_minor_prefixes_exist(self) -> None:
        assert TagPrefix.TESTING == "testing"
        assert TagPrefix.ARCHITECTURE == "architecture"
        assert TagPrefix.PERFORMANCE == "performance"
        assert TagPrefix.SECURITY == "security"

    def test_valid_prefixes_contains_all_enum_values(self) -> None:
        for prefix in TagPrefix:
            assert prefix.value in VALID_PREFIXES

    def test_valid_prefixes_count(self) -> None:
        assert len(VALID_PREFIXES) == len(TagPrefix)


# ---------------------------------------------------------------------------
# TAG_MAPPING
# ---------------------------------------------------------------------------


class TestTagMapping:
    def test_kotlin_mapping(self) -> None:
        assert TAG_MAPPING["coroutines"] == "kotlin_coroutines"
        assert TAG_MAPPING["Kotlin::Coroutines"] == "kotlin_coroutines"
        assert TAG_MAPPING["lambdas"] == "kotlin_functions"

    def test_android_mapping(self) -> None:
        assert TAG_MAPPING["activity-lifecycle"] == "android_lifecycle"
        assert TAG_MAPPING["compose"] == "android_compose"
        assert TAG_MAPPING["room"] == "android_room"

    def test_compsci_mapping(self) -> None:
        assert TAG_MAPPING["algorithms"] == "cs_algorithms"
        assert TAG_MAPPING["singleton-pattern"] == "cs_patterns"
        assert TAG_MAPPING["dijkstra"] == "cs_algorithms"

    def test_cognitive_bias_mapping(self) -> None:
        assert TAG_MAPPING["dunning-kruger"] == "bias_self"
        assert TAG_MAPPING["sunk-cost"] == "bias_decisions"
        assert TAG_MAPPING["cognitive-bias"] == "cognitive_bias"

    def test_mapping_not_empty(self) -> None:
        assert len(TAG_MAPPING) > 400


# ---------------------------------------------------------------------------
# Topic tag sets
# ---------------------------------------------------------------------------


class TestTopicTags:
    def test_all_topic_tags_is_union(self) -> None:
        expected = (
            KOTLIN_TOPIC_TAGS | ANDROID_TOPIC_TAGS | COMPSCI_TOPIC_TAGS | COGNITIVE_BIAS_TOPIC_TAGS
        )
        assert expected == ALL_TOPIC_TAGS

    def test_kotlin_topics(self) -> None:
        assert "kotlin_coroutines" in KOTLIN_TOPIC_TAGS
        assert "kotlin_general" in KOTLIN_TOPIC_TAGS

    def test_android_topics(self) -> None:
        assert "android_compose" in ANDROID_TOPIC_TAGS
        assert "android_lifecycle" in ANDROID_TOPIC_TAGS

    def test_compsci_topics(self) -> None:
        assert "cs_algorithms" in COMPSCI_TOPIC_TAGS
        assert "cs_patterns" in COMPSCI_TOPIC_TAGS

    def test_bias_topics(self) -> None:
        assert "cognitive_bias" in COGNITIVE_BIAS_TOPIC_TAGS
        assert "bias_decisions" in COGNITIVE_BIAS_TOPIC_TAGS


# ---------------------------------------------------------------------------
# normalize_tag
# ---------------------------------------------------------------------------


class TestNormalizeTag:
    def test_empty_and_whitespace(self) -> None:
        assert normalize_tag("") == ""
        assert normalize_tag("  ") == ""

    def test_known_mapping(self) -> None:
        assert normalize_tag("coroutines") == "kotlin_coroutines"
        assert normalize_tag("Kotlin::Flow") == "kotlin_flow"
        assert normalize_tag("dijkstra") == "cs_algorithms"

    def test_meta_tags_preserved(self) -> None:
        assert normalize_tag("atomic") == "atomic"
        assert normalize_tag("difficulty::hard") == "difficulty::hard"
        assert normalize_tag("lang::en") == "lang::en"
        assert normalize_tag("source::my-book") == "source::my-book"

    def test_already_prefixed_preserved(self) -> None:
        assert normalize_tag("kotlin_coroutines") == "kotlin_coroutines"
        assert normalize_tag("android_compose") == "android_compose"
        assert normalize_tag("cs_algorithms") == "cs_algorithms"
        assert normalize_tag("bias_decisions") == "bias_decisions"
        assert normalize_tag("cognitive_bias") == "cognitive_bias"

    def test_unknown_tag_lowercased_kebab(self) -> None:
        assert normalize_tag("My_Custom_Tag") == "my-custom-tag"
        assert normalize_tag("SomeNewTopic") == "somenewtopic"

    def test_strips_whitespace(self) -> None:
        assert normalize_tag("  coroutines  ") == "kotlin_coroutines"

    def test_unknown_with_colons(self) -> None:
        assert normalize_tag("foo::bar") == "foo-bar"

    def test_unknown_with_slashes(self) -> None:
        assert normalize_tag("foo/bar") == "foo-bar"


# ---------------------------------------------------------------------------
# normalize_tags
# ---------------------------------------------------------------------------


class TestNormalizeTags:
    def test_dedup_and_sort(self) -> None:
        result = normalize_tags(["coroutines", "Kotlin::Coroutines", "flow"])
        assert result == ["kotlin_coroutines", "kotlin_flow"]

    def test_empty_filtered(self) -> None:
        result = normalize_tags(["", "  ", "coroutines"])
        assert result == ["kotlin_coroutines"]

    def test_empty_list(self) -> None:
        assert normalize_tags([]) == []


# ---------------------------------------------------------------------------
# validate_tag
# ---------------------------------------------------------------------------


class TestValidateTag:
    def test_valid_double_colon_tag(self) -> None:
        assert validate_tag("android::compose") == []
        assert validate_tag("kotlin::coroutines") == []
        assert validate_tag("difficulty::hard") == []

    def test_empty_tag(self) -> None:
        issues = validate_tag("")
        assert len(issues) == 1
        assert "empty" in issues[0].lower()

    def test_underscore_prefix_flagged(self) -> None:
        issues = validate_tag("kotlin_something_new")
        assert any("::" in issue for issue in issues)

    def test_slash_separator_flagged(self) -> None:
        issues = validate_tag("android/compose")
        assert any("/" in issue for issue in issues)

    def test_too_deep_hierarchy(self) -> None:
        issues = validate_tag("a::b::c")
        assert any("deep" in issue.lower() for issue in issues)

    def test_uppercase_prefix_flagged(self) -> None:
        issues = validate_tag("Android::compose")
        assert any("lowercase" in issue.lower() for issue in issues)

    def test_duplicate_hyphen_flagged(self) -> None:
        issues = validate_tag("my--tag")
        assert any("duplicate" in issue.lower() for issue in issues)

    def test_known_topic_tag_no_underscore_issue(self) -> None:
        # Known topic tags like "kotlin_coroutines" are in ALL_TOPIC_TAGS
        # and should not be flagged for underscore usage
        issues = validate_tag("kotlin_coroutines")
        assert not any("::" in issue for issue in issues)


# ---------------------------------------------------------------------------
# suggest_tag
# ---------------------------------------------------------------------------


class TestSuggestTag:
    def test_close_match_found(self) -> None:
        suggestions = suggest_tag("coroutins")  # typo for "coroutines"
        assert len(suggestions) > 0
        assert "coroutines" in suggestions

    def test_no_match_for_gibberish(self) -> None:
        suggestions = suggest_tag("xyzzyplugh")
        assert suggestions == []

    def test_empty_input(self) -> None:
        assert suggest_tag("") == []
        assert suggest_tag("  ") == []

    def test_max_5_results(self) -> None:
        # "cs" is a prefix of many tags
        suggestions = suggest_tag("cs")
        assert len(suggestions) <= 5


# ---------------------------------------------------------------------------
# is_meta_tag / is_topic_tag
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_meta_tag(self) -> None:
        assert is_meta_tag("atomic") is True
        assert is_meta_tag("difficulty::hard") is True
        assert is_meta_tag("lang::en") is True
        assert is_meta_tag("kotlin_coroutines") is False

    def test_is_topic_tag(self) -> None:
        assert is_topic_tag("kotlin_coroutines") is True
        assert is_topic_tag("android_compose") is True
        assert is_topic_tag("cognitive_bias") is True
        assert is_topic_tag("atomic") is False
        assert is_topic_tag("difficulty::hard") is False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_difficulties(self) -> None:
        expected = frozenset({"difficulty::easy", "difficulty::medium", "difficulty::hard"})
        assert expected == VALID_DIFFICULTIES

    def test_valid_langs(self) -> None:
        assert "lang::en" in VALID_LANGS
        assert "lang::ru" in VALID_LANGS

    def test_meta_tags(self) -> None:
        assert "atomic" in META_TAGS
