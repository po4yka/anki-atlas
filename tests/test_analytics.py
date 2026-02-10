"""Tests for analytics module (taxonomy, labeling, coverage)."""

import tempfile
from pathlib import Path

import pytest

from packages.analytics.coverage import TopicCoverage, TopicGap
from packages.analytics.labeling import TopicAssignment, TopicLabeler, cosine_similarity
from packages.analytics.taxonomy import (
    Taxonomy,
    Topic,
    load_taxonomy_from_yaml,
)
from packages.common.config import Settings
from packages.indexer.embeddings import MockEmbeddingProvider


class TestTopic:
    """Tests for Topic dataclass."""

    def test_parent_path_root(self) -> None:
        """Test parent path for root topic."""
        topic = Topic(path="programming", label="Programming")
        assert topic.parent_path is None

    def test_parent_path_nested(self) -> None:
        """Test parent path for nested topic."""
        topic = Topic(path="programming/python/async", label="Async")
        assert topic.parent_path == "programming/python"

    def test_depth(self) -> None:
        """Test depth calculation."""
        assert Topic(path="a", label="A").depth == 0
        assert Topic(path="a/b", label="B").depth == 1
        assert Topic(path="a/b/c", label="C").depth == 2

    def test_name(self) -> None:
        """Test name extraction."""
        topic = Topic(path="programming/python", label="Python")
        assert topic.name == "python"


class TestTaxonomy:
    """Tests for Taxonomy class."""

    def test_empty_taxonomy(self) -> None:
        """Test empty taxonomy."""
        taxonomy = Taxonomy()
        assert taxonomy.get("foo") is None
        assert taxonomy.all_topics() == []

    def test_get_topic(self) -> None:
        """Test getting topic by path."""
        taxonomy = Taxonomy()
        topic = Topic(path="test", label="Test")
        taxonomy.topics["test"] = topic
        taxonomy.roots.append(topic)

        assert taxonomy.get("test") == topic
        assert taxonomy.get("nonexistent") is None

    def test_all_topics_order(self) -> None:
        """Test that all_topics returns depth-first order."""
        # Build a small tree
        root = Topic(path="a", label="A")
        child1 = Topic(path="a/b", label="B")
        child2 = Topic(path="a/c", label="C")
        grandchild = Topic(path="a/b/d", label="D")

        root.children = [child1, child2]
        child1.children = [grandchild]

        taxonomy = Taxonomy(
            topics={
                "a": root,
                "a/b": child1,
                "a/c": child2,
                "a/b/d": grandchild,
            },
            roots=[root],
        )

        all_topics = taxonomy.all_topics()
        paths = [t.path for t in all_topics]
        assert paths == ["a", "a/b", "a/b/d", "a/c"]

    def test_subtree(self) -> None:
        """Test subtree extraction."""
        taxonomy = Taxonomy(
            topics={
                "a": Topic(path="a", label="A"),
                "a/b": Topic(path="a/b", label="B"),
                "a/b/c": Topic(path="a/b/c", label="C"),
                "x": Topic(path="x", label="X"),
            }
        )

        subtree = taxonomy.subtree("a")
        paths = {t.path for t in subtree}
        assert paths == {"a", "a/b", "a/b/c"}


class TestLoadTaxonomyFromYaml:
    """Tests for YAML taxonomy loading."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading non-existent file returns empty taxonomy."""
        taxonomy = load_taxonomy_from_yaml(Path("/nonexistent/file.yml"))
        assert len(taxonomy.topics) == 0

    def test_load_simple_taxonomy(self) -> None:
        """Test loading simple taxonomy."""
        yaml_content = """
topics:
  - path: programming
    label: Programming
    description: General programming topics
    children:
      - path: programming/python
        label: Python
        description: Python programming language
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            taxonomy = load_taxonomy_from_yaml(yaml_path)

            assert len(taxonomy.topics) == 2
            assert len(taxonomy.roots) == 1
            assert taxonomy.roots[0].path == "programming"
            python_topic = taxonomy.get("programming/python")
            assert python_topic is not None
            assert python_topic.label == "Python"
        finally:
            yaml_path.unlink()

    def test_load_empty_yaml(self) -> None:
        """Test loading empty YAML returns empty taxonomy."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("")
            yaml_path = Path(f.name)

        try:
            taxonomy = load_taxonomy_from_yaml(yaml_path)
            assert len(taxonomy.topics) == 0
        finally:
            yaml_path.unlink()


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors is 1."""
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.0001

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors is 0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert abs(cosine_similarity(vec1, vec2)) < 0.0001

    def test_opposite_vectors(self) -> None:
        """Test similarity of opposite vectors is -1."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        assert abs(cosine_similarity(vec1, vec2) + 1.0) < 0.0001


class TestTopicLabeler:
    """Tests for TopicLabeler."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings."""
        return Settings(
            embedding_provider="mock",
            embedding_model="mock",
            embedding_dimension=384,
            postgres_url="postgresql://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
        )

    @pytest.fixture
    def mock_embedding_provider(self) -> MockEmbeddingProvider:
        """Create mock embedding provider."""
        return MockEmbeddingProvider(dimension=384)

    async def test_embed_topics(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
    ) -> None:
        """Test embedding topics."""
        taxonomy = Taxonomy(
            topics={
                "python": Topic(
                    path="python",
                    label="Python",
                    description="Python programming",
                ),
                "java": Topic(
                    path="java",
                    label="Java",
                    description="Java programming",
                ),
            },
            roots=[
                Topic(path="python", label="Python"),
                Topic(path="java", label="Java"),
            ],
        )

        labeler = TopicLabeler(
            settings=settings,
            embedding_provider=mock_embedding_provider,
        )

        embeddings = await labeler.embed_topics(taxonomy)

        assert len(embeddings) == 2
        assert "python" in embeddings
        assert "java" in embeddings
        assert len(embeddings["python"]) == 384

    async def test_embed_empty_taxonomy(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
    ) -> None:
        """Test embedding empty taxonomy."""
        labeler = TopicLabeler(
            settings=settings,
            embedding_provider=mock_embedding_provider,
        )

        embeddings = await labeler.embed_topics(Taxonomy())
        assert embeddings == {}


class TestTopicAssignment:
    """Tests for TopicAssignment dataclass."""

    def test_default_method(self) -> None:
        """Test default method is embedding."""
        assignment = TopicAssignment(
            note_id=1,
            topic_id=10,
            topic_path="test",
            confidence=0.9,
        )
        assert assignment.method == "embedding"


class TestTopicCoverage:
    """Tests for TopicCoverage dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        coverage = TopicCoverage(
            topic_id=1,
            path="test",
            label="Test",
        )
        assert coverage.note_count == 0
        assert coverage.mature_count == 0
        assert coverage.weak_notes == 0
        assert coverage.avg_confidence == 0.0


class TestTopicGap:
    """Tests for TopicGap dataclass."""

    def test_gap_types(self) -> None:
        """Test gap type values."""
        missing = TopicGap(
            topic_id=1,
            path="test",
            label="Test",
            description=None,
            gap_type="missing",
            note_count=0,
            threshold=5,
        )
        undercovered = TopicGap(
            topic_id=2,
            path="test2",
            label="Test2",
            description=None,
            gap_type="undercovered",
            note_count=2,
            threshold=5,
        )

        assert missing.gap_type == "missing"
        assert undercovered.gap_type == "undercovered"
