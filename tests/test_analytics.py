"""Tests for analytics module (taxonomy, labeling, coverage, duplicates)."""

import tempfile
from pathlib import Path

import pytest

from packages.analytics.coverage import TopicCoverage, TopicGap
from packages.analytics.duplicates import (
    DuplicateCluster,
    DuplicateDetector,
    DuplicatePair,
    DuplicateStats,
)
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


class TestDuplicatePair:
    """Tests for DuplicatePair dataclass."""

    def test_pair_creation(self) -> None:
        """Test creating a duplicate pair."""
        pair = DuplicatePair(
            note_id_a=100,
            note_id_b=200,
            similarity=0.95,
        )
        assert pair.note_id_a == 100
        assert pair.note_id_b == 200
        assert pair.similarity == 0.95


class TestDuplicateCluster:
    """Tests for DuplicateCluster dataclass."""

    def test_cluster_size(self) -> None:
        """Test cluster size calculation."""
        cluster = DuplicateCluster(
            representative_id=1,
            representative_text="Test note content",
        )
        # Empty cluster - just representative
        assert cluster.size == 1

        # Add duplicates
        cluster.duplicates = [
            {"note_id": 2, "similarity": 0.95, "text": "Similar 1"},
            {"note_id": 3, "similarity": 0.92, "text": "Similar 2"},
        ]
        assert cluster.size == 3

    def test_default_values(self) -> None:
        """Test default values for cluster."""
        cluster = DuplicateCluster(
            representative_id=1,
            representative_text="Test",
        )
        assert cluster.duplicates == []
        assert cluster.deck_names == []
        assert cluster.tags == []


class TestDuplicateStats:
    """Tests for DuplicateStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = DuplicateStats()
        assert stats.notes_scanned == 0
        assert stats.clusters_found == 0
        assert stats.total_duplicates == 0
        assert stats.avg_cluster_size == 0.0


class TestDuplicateDetector:
    """Tests for DuplicateDetector class."""

    def test_cluster_duplicates_empty(self) -> None:
        """Test clustering with empty pairs."""
        detector = DuplicateDetector()
        clusters = detector._cluster_duplicates([])
        assert clusters == {}

    def test_cluster_duplicates_single_pair(self) -> None:
        """Test clustering with single pair."""
        detector = DuplicateDetector()
        pairs = [
            DuplicatePair(note_id_a=1, note_id_b=2, similarity=0.95),
        ]
        clusters = detector._cluster_duplicates(pairs)

        # Should have one cluster with root=1 (smaller ID)
        assert 1 in clusters
        assert len(clusters[1]) == 1
        assert clusters[1][0][0] == 2  # note_id
        assert clusters[1][0][1] == 0.95  # similarity

    def test_cluster_duplicates_transitive(self) -> None:
        """Test transitive clustering (A~B, B~C -> A,B,C in same cluster)."""
        detector = DuplicateDetector()
        pairs = [
            DuplicatePair(note_id_a=1, note_id_b=2, similarity=0.95),
            DuplicatePair(note_id_a=2, note_id_b=3, similarity=0.93),
        ]
        clusters = detector._cluster_duplicates(pairs)

        # All three should be in one cluster with root=1
        assert 1 in clusters
        member_ids = {m[0] for m in clusters[1]}
        assert member_ids == {2, 3}

    def test_cluster_duplicates_separate_clusters(self) -> None:
        """Test that unrelated pairs form separate clusters."""
        detector = DuplicateDetector()
        pairs = [
            DuplicatePair(note_id_a=1, note_id_b=2, similarity=0.95),
            DuplicatePair(note_id_a=10, note_id_b=20, similarity=0.93),
        ]
        clusters = detector._cluster_duplicates(pairs)

        # Should have two separate clusters
        assert len(clusters) == 2
        assert 1 in clusters
        assert 10 in clusters
