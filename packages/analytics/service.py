"""Analytics service for topic coverage and gap analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packages.analytics.coverage import (
    TopicCoverage,
    TopicGap,
    WeakNote,
    get_coverage_tree,
    get_topic_coverage,
    get_topic_gaps,
    get_weak_notes,
)
from packages.analytics.labeling import LabelingStats, label_all_notes
from packages.analytics.taxonomy import (
    Taxonomy,
    load_taxonomy_from_database,
    load_taxonomy_from_yaml,
    sync_taxonomy_to_database,
)
from packages.common.config import Settings, get_settings


@dataclass
class AnalyticsResult:
    """Result from analytics operations."""

    coverage: TopicCoverage | None = None
    gaps: list[TopicGap] | None = None
    weak_notes: list[WeakNote] | None = None


class AnalyticsService:
    """Service for topic analytics."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize analytics service.

        Args:
            settings: Application settings.
        """
        self.settings = settings or get_settings()

    async def load_taxonomy(self, yaml_path: Path | None = None) -> Taxonomy:
        """Load taxonomy from YAML file or database.

        Args:
            yaml_path: Path to YAML file. If None, loads from database.

        Returns:
            Loaded taxonomy.
        """
        if yaml_path:
            taxonomy = load_taxonomy_from_yaml(yaml_path)
            # Sync to database
            await sync_taxonomy_to_database(taxonomy, self.settings)
            # Reload to get IDs
            return await load_taxonomy_from_database(self.settings)
        return await load_taxonomy_from_database(self.settings)

    async def label_notes(
        self,
        taxonomy: Taxonomy | None = None,
        min_confidence: float = 0.3,
    ) -> LabelingStats:
        """Label all notes with topics.

        Args:
            taxonomy: Taxonomy to use. If None, loads from database.
            min_confidence: Minimum confidence threshold.

        Returns:
            Labeling statistics.
        """
        if taxonomy is None:
            taxonomy = await load_taxonomy_from_database(self.settings)

        return await label_all_notes(taxonomy, self.settings, min_confidence)

    async def get_coverage(
        self,
        topic_path: str,
        include_subtree: bool = True,
    ) -> TopicCoverage | None:
        """Get coverage metrics for a topic.

        Args:
            topic_path: Topic path.
            include_subtree: Include child topics.

        Returns:
            Coverage metrics or None.
        """
        return await get_topic_coverage(topic_path, include_subtree, self.settings)

    async def get_gaps(
        self,
        topic_path: str,
        min_coverage: int = 1,
    ) -> list[TopicGap]:
        """Get gaps in topic coverage.

        Args:
            topic_path: Root topic path.
            min_coverage: Minimum notes for coverage.

        Returns:
            List of gaps.
        """
        return await get_topic_gaps(topic_path, min_coverage, self.settings)

    async def get_weak_notes(
        self,
        topic_path: str,
        max_results: int = 20,
    ) -> list[WeakNote]:
        """Get weak notes in a topic.

        Args:
            topic_path: Topic path.
            max_results: Maximum results.

        Returns:
            List of weak notes.
        """
        return await get_weak_notes(topic_path, max_results, settings=self.settings)

    async def get_full_analysis(
        self,
        topic_path: str,
        min_coverage: int = 1,
    ) -> AnalyticsResult:
        """Get full analysis for a topic.

        Args:
            topic_path: Topic path.
            min_coverage: Minimum coverage threshold.

        Returns:
            Complete analytics result.
        """
        coverage = await self.get_coverage(topic_path)
        gaps = await self.get_gaps(topic_path, min_coverage)
        weak = await self.get_weak_notes(topic_path)

        return AnalyticsResult(
            coverage=coverage,
            gaps=gaps,
            weak_notes=weak,
        )

    async def get_taxonomy_tree(
        self,
        root_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get taxonomy tree with coverage info.

        Args:
            root_path: Optional root to filter.

        Returns:
            List of topics with coverage data.
        """
        return await get_coverage_tree(root_path, self.settings)
