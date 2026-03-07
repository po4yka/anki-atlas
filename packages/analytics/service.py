"""Analytics service for topic coverage and gap analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class AnalyticsResult:
    """Result from analytics operations."""

    coverage: TopicCoverage | None = None
    gaps: list[TopicGap] | None = None
    weak_notes: list[WeakNote] | None = None


class AnalyticsService:
    """Service for topic analytics."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def load_taxonomy(self, yaml_path: Path | None = None) -> Taxonomy:
        """Load taxonomy from YAML or database, syncing to DB if from YAML."""
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
        """Label all notes with topics, loading taxonomy from DB if not provided."""
        if taxonomy is None:
            taxonomy = await load_taxonomy_from_database(self.settings)

        return await label_all_notes(taxonomy, self.settings, min_confidence)

    async def get_coverage(
        self,
        topic_path: str,
        include_subtree: bool = True,
    ) -> TopicCoverage | None:
        return await get_topic_coverage(topic_path, include_subtree, self.settings)

    async def get_gaps(
        self,
        topic_path: str,
        min_coverage: int = 1,
    ) -> list[TopicGap]:
        return await get_topic_gaps(topic_path, min_coverage, self.settings)

    async def get_weak_notes(
        self,
        topic_path: str,
        max_results: int = 20,
    ) -> list[WeakNote]:
        return await get_weak_notes(topic_path, max_results, settings=self.settings)

    async def get_full_analysis(
        self,
        topic_path: str,
        min_coverage: int = 1,
    ) -> AnalyticsResult:
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
        return await get_coverage_tree(root_path, self.settings)
