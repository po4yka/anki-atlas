# Topic labeling, coverage scoring, dedupe

from packages.analytics.coverage import (
    TopicCoverage,
    TopicGap,
    WeakNote,
    get_coverage_tree,
    get_topic_coverage,
    get_topic_gaps,
    get_weak_notes,
)
from packages.analytics.labeling import (
    LabelingStats,
    TopicAssignment,
    TopicLabeler,
    label_all_notes,
)
from packages.analytics.service import AnalyticsResult, AnalyticsService
from packages.analytics.taxonomy import (
    Taxonomy,
    Topic,
    get_topic_by_path,
    load_taxonomy_from_database,
    load_taxonomy_from_yaml,
    sync_taxonomy_to_database,
)

__all__ = [
    "AnalyticsResult",
    "AnalyticsService",
    "LabelingStats",
    "Taxonomy",
    "Topic",
    "TopicAssignment",
    "TopicCoverage",
    "TopicGap",
    "TopicLabeler",
    "WeakNote",
    "get_coverage_tree",
    "get_topic_by_path",
    "get_topic_coverage",
    "get_topic_gaps",
    "get_weak_notes",
    "label_all_notes",
    "load_taxonomy_from_database",
    "load_taxonomy_from_yaml",
    "sync_taxonomy_to_database",
]
