"""Topic taxonomy management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from packages.common.config import Settings, get_settings
from packages.common.database import get_connection


@dataclass
class Topic:
    """A topic in the taxonomy."""

    path: str  # e.g., "programming/python/async"
    label: str  # Human-readable label
    description: str | None = None
    topic_id: int | None = None  # Database ID (set after insert)
    children: list["Topic"] = field(default_factory=list)

    @property
    def parent_path(self) -> str | None:
        """Get parent topic path."""
        if "/" not in self.path:
            return None
        return "/".join(self.path.split("/")[:-1])

    @property
    def depth(self) -> int:
        """Get depth in taxonomy tree."""
        return self.path.count("/")

    @property
    def name(self) -> str:
        """Get the topic name (last segment of path)."""
        return self.path.split("/")[-1]


@dataclass
class Taxonomy:
    """Complete topic taxonomy."""

    topics: dict[str, Topic] = field(default_factory=dict)  # path -> Topic
    roots: list[Topic] = field(default_factory=list)

    def get(self, path: str) -> Topic | None:
        """Get topic by path."""
        return self.topics.get(path)

    def all_topics(self) -> list[Topic]:
        """Get all topics in depth-first order."""
        result: list[Topic] = []

        def visit(topic: Topic) -> None:
            result.append(topic)
            for child in topic.children:
                visit(child)

        for root in self.roots:
            visit(root)
        return result

    def subtree(self, path: str) -> list[Topic]:
        """Get all topics under a given path (including itself)."""
        result: list[Topic] = []
        for topic_path, topic in self.topics.items():
            if topic_path == path or topic_path.startswith(path + "/"):
                result.append(topic)
        return result


def load_taxonomy_from_yaml(yaml_path: Path) -> Taxonomy:
    """Load taxonomy from a YAML file.

    Expected format:
    ```yaml
    topics:
      - path: programming
        label: Programming
        description: General programming concepts
        children:
          - path: programming/python
            label: Python
            children:
              - path: programming/python/async
                label: Async Programming
    ```

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Loaded Taxonomy.
    """
    if not yaml_path.exists():
        return Taxonomy()

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    if not data or "topics" not in data:
        return Taxonomy()

    taxonomy = Taxonomy()
    _parse_topics(data["topics"], taxonomy)
    return taxonomy


def _parse_topics(
    items: list[dict[str, Any]],
    taxonomy: Taxonomy,
    parent: Topic | None = None,
) -> None:
    """Recursively parse topic items."""
    for item in items:
        path = item.get("path", "")
        if not path:
            continue

        topic = Topic(
            path=path,
            label=item.get("label", path.split("/")[-1]),
            description=item.get("description"),
        )

        taxonomy.topics[path] = topic

        if parent:
            parent.children.append(topic)
        else:
            taxonomy.roots.append(topic)

        # Parse children
        children = item.get("children", [])
        if children:
            _parse_topics(children, taxonomy, topic)


async def sync_taxonomy_to_database(
    taxonomy: Taxonomy,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Sync taxonomy to database.

    Args:
        taxonomy: Taxonomy to sync.
        settings: Application settings.

    Returns:
        Dictionary mapping path to topic_id.
    """
    settings = settings or get_settings()
    path_to_id: dict[str, int] = {}

    async with get_connection(settings) as conn:
        for topic in taxonomy.all_topics():
            # Upsert topic
            result = await conn.execute(
                """
                INSERT INTO topics (path, label, description)
                VALUES (%(path)s, %(label)s, %(description)s)
                ON CONFLICT (path) DO UPDATE SET
                    label = EXCLUDED.label,
                    description = EXCLUDED.description
                RETURNING topic_id
                """,
                {
                    "path": topic.path,
                    "label": topic.label,
                    "description": topic.description,
                },
            )
            row = await result.fetchone()
            if row:
                topic.topic_id = row["topic_id"]
                path_to_id[topic.path] = row["topic_id"]

        await conn.commit()

    return path_to_id


async def load_taxonomy_from_database(
    settings: Settings | None = None,
) -> Taxonomy:
    """Load taxonomy from database.

    Args:
        settings: Application settings.

    Returns:
        Loaded Taxonomy.
    """
    settings = settings or get_settings()
    taxonomy = Taxonomy()

    async with get_connection(settings) as conn:
        result = await conn.execute(
            "SELECT topic_id, path, label, description FROM topics ORDER BY path"
        )

        topics_list: list[Topic] = []
        async for row in result:
            topic = Topic(
                topic_id=row["topic_id"],
                path=row["path"],
                label=row["label"],
                description=row["description"],
            )
            taxonomy.topics[topic.path] = topic
            topics_list.append(topic)

    # Build tree structure
    for topic in topics_list:
        parent_path = topic.parent_path
        if parent_path and parent_path in taxonomy.topics:
            taxonomy.topics[parent_path].children.append(topic)
        elif not parent_path:
            taxonomy.roots.append(topic)

    return taxonomy


async def get_topic_by_path(
    path: str,
    settings: Settings | None = None,
) -> Topic | None:
    """Get a single topic by path.

    Args:
        path: Topic path.
        settings: Application settings.

    Returns:
        Topic or None if not found.
    """
    settings = settings or get_settings()

    async with get_connection(settings) as conn:
        result = await conn.execute(
            "SELECT topic_id, path, label, description FROM topics WHERE path = %(path)s",
            {"path": path},
        )
        row = await result.fetchone()
        if row:
            return Topic(
                topic_id=row["topic_id"],
                path=row["path"],
                label=row["label"],
                description=row["description"],
            )
    return None
