"""Tests for packages/analytics/taxonomy.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from packages.analytics.taxonomy import (
    Taxonomy,
    Topic,
    load_taxonomy_from_yaml,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestTopic:
    """Tests for Topic dataclass properties."""

    def test_parent_path_root(self) -> None:
        t = Topic(path="math", label="Math")
        assert t.parent_path is None

    def test_parent_path_nested(self) -> None:
        t = Topic(path="math/calculus", label="Calculus")
        assert t.parent_path == "math"

    def test_parent_path_deep(self) -> None:
        t = Topic(path="a/b/c", label="C")
        assert t.parent_path == "a/b"

    def test_depth_root(self) -> None:
        t = Topic(path="math", label="Math")
        assert t.depth == 0

    def test_depth_nested(self) -> None:
        t = Topic(path="a/b/c", label="C")
        assert t.depth == 2

    def test_name(self) -> None:
        t = Topic(path="programming/python", label="Python")
        assert t.name == "python"

    def test_name_root(self) -> None:
        t = Topic(path="math", label="Math")
        assert t.name == "math"


class TestTaxonomy:
    """Tests for Taxonomy tree operations."""

    def _build_sample(self) -> Taxonomy:
        root = Topic(path="prog", label="Programming")
        child = Topic(path="prog/py", label="Python")
        grandchild = Topic(path="prog/py/async", label="Async")
        child.children.append(grandchild)
        root.children.append(child)
        other = Topic(path="math", label="Math")

        return Taxonomy(
            topics={
                "prog": root,
                "prog/py": child,
                "prog/py/async": grandchild,
                "math": other,
            },
            roots=[root, other],
        )

    def test_get_found(self) -> None:
        tax = self._build_sample()
        assert tax.get("prog/py") is not None
        assert tax.get("prog/py").label == "Python"  # type: ignore[union-attr]

    def test_get_not_found(self) -> None:
        tax = self._build_sample()
        assert tax.get("nonexistent") is None

    def test_all_topics_depth_first(self) -> None:
        tax = self._build_sample()
        paths = [t.path for t in tax.all_topics()]
        assert paths == ["prog", "prog/py", "prog/py/async", "math"]

    def test_subtree(self) -> None:
        tax = self._build_sample()
        sub = tax.subtree("prog")
        paths = sorted(t.path for t in sub)
        assert paths == ["prog", "prog/py", "prog/py/async"]

    def test_subtree_no_match(self) -> None:
        tax = self._build_sample()
        assert tax.subtree("nonexistent") == []


class TestLoadTaxonomyFromYaml:
    """Tests for YAML loading."""

    def test_valid_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
topics:
  - path: programming
    label: Programming
    description: General programming
    children:
      - path: programming/python
        label: Python
        children:
          - path: programming/python/async
            label: Async Programming
  - path: math
    label: Mathematics
"""
        yaml_file = tmp_path / "topics.yml"
        yaml_file.write_text(yaml_content)

        tax = load_taxonomy_from_yaml(yaml_file)

        assert len(tax.topics) == 4
        assert len(tax.roots) == 2
        assert tax.get("programming/python/async") is not None
        assert tax.get("programming").description == "General programming"  # type: ignore[union-attr]

    def test_missing_file(self, tmp_path: Path) -> None:
        tax = load_taxonomy_from_yaml(tmp_path / "nonexistent.yml")
        assert len(tax.topics) == 0
        assert len(tax.roots) == 0

    def test_empty_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "empty.yml"
        yaml_file.write_text("")
        tax = load_taxonomy_from_yaml(yaml_file)
        assert len(tax.topics) == 0

    def test_no_topics_key(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "bad.yml"
        yaml_file.write_text("other_key: value\n")
        tax = load_taxonomy_from_yaml(yaml_file)
        assert len(tax.topics) == 0
