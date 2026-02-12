"""Unit tests for MigrationResult dataclass."""

from packages.common.database import MigrationResult


class TestMigrationResult:
    def test_default_empty_lists(self) -> None:
        result = MigrationResult()
        assert result.applied == []
        assert result.skipped == []

    def test_applied_field(self) -> None:
        result = MigrationResult(applied=["001_initial_schema"])
        assert result.applied == ["001_initial_schema"]
        assert result.skipped == []

    def test_skipped_field(self) -> None:
        result = MigrationResult(skipped=["001_initial_schema"])
        assert result.applied == []
        assert result.skipped == ["001_initial_schema"]

    def test_both_fields(self) -> None:
        result = MigrationResult(
            applied=["002_add_column"],
            skipped=["001_initial_schema"],
        )
        assert result.applied == ["002_add_column"]
        assert result.skipped == ["001_initial_schema"]

    def test_instances_do_not_share_lists(self) -> None:
        a = MigrationResult()
        b = MigrationResult()
        a.applied.append("001")
        assert b.applied == []
