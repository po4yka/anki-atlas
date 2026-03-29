from __future__ import annotations

import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "check_doc_consistency.py"
SPEC = importlib.util.spec_from_file_location("check_doc_consistency", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

EXPECTED_CLI = [
    "version",
    "migrate",
    "tui",
    "sync",
    "index",
    "search",
    "topics tree",
    "topics load",
    "topics label",
    "coverage",
    "gaps",
    "weak-notes",
    "duplicates",
    "generate",
    "validate",
    "obsidian-sync",
    "tag-audit",
]

WORKSPACE_MEMBERS = [
    "crates/common",
    "crates/surface-contracts",
    "crates/surface-runtime",
    "crates/perf-support",
    "bins/api",
    "bins/cli",
    "bins/mcp",
    "bins/perf-harness",
    "bins/worker",
]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n")


def render_cli_inventory(items: list[str]) -> str:
    return "\n".join(f"- `{item}`" for item in items)


def render_workspace_layout(members: list[str]) -> str:
    crates = [member.split("/", 1)[1] for member in members if member.startswith("crates/")]
    bins = [member.split("/", 1)[1] for member in members if member.startswith("bins/")]
    lines = ["```text", "bins/"]
    lines.extend(f"  {entry}/" for entry in bins)
    lines.append("crates/")
    lines.extend(f"  {entry}/" for entry in crates)
    lines.append("```")
    return "\n".join(lines)


def create_repo(root: Path) -> None:
    write_text(
        root / "Cargo.toml",
        """
        [workspace]
        members = [
          "crates/common",
          "crates/surface-contracts",
          "crates/surface-runtime",
          "crates/perf-support",
          "bins/api",
          "bins/cli",
          "bins/mcp",
          "bins/perf-harness",
          "bins/worker",
        ]

        [workspace.package]
        rust-version = "1.88"
        """,
    )

    write_text(
        root / "bins/cli/src/args.rs",
        """
        pub enum Commands {
            Version,
            Migrate,
            Tui,
            Sync(SyncArgs),
            Index(IndexArgs),
            Search(SearchArgs),
            Topics(TopicsArgs),
            Coverage(CoverageArgs),
            Gaps(GapsArgs),
            WeakNotes(WeakNotesArgs),
            Duplicates(DuplicatesArgs),
            Generate(GenerateArgs),
            Validate(ValidateArgs),
            ObsidianSync(ObsidianSyncArgs),
            TagAudit(TagAuditArgs),
        }

        pub enum TopicsCommand {
            Tree(TopicsTreeArgs),
            Load(TopicsLoadArgs),
            Label(TopicsLabelArgs),
        }
        """,
    )

    write_text(
        root / "AGENTS.md",
        """
        # anki-atlas

        - Rust 1.88+ (edition 2024)
        - CLI: see README.md#cli-surface for the current command inventory
        """,
    )

    write_text(
        root / "README.md",
        f"""
        # Anki Atlas

        - Rust `1.88+`

        ## CLI Surface

        The CLI currently supports:

        {render_cli_inventory(EXPECTED_CLI)}

        ## Workspace Layout

        {render_workspace_layout(WORKSPACE_MEMBERS)}
        """,
    )

    write_text(
        root / "docs/FIRST_TIME_SETUP.md",
        """
        # First Time Setup

        - Rust `1.88+`
        """,
    )

    write_text(
        root / "docs/ARCHITECTURE.md",
        f"""
        # Architecture

        ## Workspace Layout

        {render_workspace_layout(WORKSPACE_MEMBERS)}

        ### CLI and TUI

        The CLI exposes:

        {render_cli_inventory(EXPECTED_CLI)}
        """,
    )


class DocConsistencyTests(unittest.TestCase):
    def create_root(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        create_repo(root)
        return root

    def test_consistent_repo_passes(self) -> None:
        root = self.create_root()
        self.assertEqual(MODULE.run_checks(root), [])

    def test_version_mismatch_is_reported(self) -> None:
        root = self.create_root()
        write_text(
            root / "README.md",
            (root / "README.md").read_text().replace("1.88+", "1.87+"),
        )

        errors = MODULE.run_checks(root)
        self.assertTrue(any("README.md: Rust version mismatch" in error for error in errors))

    def test_missing_workspace_member_is_reported(self) -> None:
        root = self.create_root()
        write_text(
            root / "docs/ARCHITECTURE.md",
            (root / "docs/ARCHITECTURE.md")
            .read_text()
            .replace("  surface-contracts/\n", ""),
        )

        errors = MODULE.run_checks(root)
        self.assertTrue(
            any(
                "docs/ARCHITECTURE.md: workspace layout mismatch" in error
                and "crates/surface-contracts" in error
                for error in errors
            )
        )

    def test_stale_cli_inventory_is_reported(self) -> None:
        root = self.create_root()
        write_text(
            root / "README.md",
            (root / "README.md").read_text().replace("- `tui`\n", ""),
        )

        errors = MODULE.run_checks(root)
        self.assertTrue(any("README.md: CLI inventory mismatch" in error for error in errors))

    def test_hardcoded_subcommand_count_is_reported(self) -> None:
        root = self.create_root()
        write_text(
            root / "AGENTS.md",
            """
            # anki-atlas

            - Rust 1.88+ (edition 2024)
            - CLI has 12 subcommands
            """,
        )

        errors = MODULE.run_checks(root)
        self.assertIn("AGENTS.md: remove hardcoded CLI subcommand counts", errors)


if __name__ == "__main__":
    unittest.main()
