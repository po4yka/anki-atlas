#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


RUST_VERSION_DOCS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("docs/FIRST_TIME_SETUP.md"),
)

WORKSPACE_LAYOUT_DOCS = (
    Path("README.md"),
    Path("docs/ARCHITECTURE.md"),
)

CLI_INVENTORY_MARKERS = {
    Path("README.md"): "The CLI currently supports:",
    Path("docs/ARCHITECTURE.md"): "The CLI exposes:",
}

RUST_VERSION_PATTERN = re.compile(r"Rust\s+`?([0-9]+\.[0-9]+)\+`?")
SUBCOMMAND_COUNT_PATTERN = re.compile(r"\b\d+\s+subcommands?\b", re.IGNORECASE)
ENUM_PATTERN_TEMPLATE = r"pub enum {name}\s*\{{"
VARIANT_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)")


def load_workspace_metadata(root: Path) -> tuple[str, list[str]]:
    cargo_toml = root / "Cargo.toml"
    with cargo_toml.open("rb") as handle:
        cargo = tomllib.load(handle)

    workspace = cargo["workspace"]
    rust_version = workspace["package"]["rust-version"]
    members = workspace["members"]
    return rust_version, members


def camel_to_kebab(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def extract_block(text: str, marker: str) -> str:
    match = re.search(ENUM_PATTERN_TEMPLATE.format(name=re.escape(marker)), text)
    if not match:
        raise ValueError(f"could not find enum {marker}")

    start = match.end()
    depth = 1
    index = start
    while index < len(text) and depth > 0:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1

    if depth != 0:
        raise ValueError(f"unbalanced braces while parsing enum {marker}")

    return text[start : index - 1]


def parse_variants(enum_body: str) -> list[str]:
    variants: list[str] = []
    for line in enum_body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#[") or stripped.startswith("//"):
            continue
        match = VARIANT_PATTERN.match(stripped)
        if match:
            variants.append(match.group(1))
    return variants


def parse_cli_inventory(args_rs_text: str) -> list[str]:
    commands = parse_variants(extract_block(args_rs_text, "Commands"))
    topics = parse_variants(extract_block(args_rs_text, "TopicsCommand"))

    inventory: list[str] = []
    for command in commands:
        if command == "Topics":
            inventory.extend(f"topics {camel_to_kebab(topic)}" for topic in topics)
        else:
            inventory.append(camel_to_kebab(command))
    return inventory


def extract_bullets_after_marker(text: str, marker: str) -> list[str]:
    marker_index = text.find(marker)
    if marker_index == -1:
        raise ValueError(f"could not find marker {marker!r}")

    bullets: list[str] = []
    started = False
    for line in text[marker_index + len(marker) :].splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            started = True
            bullets.append(stripped[2:].strip().strip("`"))
            continue
        if started and stripped:
            break
        if started and not stripped:
            break

    if not bullets:
        raise ValueError(f"no bullets found after marker {marker!r}")
    return bullets


def extract_workspace_entries(text: str) -> set[str]:
    marker = "## Workspace Layout"
    marker_index = text.find(marker)
    if marker_index == -1:
        raise ValueError("could not find workspace layout heading")

    fenced = re.search(r"```text\n(.*?)\n```", text[marker_index:], re.DOTALL)
    if not fenced:
        raise ValueError("could not find workspace layout code block")

    block = fenced.group(1)
    entries: set[str] = set()
    section: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "bins/":
            section = "bins"
            continue
        if stripped == "crates/":
            section = "crates"
            continue
        if section is None or not raw_line.startswith("  "):
            continue

        entry = stripped.split("#", 1)[0].strip()
        if not entry.endswith("/"):
            continue
        entries.add(f"{section}/{entry[:-1]}")

    if not entries:
        raise ValueError("workspace layout code block did not contain any entries")
    return entries


def run_checks(root: Path) -> list[str]:
    rust_version, workspace_members = load_workspace_metadata(root)
    expected_cli = parse_cli_inventory((root / "bins/cli/src/args.rs").read_text())
    errors: list[str] = []

    for relative_path in RUST_VERSION_DOCS:
        text = (root / relative_path).read_text()
        found_versions = RUST_VERSION_PATTERN.findall(text)
        if not found_versions:
            errors.append(
                f"{relative_path}: missing Rust version declaration matching workspace rust-version"
            )
            continue
        mismatches = sorted({version for version in found_versions if version != rust_version})
        if mismatches:
            errors.append(
                f"{relative_path}: Rust version mismatch; expected {rust_version}+ but found {', '.join(mismatches)}+"
            )

    for relative_path in WORKSPACE_LAYOUT_DOCS:
        text = (root / relative_path).read_text()
        documented = extract_workspace_entries(text)
        expected = set(workspace_members)
        missing = sorted(expected - documented)
        unexpected = sorted(documented - expected)
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected {', '.join(unexpected)}")
            errors.append(f"{relative_path}: workspace layout mismatch; {'; '.join(details)}")

    for relative_path, marker in CLI_INVENTORY_MARKERS.items():
        text = (root / relative_path).read_text()
        documented = extract_bullets_after_marker(text, marker)
        if documented != expected_cli:
            errors.append(
                f"{relative_path}: CLI inventory mismatch; expected {expected_cli} but found {documented}"
            )

    agents_text = (root / "AGENTS.md").read_text()
    if SUBCOMMAND_COUNT_PATTERN.search(agents_text):
        errors.append("AGENTS.md: remove hardcoded CLI subcommand counts")

    return errors


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    errors = run_checks(root)
    if not errors:
        print("documentation consistency checks passed")
        return 0

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
