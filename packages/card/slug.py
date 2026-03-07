"""Service for generating and managing card slugs."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Final

MAX_COMPONENT_LENGTH: Final[int] = 50
MAX_SLUG_LENGTH: Final[int] = 100

SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9-]")
MULTI_HYPHEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"-+")


class SlugService:
    """Service for generating unique, deterministic slugs.

    Slugs follow the format: topic-keyword-index-lang
    (e.g., "python-decorators-1-en").
    """

    @staticmethod
    def slugify(text: str) -> str:
        """Convert text to URL-friendly slug."""
        if not text:
            return ""

        normalized = unicodedata.normalize("NFKD", text)
        ascii_text = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        lower_text = ascii_text.lower()
        with_hyphens = re.sub(r"[\s_./\\]+", "-", lower_text)
        clean_text = SLUG_PATTERN.sub("", with_hyphens)
        single_hyphens = MULTI_HYPHEN_PATTERN.sub("-", clean_text)
        stripped = single_hyphens.strip("-")

        if len(stripped) > MAX_COMPONENT_LENGTH:
            truncated = stripped[:MAX_COMPONENT_LENGTH]
            last_hyphen = truncated.rfind("-")
            if last_hyphen > MAX_COMPONENT_LENGTH // 2:
                truncated = truncated[:last_hyphen]
            stripped = truncated.rstrip("-")

        return stripped

    @staticmethod
    def compute_hash(content: str, length: int = 6) -> str:
        """Compute short SHA-256 hash of content."""
        if length < 1 or length > 64:
            raise ValueError(f"length must be between 1 and 64, got {length}")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:length]

    @classmethod
    def generate_slug(cls, topic: str, keyword: str, index: int, lang: str) -> str:
        """Generate slug: topic-keyword-index-lang."""
        if index < 0:
            raise ValueError(f"index must be non-negative, got {index}")
        if not lang or len(lang) != 2:
            raise ValueError(f"lang must be exactly 2 characters, got '{lang}'")

        topic_slug = cls.slugify(topic)
        keyword_slug = cls.slugify(keyword)

        if not topic_slug:
            topic_slug = "untitled"
        if not keyword_slug:
            keyword_slug = "card"

        slug = f"{topic_slug}-{keyword_slug}-{index}-{lang.lower()}"

        if len(slug) > MAX_SLUG_LENGTH:
            available = MAX_SLUG_LENGTH - len(f"-{index}-{lang}")
            half = available // 2 - 1
            topic_slug = topic_slug[:half].rstrip("-")
            keyword_slug = keyword_slug[:half].rstrip("-")
            slug = f"{topic_slug}-{keyword_slug}-{index}-{lang.lower()}"

        return slug

    @classmethod
    def generate_slug_base(cls, topic: str, keyword: str, index: int) -> str:
        """Generate slug base without language suffix."""
        if index < 0:
            raise ValueError(f"index must be non-negative, got {index}")

        topic_slug = cls.slugify(topic)
        keyword_slug = cls.slugify(keyword)

        if not topic_slug:
            topic_slug = "untitled"
        if not keyword_slug:
            keyword_slug = "card"

        return f"{topic_slug}-{keyword_slug}-{index}"

    @classmethod
    def generate_deterministic_guid(cls, slug: str, source_path: str) -> str:
        """Generate deterministic GUID for stable Anki sync."""
        if not slug:
            raise ValueError("slug cannot be empty")
        if not source_path:
            raise ValueError("source_path cannot be empty")

        combined = f"{slug}:{source_path}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]

    @classmethod
    def extract_components(cls, slug: str) -> dict[str, str | int]:
        """Extract components from a slug."""
        result: dict[str, str | int] = {
            "topic": "",
            "keyword": "",
            "index": 0,
            "lang": "",
        }

        if not slug:
            return result

        parts = slug.rsplit("-", 2)

        if len(parts) >= 1 and len(parts[-1]) == 2 and parts[-1].isalpha():
            result["lang"] = parts[-1]
            parts = parts[:-1]

        if len(parts) >= 1 and parts[-1].isdigit():
            result["index"] = int(parts[-1])
            parts = parts[:-1]

        if len(parts) >= 1:
            remaining = "-".join(parts)
            remaining_parts = remaining.split("-")
            if len(remaining_parts) >= 2:
                mid = len(remaining_parts) // 2
                result["topic"] = "-".join(remaining_parts[:mid])
                result["keyword"] = "-".join(remaining_parts[mid:])
            else:
                result["topic"] = remaining
                result["keyword"] = ""

        return result

    @staticmethod
    def is_valid_slug(slug: str) -> bool:
        """Check if a string is a valid slug."""
        if not slug or len(slug) < 3:
            return False
        if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", slug):
            return False
        if "--" in slug:
            return False
        return bool(re.search(r"-[a-z]{2}$", slug))

    @staticmethod
    def compute_content_hash(front: str, back: str) -> str:
        """Compute 12-character content hash from front and back."""
        content = f"{front.strip()}|{back.strip()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def compute_metadata_hash(note_type: str, tags: list[str]) -> str:
        """Compute 6-character metadata hash from note type and tags."""
        content = f"{note_type}|{','.join(sorted(tags))}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:6]
