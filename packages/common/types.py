"""Shared type definitions for Anki Atlas."""

from __future__ import annotations

import enum
from typing import NewType


class Language(enum.StrEnum):
    """Supported card languages."""

    EN = "en"
    RU = "ru"
    DE = "de"
    FR = "fr"
    ES = "es"
    IT = "it"
    PT = "pt"
    ZH = "zh"
    JA = "ja"
    KO = "ko"


SlugStr = NewType("SlugStr", str)
CardId = NewType("CardId", int)
NoteId = NewType("NoteId", int)
DeckName = NewType("DeckName", str)
