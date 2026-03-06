# Spec 02: Common Package Extensions

## Goal

Add shared types and exceptions to `packages/common/` that will be used by migrated code.

## Source

- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/domain/entities/card.py` -- `VALID_LANGUAGES` frozenset, `VALID_NOTE_TYPES` frozenset
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/exceptions.py` -- exception hierarchy
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/domain/services/slug_service.py` -- slug patterns

## Target

### `packages/common/types.py` (NEW)

Add shared type definitions:

```python
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
```

### `packages/common/exceptions.py` (EXTEND existing)

Add new exception classes to the existing hierarchy:

- `CardGenerationError(AnkiAtlasError)` -- card generation failures
- `CardValidationError(AnkiAtlasError)` -- card validation failures
- `ProviderError(AnkiAtlasError)` -- LLM provider failures
- `ObsidianParseError(AnkiAtlasError)` -- Obsidian note parsing failures
- `SyncConflictError(AnkiAtlasError)` -- sync conflict errors
- `AnkiConnectError(AnkiAtlasError)` -- AnkiConnect communication errors

Read the existing `packages/common/exceptions.py` first to understand the current hierarchy and base class name.

## Acceptance Criteria

- [ ] `packages/common/types.py` exists with `Language`, `SlugStr`, `CardId`, `NoteId`, `DeckName`
- [ ] `packages/common/exceptions.py` extended with 6 new exception classes
- [ ] New types importable: `from packages.common.types import Language, SlugStr, CardId`
- [ ] New exceptions importable: `from packages.common.exceptions import CardGenerationError`
- [ ] Update `packages/common/__init__.py` to re-export new types and exceptions
- [ ] `make check` passes
