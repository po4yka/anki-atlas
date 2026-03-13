# anki-conventions - Quick Reference

Mastery-oriented Anki card patterns, query syntax, FSRS settings.

## Core Rules

- **Mastery**: why/when/how, not "what is X". Test understanding, not recall.
- **Bilingual**: EN + RU cards (Cyrillic only, no transliteration).
- **Atomic**: one concept/card (relaxed for code -- needs context).
- **Tags**: `prefix::topic` kebab-case, max 2 levels. Required: `difficulty::` + topic tag.
- **Prefixes**: `android::`, `kotlin::`, `cs::`, `topic::`, `difficulty::`, `lang::`, `source::`, `context::`
- **FSRS**: retention 0.90, learning step 15m/30m, optimize monthly.

## Query Syntax

`deck:Name`, `tag:tagname`, `is:due`, `is:new`, `prop:due<0`. Combine with spaces (AND) or `OR`.

## MCP Tools

`mcp__anki__addNote`, `findNotes`, `notesInfo`, `updateNoteFields`, `deleteNotes`

## Refs

`docs/anki/card-model.md`, `docs/anki/tag-taxonomy.md`, `docs/anki/deck-naming.md`
