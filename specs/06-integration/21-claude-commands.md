# Spec 21: Claude Code Commands

## Goal

Migrate Claude Code slash commands from ai-agent-anki into anki-atlas's `.claude/commands/`.

## Source

- `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/create-card.md`
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/improve-card.md`
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/sync-vault.md`
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/review-session.md`
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/deck-stats.md`
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/search-cards.md`
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/tag-audit.md`

## Target

### `.claude/commands/anki/` directory:

Copy and adapt all 7 commands. Key changes:
- Update CLI references: `anki-atlas generate`, `anki-atlas validate`, etc.
- Update MCP tool names: `ankiatlas_generate`, `ankiatlas_validate`, etc.
- Update any file paths to reference anki-atlas's structure
- Keep the same command names for user familiarity

```
.claude/commands/anki/
  create-card.md
  improve-card.md
  sync-vault.md
  review-session.md
  deck-stats.md
  search-cards.md
  tag-audit.md
```

## Acceptance Criteria

- [ ] All 7 commands copied to `.claude/commands/anki/`
- [ ] CLI and MCP references updated for anki-atlas
- [ ] No references to old repo paths
- [ ] Commands work with Claude Code (valid markdown format)
- [ ] `make check` passes
