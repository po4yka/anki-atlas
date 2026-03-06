# Spec 19: MCP Tools

## Goal

Add new MCP tools to the existing `apps/mcp/` for card generation, validation, and obsidian sync.

## Source

claude-code-obsidian-anki skills (inform tool design):
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/generate-cards/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/review-cards/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/sync-cards/SKILL.md`

ai-agent-anki MCP patterns:
- `/Users/npochaev/GitHub/ai-agent-anki/.mcp.json` -- MCP server config
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/` -- command patterns

Existing in anki-atlas:
- `apps/mcp/tools.py` -- existing MCP tool definitions (12KB)
- `apps/mcp/server.py` -- MCP server setup
- `apps/mcp/formatters.py` -- response formatting

## Target

### `apps/mcp/tools.py` (EXTEND)

Add new MCP tools:

1. **`ankiatlas_generate`** -- Generate flashcards from text
   - Input: text content, optional model, deck, language
   - Output: generated cards with quality scores
   - Uses: `packages/generator/`, `packages/validation/`

2. **`ankiatlas_validate`** -- Validate existing cards
   - Input: card IDs or deck name
   - Output: validation results with suggestions
   - Uses: `packages/validation/`

3. **`ankiatlas_obsidian_sync`** -- Sync Obsidian vault
   - Input: vault path, optional filters
   - Output: sync results (created, updated, skipped)
   - Uses: `packages/obsidian/sync`

4. **`ankiatlas_tag_audit`** -- Audit tag taxonomy
   - Input: optional deck filter
   - Output: violations and suggestions
   - Uses: `packages/taxonomy/`

Follow the existing tool registration pattern in `apps/mcp/tools.py`.

## Acceptance Criteria

- [ ] Four new tools registered in the MCP server
- [ ] Tools follow existing patterns in `apps/mcp/tools.py`
- [ ] Tools use `apps/mcp/formatters.py` for response formatting
- [ ] Existing MCP tools still work
- [ ] `from apps.mcp.tools import ...` includes new tools
- [ ] Tests in `tests/test_mcp_new.py` cover: tool registration, input validation, mock execution
- [ ] `make check` passes
