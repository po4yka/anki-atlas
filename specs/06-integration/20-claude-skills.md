# Spec 20: Claude Code Skills

## Goal

Migrate Claude Code skills from both source repos into anki-atlas's `.claude/skills/`.

## Source

claude-code-obsidian-anki skills (9 skills):
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/analyze-note/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/bulk-process/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/cleanup-cards/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/detect-changes/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/find-gaps/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/generate-cards/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/review-cards/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/show-stats/SKILL.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/sync-cards/SKILL.md`

Shared references:
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/_shared/card-model.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/_shared/cli-reference.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/_shared/deck-naming.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/_shared/tag-taxonomy.md`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/.claude/skills/_shared/thresholds.md`

ai-agent-anki skill:
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/skills/anki-conventions/SKILL.md`
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/skills/anki-conventions/references/*.md` (9 reference files)

## Target

### `.claude/skills/` directory:

Copy and adapt skills. Key changes in each skill:
- Update CLI references: use `anki-atlas` CLI commands
- Update MCP tool references: use `ankiatlas_*` tool names
- Update import paths if skills reference code
- Merge overlapping content (e.g., tag taxonomy exists in both)

Structure:
```
.claude/skills/
  analyze-note/SKILL.md
  bulk-process/SKILL.md
  cleanup-cards/SKILL.md
  detect-changes/SKILL.md
  find-gaps/SKILL.md
  generate-cards/SKILL.md
  review-cards/SKILL.md
  show-stats/SKILL.md
  sync-cards/SKILL.md
  anki-conventions/
    SKILL.md
    references/
      card-maintenance.md
      card-patterns.md
      deck-organization.md
      fsrs-settings.md
      note-types.md
      programming-cards.md
      query-syntax.md
      tag-conventions.md
      troubleshooting.md
  _shared/
    card-model.md
    cli-reference.md    # Updated for anki-atlas CLI
    deck-naming.md
    tag-taxonomy.md
    thresholds.md
```

## Acceptance Criteria

- [ ] All skills copied and adapted to anki-atlas
- [ ] CLI references updated to `anki-atlas` commands
- [ ] MCP tool references updated to `ankiatlas_*` names
- [ ] No references to old repo paths or module names
- [ ] Shared references updated for unified repo
- [ ] `make check` passes (skills are markdown, won't affect Python checks)
