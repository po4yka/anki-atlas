# Spec 22: Campaigns

## Goal

Migrate campaign configuration files for batch card generation into `config/campaigns/`.

## Source

- `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/algorithms.yaml`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/android.yaml`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/backend.yaml`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/compsci.yaml`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/kotlin.yaml`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/system-design.yaml`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/template.yaml`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/README.md`

## Target

### `config/campaigns/` directory:

Copy and adapt campaign YAML configs:
```
config/campaigns/
  algorithms.yaml
  android.yaml
  backend.yaml
  compsci.yaml
  kotlin.yaml
  system-design.yaml
  template.yaml
  README.md
```

Key changes in each campaign file:
- Update deck name references if needed
- Update tool/CLI references to anki-atlas
- Ensure tag prefixes match `packages/taxonomy/` conventions
- Keep campaign structure and content intact

## Acceptance Criteria

- [ ] All campaign files copied to `config/campaigns/`
- [ ] YAML files are valid (parseable)
- [ ] References updated for anki-atlas
- [ ] Template file serves as documentation for creating new campaigns
- [ ] `make check` passes
