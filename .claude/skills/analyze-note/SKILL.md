---
name: analyze-note
description: Analyzes Obsidian note structure, identifies card-worthy topics, and assesses quality before flashcard creation. Use to prepare notes or assess card potential.
argument-hint: "<note-path>"
allowed-tools: Read, Glob, Grep, Bash
---

# Analyze Note for Card Creation

Analyze a note's structure and content to assess card-worthiness before creating flashcards.

**Card Model**: See [card-model.md](../_shared/card-model.md). Estimated cards = topics x 2 (EN + RU).

## When to Use

- User asks to "analyze a note" or "review note quality"
- User wants to "prepare a note for cards" or "assess card potential"
- User asks "what cards can I make from this note?"
- Before bulk card generation

## Workflow

### Step 1: Load Note

```bash
uv run anki-atlas generate path/to/note.md --dry-run
```

Or use Read tool directly. Examine:
- Frontmatter metadata (id, title, topic, tags)
- Section structure (headers, Q&A format)
- Content depth and clarity

### Step 2: Analyze Quality

**Structure Quality**:
- [ ] Has clear Q&A sections?
- [ ] Uses blockquotes for questions?
- [ ] Answers well-organized?
- [ ] Has follow-up questions?

**Content Depth**:
- [ ] Answers provide enough detail?
- [ ] Code examples where relevant?
- [ ] Explanations clear and concise?

### Step 3: Assess Cognitive Load

| Level | Content | Strategy |
|-------|---------|----------|
| LOW | Single concept | 1 card/lang |
| MEDIUM | Code, 2-3 facts | 2-3 cards, scaffolded |
| HIGH | Multiple concepts | 4+ cards, split prerequisites |

### Step 4: Identify Card-Worthy Topics

Select concepts worth making cards for (typically 3-10):

| Priority | What to Include |
|----------|-----------------|
| HIGH | Core definitions, key distinctions, patterns |
| MEDIUM | Supporting examples, edge cases |
| SKIP | Obvious facts, implementation details |

### Step 5: Check Existing Cards

```bash
uv run anki-atlas search "topic from note" --semantic --top 10
```

### Step 6: Generate Report

```
Analysis of notes/python-decorators.md
======================================

Note Quality: 8/10 (GOOD)

Structure:
- [OK] Has frontmatter
- [OK] Clear Q&A format
- [OK] Code examples
- [OK] Bilingual content
- [MISSING] Follow-up questions

Card-Worthy Topics (5 identified):
1. [HIGH] Definition - 2 cards needed (EN + RU)
2. [HIGH] @syntax - 2 cards needed
3. [MEDIUM] Use cases - 2 cards needed
4. [MEDIUM] functools.wraps - 2 cards needed
5. [LOW] Class decorators - needs more detail

Existing Cards: 4 found (2 topics covered)
New Cards Needed: 6 (3 topics x 2 languages)

Recommendations:
1. Add follow-up questions section
2. Expand class decorators explanation
3. Create cards for uncovered topics
```

## Quality Scores

| Score | Quality | Action |
|-------|---------|--------|
| 9-10 | EXCELLENT | Ready for cards |
| 7-8 | GOOD | Minor fixes needed |
| 5-6 | FAIR | Improve before cards |
| 3-4 | POOR | Major revision needed |
| 1-2 | INCOMPLETE | Stub/draft |

## Next Steps

Based on analysis:
- **Good quality?** -> `/generate-cards`
- **Needs improvement?** -> Suggest specific fixes
- **Has existing cards?** -> `/review-cards`
- **Multiple notes?** -> `/bulk-process`

## Related

- `/generate-cards` - Create cards after analysis
- `/find-gaps` - Check coverage across vault
- [card-model.md](../_shared/card-model.md) - Card requirements
