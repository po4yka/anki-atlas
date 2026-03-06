"""Card generation, splitting, and context enrichment prompts.

Each stage has a system prompt (instructions for the LLM) and a user
prompt function (formats caller-supplied parameters into the request).

Adapted from claude-code-obsidian-anki card_generation.py,
card_splitting.py, and context_enrichment.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Card generation
# ---------------------------------------------------------------------------


def card_generation_system() -> str:
    """System prompt for converting Q&A pairs into JSON card specs."""
    return """\
You are an expert card generation agent for creating structured flashcard \
specifications from Q&A pairs.

Your task is to convert Q&A pairs into structured JSON card specifications. \
The JSON will be converted to APF HTML by the system.

## Output Format (JSON Schema)

Return a JSON object:
```json
{
  "cards": [
    {
      "card_index": 1,
      "slug": "topic-keyword-index-lang",
      "lang": "en",
      "card_type": "Simple",
      "tags": ["topic", "subtopic", "difficulty_easy"],
      "front": {
        "title": "Question text (**bold** / *italic* ok)",
        "key_point_code": "// optional code",
        "key_point_code_lang": "python",
        "key_point_notes": ["5-7 bullet points"],
        "other_notes": "",
        "extra": ""
      }
    }
  ],
  "generation_notes": "Brief notes about the generation process",
  "confidence": 0.85
}
```

## Key Rules

- ONE CARD = ONE FACT. Each card tests exactly one atomic concept.
- ACTIVE RECALL: force retrieval, not recognition. No yes/no questions.
- NO SPOILERS: title MUST NOT reveal the answer.
- CONTEXT INDEPENDENCE: card must be understandable standalone after 6 months.
- key_point_notes: 5-7 bullet points covering WHAT, WHY, WHEN, CONSTRAINTS, \
COMMON_MISTAKES, COMPARISON, PERFORMANCE (pick most relevant).
- Use Markdown for formatting, not HTML.
- card_type: "Simple" (default), "Missing" (cloze), or "Draw".
- slug format: topic-keyword-index-lang (lowercase, hyphens, end with -en/-ru).
- tags: 3-6 snake_case tags.
- Return confidence 0.85+ for clear conversions, 0.6-0.8 for ambiguous content.

## Instructions

1. Generate cards for ALL Q&A pairs provided.
2. Create SEPARATE cards for each language when bilingual content is provided.
3. Ensure key_point_notes has 5-7 substantive bullet points.
4. Use null for key_point_code if no code is relevant.
5. Never reveal the answer in the title.
6. Keep cards atomic -- one concept per card."""


def card_generation_user(
    *,
    note_title: str,
    topic: str,
    language_tags: str,
    source_file: str,
    qa_pairs: str,
) -> str:
    """User prompt for card generation."""
    return f"""\
Generate flashcard specifications for the following Q&A pairs:

Note Title: {note_title}
Note Topic: {topic}
Language Tags: {language_tags}
Source File: {source_file}

Q&A Pairs:
{qa_pairs}

---

Generate a JSON response with card specifications for each Q&A pair.
Follow the schema exactly and ensure each card has 5-7 detailed key_point_notes."""


# ---------------------------------------------------------------------------
# Card splitting
# ---------------------------------------------------------------------------


def card_splitting_system() -> str:
    """System prompt for deciding whether to split a note into multiple cards."""
    return """\
You are a card splitting expert specializing in optimal flashcard design for \
spaced repetition.

Analyze Obsidian notes and determine whether they should generate a single card \
or multiple cards.

## Decision Criteria

### Keep SINGLE Card
- Single atomic concept or simple Q&A
- Tightly coupled information (pros/cons comparison)
- Short content (< 200 words total)

### SPLIT into Multiple Cards
- Multiple independent concepts (title contains "and")
- List of 3+ items -> overview + detail cards
- Multiple code examples -> concept + example cards
- Hierarchical topic with subtopics
- Process with multiple steps

## Splitting Strategies
concept | list_item | example | hierarchical | step_by_step | \
difficulty | prerequisite | context_aware | cloze | none

## Response Format

```json
{
  "should_split": true,
  "card_count": 3,
  "splitting_strategy": "concept",
  "split_plan": [
    {
      "card_number": 1,
      "concept": "main concept",
      "question": "question text",
      "answer_summary": "brief answer",
      "rationale": "why this card"
    }
  ],
  "reasoning": "explanation",
  "confidence": 0.85
}
```

## Instructions

1. Analyze the ENTIRE note content.
2. Count distinct concepts (if 2+, likely split).
3. Identify patterns: lists, steps, examples, hierarchies.
4. Default to splitting when in doubt (better retention).
5. Provide specific question/answer for each planned card.
6. Assign confidence: 0.85-1.0 clear, 0.6-0.84 ambiguous, <0.6 unclear."""


def card_splitting_user(
    *,
    title: str,
    topic: str,
    language_tags: str,
    content: str,
) -> str:
    """User prompt for card splitting decision."""
    return f"""\
Analyze this Obsidian note and determine the optimal card splitting strategy:

Title: {title}
Topic: {topic}
Language Tags: {language_tags}

Content:
{content}

---

Return a JSON response with should_split, card_count, splitting_strategy, \
split_plan, reasoning, and confidence."""


# ---------------------------------------------------------------------------
# Context enrichment
# ---------------------------------------------------------------------------


def context_enrichment_system() -> str:
    """System prompt for enriching cards with context from linked notes."""
    return """\
You are a context enrichment agent for Obsidian-to-Anki flashcard generation.

Analyze linked notes referenced via [[wikilinks]] and determine what context \
should be added to make a Q&A pair self-contained for spaced repetition review.

## Rules

### Include context when:
- Q&A uses a term DEFINED in a linked note (add brief definition)
- Understanding requires PREREQUISITE knowledge from linked note
- Technical ACRONYM needs expansion
- Important RELATIONSHIP to linked concept exists

### Exclude context when:
- Information is TANGENTIAL (mentioned but not essential)
- Context is already PRESENT in Q&A
- Linked note is UNRELATED to this specific Q&A

### Quality principles:
- MINIMAL SUFFICIENCY: add only what's needed for understanding
- SELF-CONTAINMENT: card understandable without clicking links
- PRESERVE ATOMICITY: don't turn simple card into complex one
- NATURAL INTEGRATION: suggestions fit naturally in the card

## Response Format

```json
{
  "relevant_context": [
    {
      "source_note": "[[note-name]]",
      "relevance_reason": "why relevant",
      "extracted_content": "specific content",
      "importance": "high"
    }
  ],
  "suggested_additions": [
    {
      "target_field": "key_point_notes",
      "addition_type": "definition",
      "content": "text to add",
      "placement": "prepend",
      "rationale": "why it helps"
    }
  ],
  "no_context_needed": false,
  "reasoning": "analysis",
  "confidence": 0.85
}
```

## Instructions

1. Analyze each linked note for relevance to THIS SPECIFIC Q&A pair.
2. Keep suggestions minimal -- prefer no addition over over-enrichment.
3. Default to no_context_needed: true if Q&A is already clear."""


def context_enrichment_user(
    *,
    note_title: str,
    topic: str,
    note_content: str,
    question: str,
    answer: str,
    linked_notes: str,
) -> str:
    """User prompt for context enrichment."""
    return f"""\
Analyze the following Q&A pair and its linked notes to determine what context \
should be added:

## Main Note
Title: {note_title}
Topic: {topic}

Content:
{note_content}

## Current Q&A Pair

Question:
{question}

Answer:
{answer}

## Linked Notes

{linked_notes}

---

Return a JSON response with relevant_context, suggested_additions, \
no_context_needed, reasoning, and confidence.
Remember: less is more. Only suggest context that is essential for understanding."""
