# Card Design Patterns

Best practices for creating effective Anki flashcards that maximize retention.

## User-Specific Preferences

**Mastery-oriented philosophy:** Cards are designed for deep understanding, not elementary learning.

| Principle | Application |
|-----------|-------------|
| **Precise terminology** | Use correct technical terms, not simplified language |
| **Depth over brevity** | Include nuanced explanations and conceptual connections |
| **Expert-level questions** | Test understanding and reasoning, not just recognition |
| **Why and how** | Focus on reasoning alongside facts |

**For programming/technical cards:** The atomic rule is relaxed. Longer cards with code snippets, multi-line examples, and contextual explanations are acceptable.

See `programming-cards.md` for detailed technical card patterns.

## Core Principles

### 1. Atomic Cards (One Concept)

**One concept per card.** Break topics into meaningful units, but include depth.

**Elementary (avoid):**
```
Q: What is the boiling point of water?
A: 100C
```

**Mastery (prefer):**
```
Q: Why does water's boiling point (100C at sea level) decrease at higher altitudes?
A: Lower atmospheric pressure means fewer air molecules pressing down on the water surface.
   Water molecules need less kinetic energy to overcome this reduced pressure and escape
   as vapor. At 3000m elevation, water boils at ~90C.
```

### 2. Precise Questions

Questions should test understanding with unambiguous correct answers.

**Elementary (avoid):**
```
Q: What keyword starts a function definition in Python?
A: def
```

**Mastery (prefer):**
```
Q: How does Python's `def` keyword differ from `lambda` for defining functions?
A: `def` creates a named function with a block body (multiple statements, docstrings, decorators).
   `lambda` creates anonymous, single-expression functions.

   Use `def` for: reusable logic, complex operations, anything needing documentation.
   Use `lambda` for: short callbacks, key functions in sort/filter, functional patterns.
```

### 3. Contextual and Connected

Include context and connect to related concepts.

### 4. Cloze Deletions

Use for definitions, lists, and fill-in-the-blank learning.

**Definition:**
```
{{c1::Photosynthesis}} is the process by which plants convert
{{c2::sunlight}} into {{c3::chemical energy}}.
```

**List (one card per item):**
```
The three branches of US government are:
- {{c1::Legislative}}
- {{c2::Executive}}
- {{c3::Judicial}}
```

### 5. Bidirectional Learning

Use "Basic (and reversed)" for concepts that benefit from both directions.

**Good candidates:**
- Vocabulary (word <-> definition)
- Translations (English <-> Spanish)
- Symbols (symbol <-> meaning)

## Card Templates (Mastery-Oriented)

### Understanding "Why" (Not Just "What")

```
Front: Why did [event] occur when it did, and what were the key preconditions?
Back: [explanation of causes, context, and contributing factors]
```

### Trade-off Comparison

```
Front: When would you choose [A] over [B], and what trade-offs does this involve?
Back:
Choose A when: [conditions]
Choose B when: [conditions]

Trade-offs:
- A: [advantages] / [disadvantages]
- B: [advantages] / [disadvantages]
```

### Code Pattern with Reasoning

```
Front: Python: When should you use [pattern], and what problem does it solve?
Back:
[code example]
Use when: [conditions]
Solves: [problem]
Avoid when: [counter-indications]
```

## Anti-Patterns to Avoid

### 1. Elementary Surface Questions

### 2. Recognition Instead of Understanding

### 3. Isolated Facts

### 4. Lists Without Reasoning

Split enumeration into cards that explain each item's purpose.

### 5. Surface Definitions

Connect definitions to when/why you'd use the concept, not just what it means.

## Tagging Strategy

### Format Rules

| Rule | Convention | Example |
|------|-----------|---------|
| Hierarchy separator | `::` (double-colon) | `android::compose` |
| Word separator | `-` (kebab-case) | `state-management` |
| Code identifiers | Original casing | `ArrayList`, `WorkManager` |
| Max depth | 2 levels | `kotlin::coroutines` |

### Domain Prefixes

| Prefix | Examples |
|--------|---------|
| `android::` | `android::compose`, `android::lifecycle`, `android::room` |
| `kotlin::` | `kotlin::coroutines`, `kotlin::flow`, `kotlin::null-safety` |
| `cs::` | `cs::algorithms`, `cs::data-structures`, `cs::concurrency` |
| `topic::` | `topic::system-design`, `topic::security`, `topic::patterns` |
| `difficulty::` | `difficulty::easy`, `difficulty::medium`, `difficulty::hard` |
| `source::` | `source::book-name`, `source::course-name` |
| `context::` | `context::interview-prep`, `context::certification` |

See `tag-conventions.md` for full rules, normalization, and validation.

## Mastery Quality Checklist

Before adding a card, verify:

- [ ] Uses precise technical terminology
- [ ] Question requires understanding, not just recall
- [ ] Answer explains "why" or "how", not just "what"
- [ ] Connects to related concepts or principles
- [ ] Would help someone apply knowledge, not just recognize it
- [ ] Tests at the level of a practitioner, not a beginner
- [ ] Worth remembering long-term?
- [ ] Are appropriate tags applied?

## Related References

- **Technical cards:** See `programming-cards.md` for code-specific patterns
- **Maintenance:** See `card-maintenance.md` for reformulating problematic cards
- **Organization:** See `deck-organization.md` for tagging strategies
