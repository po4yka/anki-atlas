# Anki Note Types Reference

Guide to built-in and common note types (models) in Anki.

## Built-in Note Types

### Basic

The simplest note type with Front and Back fields.

**Fields:**
- Front
- Back

**Cards Generated:** 1 (Front -> Back)

**Use Cases:**
- Simple Q&A
- Definitions
- Facts with one-way recall

### Basic (and reversed card)

Same as Basic but generates two cards for bidirectional learning.

**Fields:**
- Front
- Back

**Cards Generated:** 2 (Front -> Back, Back -> Front)

**Use Cases:**
- Vocabulary learning
- Translations
- Symbol <-> meaning pairs
- Bidirectional associations

### Basic (optional reversed card)

Like Basic, with optional reverse card controlled by a field.

**Fields:**
- Front
- Back
- Add Reverse

**Cards Generated:** 1 or 2 (reverse only if Add Reverse has content)

### Basic (type in the answer)

Front card includes a text input for typing the answer.

**Fields:**
- Front
- Back

**Cards Generated:** 1

**Use Cases:**
- Spelling practice
- Exact recall required
- Language learning (typing words)

**Note:** Uses `{{type:Back}}` in template.

### Cloze

Special note type for fill-in-the-blank cards.

**Fields:**
- Text
- Extra (optional additional info)

**Cards Generated:** One per cloze deletion

**Syntax:**
```
{{c1::hidden text}}
{{c2::another hidden}}
{{c1::same card as first}}
```

**Use Cases:**
- Definitions
- Lists and enumerations
- Fill-in-the-blank
- Contextual learning

**Hint syntax:**
```
{{c1::answer::hint text}}
```

## Custom Note Types

### Q&A with Source

**Fields:** Question, Answer, Source, Notes

### Vocabulary

**Fields:** Word, Definition, Example Sentence, Part of Speech, Pronunciation

### Code Snippet

**Fields:** Language, Description, Code, Output, Explanation

### Image Occlusion

**Fields:** Image, Header, Remarks
**Note:** Requires Image Occlusion add-on.

## Creating Custom Note Types

### Via MCP

```
mcp__anki__modelNames      # List existing types
mcp__anki__modelFieldNames # Get fields for a type
```

## Choosing the Right Note Type

| Scenario | Recommended Type |
|----------|------------------|
| Simple fact | Basic |
| Word <-> translation | Basic (reversed) |
| Definition with context | Cloze |
| List memorization | Cloze (one item per c#) |
| Spelling practice | Basic (type answer) |
| Complex topic | Multiple Basic cards |
| Code syntax | Custom code type |
| Visual learning | Image Occlusion |

## Common Issues

### Too Many Card Types

Stick to a few note types. More types = more maintenance.

### Overly Complex Templates

Keep templates simple. Complex CSS/JS can break on mobile.

### Inconsistent Field Usage

Decide field purposes upfront and stick to them.
