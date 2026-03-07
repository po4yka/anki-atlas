# Card Maintenance Reference

Guidelines for maintaining a healthy Anki collection over time.

## Daily Habits

| Habit | Impact |
|-------|--------|
| Review at the same time daily | Builds routine, reduces decision fatigue |
| Do reviews before adding new cards | Prevents backlog accumulation |
| Morning reviews recommended | Positive start, reliable slot |
| Keep sessions under 30 minutes | Maintains focus and motivation |

## Sustainable New Card Rates

| Context | Daily New Cards | Expected Daily Reviews |
|---------|-----------------|------------------------|
| Casual learning | 5-10 | 35-100 |
| Active study | 10-20 | 70-200 |
| Intensive (short-term) | 30-50 | 210-500 |
| Burnout territory | 50+ | 350+ |

**Rule of thumb:** Multiply daily new cards by 7-10 for eventual daily review load.

## Backlog Handling

### Missed One Day

Just catch up tomorrow. This barely qualifies as a backlog.

### Missed Several Days

**"Stop the bleeding" approach:**

1. Find overdue cards:
   ```
   prop:due<0
   ```

2. Suspend all overdue cards:
   - Select all in browser
   - Cards > Suspend

3. Tag suspended cards:
   ```
   tag:backlog-YYYYMMDD
   ```

4. Daily recovery:
   - Keep current with newly-due cards
   - Unsuspend 20-50 backlog cards daily
   - Prioritize by importance

### Missed Weeks or Months

1. **Audit ruthlessly:**
   - Delete decks you no longer need
   - Suspend cards for topics you've abandoned
   - Be honest about what's worth recovering

2. **Recognize the math:**
   - Most cards are already overdue
   - The pile isn't growing much
   - Recovery is faster than you expect

3. **Recovery strategy:**
   - Start with most important deck
   - Set low daily limits initially
   - Gradually increase as you catch up

## What Never to Do

| Anti-Pattern | Why It's Bad |
|--------------|--------------|
| Reset your deck | Loses all progress on cards you still remember |
| Use automatic rescheduling tools | Hides true cost of missed study |
| Declare "Anki bankruptcy" prematurely | Often unnecessary |
| Delete all overdue cards | Wastes prior learning effort |

## Card Reformulation Signals

When a card consistently fails, don't just keep failing it. Reformulate for mastery.

### Signs a Card Needs Rewriting

| Signal | Likely Problem |
|--------|----------------|
| Leech status (8+ failures) | Card is fundamentally flawed |
| Always mix up with another card | Insufficient differentiation |
| Answer feels arbitrary | Missing connection to understanding |
| Can't remember what question means | Poor context or phrasing |
| Answer is too long to recall | Not atomic enough |
| **Know answer but can't apply it** | **Tests recognition, not understanding** |
| **Feels too elementary** | **Needs depth - add "why" and trade-offs** |

### Reformulation Strategies

**Deepen for Mastery:**
- Transform "what" questions to "why" or "when"
- Add trade-offs and decision guidance
- Connect to related concepts and principles
- Include reasoning in the answer

| Elementary | Mastery |
|------------|---------|
| "What is X?" | "When would you choose X over Y?" |
| "Define X" | "How does X differ from Y, and when is each used?" |
| Single fact | Trade-offs + reasoning + connections |

**Split the card:**
- One concept per card (with appropriate depth)
- Break lists into individual items with reasoning for each
- Separate related concepts into comparison cards

**Add context and connections:**
- Include enough clues to disambiguate
- Reference related concepts
- Explain why this matters

**Rephrase for understanding:**
- Ask about reasoning, not just facts
- Test application, not recognition
- Use precise technical terminology

**Delete:**
- If you don't care about the content
- If it's outdated or incorrect
- If it's better learned through practice
- **If it's too elementary and not worth deepening**

## Leech Management

Leeches are cards that repeatedly fail (default: 8 times).

### When a Card Becomes a Leech

1. Anki suspends it automatically
2. Don't just unsuspend and hope
3. **Analyze why it's failing:**
   - Is the content meaningful to you?
   - Is the question clear?
   - Is the answer memorable?

4. **Take action:**
   - Reformulate using strategies above
   - Add prerequisite cards
   - Delete if not worth the effort

## Scheduled Maintenance

### Weekly (5 minutes)

- [ ] Check for leeches: `is:suspended tag:leech`
- [ ] Review flagged cards for attention
- [ ] Delete any cards you've discovered are wrong

### Monthly (15 minutes)

- [ ] Click "Optimize" in FSRS settings
- [ ] Review cards added 30+ days ago for quality
- [ ] Check for outdated technical content
- [ ] Review tag organization

### Quarterly (30 minutes)

- [ ] Audit entire deck structure
- [ ] Delete abandoned decks
- [ ] Update cards for changed information
- [ ] Review statistics for trends

## Anti-Patterns

### The "Scheduled Garbage" Problem

Creating cards from material you don't understand:
- Cards become guessing games
- Knowledge doesn't transfer
- Wastes time on both creation and review

**Fix:** Learn first, card second. Always.

### Passive Recognition

Flipping cards without committing to an answer:
- Not actually testing memory
- Inflates success rate artificially
- Kills learning effectiveness

**Fix:** Say or write answer before flipping.

### Over-Ankification

Spending more time on Anki than on actual learning/practice:
- Anki supports learning, doesn't replace it
- Procedural skills need practice, not cards
- Diminishing returns on card quantity

**Fix:** 80% immersion/practice, 20% Anki.
