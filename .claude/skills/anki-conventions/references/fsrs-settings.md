# FSRS Settings Reference

Quick reference for configuring the Free Spaced Repetition Scheduler (FSRS) in Anki.

## Setup Checklist

- [ ] Enable FSRS in deck options (Deck Options > FSRS)
- [ ] Set desired retention to **0.90** (adjust 0.85-0.95 based on stakes)
- [ ] Configure learning step: **15m or 30m** (single step recommended)
- [ ] Set relearning step: **10-15m**
- [ ] Click "Optimize" monthly to tune parameters
- [ ] Verify you have Anki 24.11+ (FSRS-5) or 25.02+ (FSRS-6)

## Retention vs Workload Tradeoffs

| Retention Target | Relative Workload | Use Case |
|------------------|-------------------|----------|
| 0.85 | ~0.7x | Casual learning, large vocabulary decks |
| **0.90** | **1x (baseline)** | **Default - good balance for most purposes** |
| 0.92 | ~1.3x | Important material |
| 0.95 | ~2x | High-stakes exams, medical boards |
| 0.97 | ~3.7x | Critical precision required |
| 0.99 | ~10x | Not practical for most users |

**Rule of thumb:** Each 5% increase in retention roughly doubles your workload.

## Settings That Disappear with FSRS

When FSRS is enabled, these SM-2 settings become irrelevant:

| Obsolete Setting | Why |
|------------------|-----|
| Graduating interval | FSRS calculates optimal intervals |
| Easy interval | FSRS doesn't use ease factors |
| Easy bonus | No ease system |
| Interval modifier | Replaced by desired retention |
| Starting ease | No ease in FSRS |
| Hard interval | FSRS handles "Hard" differently |
| New interval (%) | FSRS calculates from scratch |

## Settings That Still Matter

| Setting | Recommended | Notes |
|---------|-------------|-------|
| Learning steps | 15m or 30m | Keep under 1 day, single step ideal |
| Relearning steps | 10-15m | Short, single step |
| Daily new card limit | 5-20 | Multiply by 7-10 for eventual daily reviews |
| Maximum interval | 180-365 | Personal preference |

## Button Behavior Under FSRS

| Button | When to Use | FSRS Interpretation |
|--------|-------------|---------------------|
| **Again** | Forgot the answer | Failed recall, relearn |
| **Hard** | Recalled with difficulty | **Do NOT use when forgotten** |
| **Good** | Normal recall | Standard success |
| **Easy** | Instant recall | Strong memory |

**Critical:** FSRS assumes "Hard" means "recalled with difficulty." Using Hard when you forgot breaks the algorithm. Use only **Again** and **Good** for most reviews.

## Optimization

### Monthly Optimization
1. Go to Deck Options > FSRS
2. Click "Optimize"
3. Wait for parameter calculation
4. Review estimated retention accuracy

### FSRS Helper Add-on (759844606)
Provides additional features:
- Reschedule based on full review history
- Load balancing across days
- Steps Stats for recommended learning steps
- Schedule breaks

## Version Differences

| Version | Anki Release | Key Features |
|---------|--------------|--------------|
| FSRS-4 | < 24.11 | Original implementation |
| FSRS-5 | 24.11+ | Short-term memory modeling, recency weighting |
| FSRS-6 | 25.02+ | 21 parameters, considers all daily reviews |

## Troubleshooting

### Cards Seem Too Easy/Hard
- Click "Optimize" to retrain on your data
- Adjust desired retention (higher = more frequent)
- Wait 2-4 weeks for algorithm to calibrate

### Migration from SM-2
- Enable FSRS - existing cards will adapt automatically
- Don't reset your deck (you lose progress)
- Intervals will adjust over the next few review cycles

### FSRS Helper Not Working
- Requires Anki 24.11+ for full compatibility
- Some features temporarily removed in Anki 25.x
- Check AnkiWeb add-on page for current status
