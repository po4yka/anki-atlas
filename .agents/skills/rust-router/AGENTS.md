# Rust Router - Quick Reference

Routes all Rust questions to the correct skill or sub-file based on intent analysis.

## Routing Decision Tree

```
Rust question?
    |
    +-- Compile error (E0xxx)?
    |   +-- Ownership/borrow/lifetime --> m01-ownership
    |   +-- Send/Sync/async --> m07-concurrency
    |   +-- Type mismatch --> SKILL.md type section
    |
    +-- Design question?
    |   +-- Error handling --> m06-error-handling
    |   +-- Performance --> m10-performance
    |   +-- Code style --> coding-guidelines
    |
    +-- Learning/conceptual?
    |   +-- Mental model --> m14-mental-model
    |   +-- Anti-pattern --> m15-anti-pattern
    |
    +-- Unsafe code? --> unsafe-checker
```

## Dual-Skill Loading

| Scenario | Primary | Secondary |
|----------|---------|-----------|
| Ownership error in async | m01-ownership | m07-concurrency |
| Error handling perf | m06-error-handling | m10-performance |
| Anti-pattern in unsafe | m15-anti-pattern | unsafe-checker |

## Sub-File Index

| File | Content |
|------|---------|
| `examples/workflow.md` | End-to-end routing workflow examples |
| `integrations/os-checker.md` | OS-specific checker integration |
| `patterns/negotiation.md` | Multi-skill negotiation patterns |

## Essential Checklist

- [ ] Identify question category (error, design, learning)
- [ ] Check if dual-skill loading needed
- [ ] Route to most specific skill first
- [ ] Fall back to SKILL.md for general Rust questions
