# Mental Models - Quick Reference

Provides intuitive mental models for understanding Rust concepts.

## Mental Model Summary

| Concept | Model | Analogy |
|---------|-------|---------|
| Ownership | Unique key | One person has the house key |
| Move | Key handover | Giving away your key |
| `&T` | Lending for reading | Lending a book (can't write in it) |
| `&mut T` | Exclusive editing | Only you can edit the document |
| Lifetime `'a` | Valid scope | "Ticket valid until..." |
| `Clone` | Photocopy | Making a duplicate |
| `Copy` | Auto-photocopy | Numbers are inherently copyable |
| `Drop` | Destructor | Cleanup when leaving scope |
| Trait | Interface + behavior | A contract to fulfill |
| `Box<T>` | Heap pointer | A labeled storage locker |
| `Arc<T>` | Shared access pass | Multiple people with the same key |

## Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "References are pointers" | References are *guaranteed-valid* pointers with rules |
| "Clone is always expensive" | Depends on type; `Rc::clone` is just a counter bump |
| "Lifetimes change runtime behavior" | Lifetimes are compile-time only, zero runtime cost |
| "Ownership is like GC" | Ownership is deterministic; GC is not |

## Sub-File Index

| File | Content |
|------|---------|
| `patterns/thinking-in-rust.md` | Deep-dive into thinking patterns for Rust |

## Essential Checklist

- [ ] Start with the correct mental model before coding
- [ ] Map prior language experience to Rust equivalents
- [ ] Identify and correct misconceptions early
- [ ] Use analogies to build intuition, then refine with precision
