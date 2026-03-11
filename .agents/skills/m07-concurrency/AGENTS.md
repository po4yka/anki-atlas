# Concurrency - Quick Reference

Guides concurrency design: async vs threads, Send/Sync, synchronization primitives.

## Domain Detection

| Workload | Solution | Runtime |
|----------|----------|---------|
| I/O-bound (network, files) | `async` / `await` | tokio |
| CPU-bound (compute) | `std::thread` or `rayon` | OS threads |
| Mixed | Async + `spawn_blocking` | tokio |
| Embarrassingly parallel | `rayon::par_iter` | rayon |

## Send/Sync Quick Reference

| Type | Send | Sync | Notes |
|------|------|------|-------|
| `Arc<T: Send + Sync>` | Yes | Yes | Shared ownership across threads |
| `Mutex<T: Send>` | Yes | Yes | Interior mutability with locking |
| `RwLock<T: Send + Sync>` | Yes | Yes | Multiple readers, single writer |
| `Rc<T>` | No | No | Single-thread only |
| `Cell<T>` | Yes | No | Single-thread interior mutability |
| `RefCell<T>` | Yes | No | Single-thread borrow checking |

## Sub-File Index

| File | Content |
|------|---------|
| `comparison.md` | Async runtimes comparison (tokio vs async-std) |
| `examples/thread-patterns.md` | Thread spawning, joining, and communication |
| `patterns/async-patterns.md` | Async patterns: select, join, streams |
| `patterns/common-errors.md` | Common concurrency bugs and fixes |

## Essential Checklist

- [ ] Determine workload type (I/O vs CPU)
- [ ] All shared types are `Send + Sync`
- [ ] Use `Arc<Mutex<T>>` for shared mutable state
- [ ] Prefer channels over shared state when possible
- [ ] Use `spawn_blocking` for CPU work in async context
- [ ] Check for deadlock potential in lock ordering
