# Performance Optimization - Quick Reference

Guides performance analysis and optimization in Rust.

## Optimization Priority

| Priority | Area | Technique | Impact |
|----------|------|-----------|--------|
| 1 | Algorithm | Better complexity | Highest |
| 2 | Allocation | Pre-allocate, reuse buffers | High |
| 3 | Data layout | Contiguous, cache-friendly | Medium-High |
| 4 | Parallelism | rayon, async I/O | Medium |
| 5 | Micro | SIMD, inlining, branchless | Low-Medium |

## Measure First

| Tool | Use For |
|------|---------|
| `criterion` | Micro-benchmarks |
| `flamegraph` | CPU profiling |
| `dhat` / `heaptrack` | Allocation profiling |
| `perf stat` | Hardware counters |
| `cargo bench` | Quick benchmarks |

## Sub-File Index

| File | Content |
|------|---------|
| `patterns/optimization-guide.md` | Detailed optimization patterns and examples |

## Essential Checklist

- [ ] Measure before optimizing (no guessing)
- [ ] Start with algorithm and data structure choice
- [ ] Use `with_capacity()` for Vec/String/HashMap
- [ ] Prefer `&[T]` over `Vec<T>` in function params
- [ ] Profile allocations before micro-optimizing
- [ ] Benchmark before and after with criterion
