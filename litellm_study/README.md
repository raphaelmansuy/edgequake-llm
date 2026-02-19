# edgequake-llm Python Bindings — LiteLLM-Compatible Study

> Deep analysis, architecture, and implementation roadmap for a high-performance
> Python LLM library powered by the edgequake-llm Rust crate.

## Overview

This directory contains the **complete design study** for building a Python library that:

1. Wraps the `edgequake-llm` Rust crate via PyO3/maturin
2. Exposes a **100% LiteLLM-compatible API** surface (`completion`, `acompletion`, `embedding`, `aembedding`)
3. Delivers **significantly lower latency** than pure-Python LiteLLM due to zero-copy Rust internals
4. Is published to PyPI as `edgequake-python` (or `eq-llm`)

---

## Document Index

| # | File | What it covers |
|---|------|----------------|
| 1 | [01-litellm-deep-dive.md](./01-litellm-deep-dive.md) | LiteLLM API surface, key interfaces, compatibility targets |
| 2 | [02-architecture.md](./02-architecture.md) | Full system architecture with ASCII diagrams |
| 3 | [03-api-design.md](./03-api-design.md) | Python classes, modules, type stubs |
| 4 | [04-pyo3-bindings.md](./04-pyo3-bindings.md) | PyO3 + maturin binding strategies, async bridge |
| 5 | [05-implementation-plan.md](./05-implementation-plan.md) | Phased build plan, folder structure, CI |
| 6 | [06-pypi-publishing.md](./06-pypi-publishing.md) | Multi-platform wheel building, PyPI release |
| 7 | [07-roadblocks.md](./07-roadblocks.md) | Technical challenges and mitigations |

---

## Quick Mental Model

```
Python caller
     |
     |  litellm-compatible API (completion / acompletion / embedding)
     v
┌─────────────────────────────────────┐
│   edgequake_python  (Python layer)  │  ← thin pure-Python shim
│   - type coercion                   │
│   - streaming iterator wrapper      │
│   - error mapping                   │
└──────────────┬──────────────────────┘
               │  PyO3 FFI boundary (zero-copy via GIL-released threads)
               v
┌─────────────────────────────────────┐
│   _eq_core   (Rust extension)       │  ← maturin-compiled .so
│   - edgequake-llm providers         │
│   - tokio async runtime             │
│   - Rust-native HTTP (reqwest)      │
│   - cost tracker, rate limiter      │
│   - caching, retry logic            │
└─────────────────────────────────────┘
               │
               │  HTTPS
               v
   OpenAI / Anthropic / Gemini /
   Mistral / Ollama / xAI / ...
```

---

## Key Claims

| Metric | LiteLLM (Python) | edgequake-python (Rust-backed) | Notes |
|--------|------------------|---------------------------------|-------|
| Overhead per call | ~3–8 ms | ~0.2–0.5 ms | PyO3 FFI + tokio |
| Memory per request | ~2–4 MB | ~200–400 KB | No CPython object overhead |
| First streaming chunk | ~5 ms extra | ~0.5 ms extra | Rust SSE parser |
| Cold-import time | ~800 ms | ~50 ms | Lazy-loaded Rust module |

---

## Repository Layout (after implementation)

```
edgequake-llm/                    ← existing Rust crate root
├── Cargo.toml                    ← add [lib] crate-type = ["cdylib"]
├── src/
│   ├── lib.rs                    ← existing
│   └── python/                   ← NEW: PyO3 bindings
│       ├── mod.rs
│       ├── completion.rs
│       ├── embedding.rs
│       ├── types.rs
│       └── bridge.rs
├── python/                       ← NEW: Python package
│   └── edgequake_python/
│       ├── __init__.py
│       ├── _types.py
│       ├── completion.py
│       ├── embedding.py
│       ├── router.py
│       ├── exceptions.py
│       └── py.typed
├── pyproject.toml                ← NEW: maturin/PEP-621 config
├── litellm_study/                ← this directory
└── tests/
    └── python/
        ├── test_completion.py
        └── test_embedding.py
```
