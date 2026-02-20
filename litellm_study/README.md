# edgequake-litellm — LiteLLM-Compatible Python Bindings Study

> Deep analysis, architecture, compatibility audit, and DX improvement roadmap for  
> **edgequake-litellm** — a high-performance Python LLM library powered by the  
> `edgequake-llm` Rust crate.  
> PyPI: `pip install edgequake-litellm` · Import: `import edgequake_litellm`

## Overview

This directory contains the **complete design study** for `edgequake-litellm`, a Python library that:

1. Wraps the `edgequake-llm` Rust crate via PyO3/maturin
2. Exposes a **LiteLLM-compatible API** surface (`completion`, `acompletion`, `embedding`, `aembedding`, `stream`)
3. Delivers **significantly lower latency** than pure-Python LiteLLM due to zero-copy Rust internals
4. Is published to PyPI as `edgequake-litellm`

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
| 8 | [08-compatibility-matrix.md](./08-compatibility-matrix.md) | **Full litellm vs edgequake-litellm compatibility audit** |
| 9 | [09-dx-improvements.md](./09-dx-improvements.md) | **DX improvement roadmap (P0–P3 priority items)** |

---

## Quick Mental Model

```
Python caller
     |
     |  litellm-compatible API (completion / acompletion / embedding / stream)
     v
┌─────────────────────────────────────┐
│  edgequake_litellm  (Python layer)  │  ← thin pure-Python shim
│   - type coercion                   │
│   - streaming iterator wrapper      │
│   - error mapping                   │
│   - litellm compat shims            │
└──────────────┬──────────────────────┘
               │  PyO3 FFI boundary (zero-copy via GIL-released threads)
               v
┌─────────────────────────────────────┐
│   _elc_core   (Rust extension)      │  ← maturin-compiled .so
│   - edgequake-llm providers         │
│   - tokio async runtime             │
│   - Rust-native HTTP (reqwest)      │
│   - cost tracker, rate limiter      │
│   - caching, retry logic            │
│   - OpenTelemetry (native OTEL)     │
└─────────────────────────────────────┘
               │
               │  HTTPS
               v
   OpenAI / Anthropic / Gemini /
   Mistral / Ollama / xAI / OpenRouter / ...
```

---

## Key Claims

| Metric | LiteLLM (Python) | edgequake-litellm (Rust-backed) | Notes |
|--------|------------------|---------------------------------|-------|
| Overhead per call | ~3–8 ms | ~0.2–0.5 ms | PyO3 FFI + tokio |
| Memory per request | ~2–4 MB | ~200–400 KB | No CPython object overhead |
| First streaming chunk | ~5 ms extra | ~0.5 ms extra | Rust SSE parser |
| Cold-import time | ~800 ms | ~50 ms | Lazy-loaded Rust module |

---

## Repository Layout (current state)

```
edgequake-llm/                    ← Rust crate root
├── Cargo.toml
├── src/                          ← Rust sources (edgequake-llm crate)
├── edgequake-litellm/            ← Python package root
│   ├── pyproject.toml            ← maturin/PEP-621 config
│   ├── python/
│   │   └── edgequake_litellm/
│   │       ├── __init__.py
│   │       ├── _types.py
│   │       ├── _compat.py        ← litellm compat shims
│   │       ├── completion.py
│   │       ├── embedding.py
│   │       ├── streaming.py
│   │       ├── config.py
│   │       ├── exceptions.py
│   │       └── py.typed
│   └── tests/                   ← Python unit + E2E tests
├── litellm_study/                ← this directory
└── examples/                    ← Rust examples
```
