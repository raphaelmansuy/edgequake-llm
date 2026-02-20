# Task Log: 2026-02-20 — edgequake-python PyO3 bindings — Build + Test fix

## Actions
- Fixed 7 PyO3 0.28 compile errors: `py.allow_threads` → `runtime().block_on()` (sync), `Python::with_gil` → `Python::try_attach`, `tool_choice.as_ref()` → `tool_choice` (Option<ToolChoice> not &), unused `PyStreamChunk` import removed
- Added `LlmError::NotSupported` fallback in `stream_completion` so providers without streaming (mock, lmstudio, etc.) return a single-chunk synthetic stream via `chat()`
- Fixed 5 `#[pyclass]` deprecation warnings via `skip_from_py_object` attribute
- Fixed unused `py` variable in `lib.rs` (`_py`)
- Added `edgequake-python/.gitignore` to exclude `*.so`, `__pycache__/`, `*.pyc`
- Built with `maturin develop --release` (abi3 cp39+ wheel, 30s compile)
- Ran 76 unit tests → 76 passed, 0 failures
- Committed 33 files to `feat/litellm`, pushed to GitHub

## Decisions
- Used `runtime().block_on()` (GIL-held sync) instead of `allow_threads` — correct for single-threaded callers; can be revisited with `py.detach()` once PyO3 0.29 stabilises
- Streaming fallback emits `Content` + `Finished` chunks from `chat()` — transparent to Python callers
- `ToolChoice` cloned before stream attempt so it remains available in fallback branch

## Next steps
- Open GitHub PR for `feat/litellm` → `main`
- Run E2E tests once OPENAI_API_KEY / ANTHROPIC_API_KEY are available
- Verify CI pipelines (python-ci.yml, python-publish.yml) run green on push

## Lessons/Insights
- PyO3 0.28 renamed `Python::with_gil` → `Python::try_attach` and removed `allow_threads`; compiler error messages hint at both replacements clearly
- `ToolChoice: Option<ToolChoice>` is moved into async closures — must clone before branching
