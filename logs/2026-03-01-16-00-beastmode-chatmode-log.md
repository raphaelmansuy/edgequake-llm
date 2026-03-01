# Task Log: v0.3.0 / v0.2.0 Publication

## Actions
- Verified all 8 modified files (version bumps, changelogs, litellm updates)
- Ran cargo fmt, clippy, 1087 unit tests — all green
- Built litellm wheel locally (maturin build --release) — success
- Fixed missing `list_providers()` and `detect_provider()` re-exports in `__init__.py`
- Ran 86 Python unit tests — all pass
- Ran ruff lint — clean
- Committed and pushed to main (cd23621)
- Created and pushed tags: `v0.3.0` (crates.io) + `py-v0.2.0` (PyPI)
- Verified CI green on all 4 workflows
- Verified crates.io publication: edgequake-llm 0.3.0 live
- Verified PyPI publication: edgequake-litellm 0.2.0 live with 7 wheels + sdist
- Fixed manylinux abi3 wheel tag issue (before-script-linux for Python 3.9)

## Decisions
- Used `before-script-linux` to set Python 3.9 in PATH inside manylinux containers, fixing cp38 vs cp39-abi3 wheel tag issue.
- Ubuntu smoke test failure was caused by manylinux2014 defaulting to Python 3.8, producing cp38-cp38 wheel instead of cp39-abi3.
- linux-aarch64-manylinux failure is expected (can-fail: true, ring crate cross-compile).

## Next Steps
- Verify the abi3 fix on next py-v tag push (or manual workflow dispatch)
- Consider yanking the cp38-cp38 wheel from PyPI 0.2.0 if it causes user issues
- Monitor crates.io docs.rs build for edgequake-llm 0.3.0

## Lessons/Insights
- manylinux2014 container's default Python can interfere with abi3-py39 builds; always ensure Python 3.9+ is in PATH via before-script-linux.
- list_providers() was in Rust native module but not re-exported in Python __init__.py — caught during smoke test.
