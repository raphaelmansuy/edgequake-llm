# 2026-02-21 Fix flaky test_from_env_fallback_to_mock

## Actions
- Added `remove_var` for `AZURE_OPENAI_CONTENTGEN_API_KEY`, `AZURE_OPENAI_CONTENTGEN_API_ENDPOINT`, `AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` to `test_from_env_fallback_to_mock` cleanup block in `src/factory.rs`

## Decisions
- Root cause: `factory::from_env()` checks `AZURE_OPENAI_CONTENTGEN_API_KEY` before `AZURE_OPENAI_API_KEY`; prior test `test_from_env_auto_detects_azure_with_contentgen_vars` set the CONTENTGEN vars, and the fallback test never cleared them
- No structural changes needed — simple missing `remove_var` lines at test top

## Next steps
- Update `docs/providers.md` Azure section (pending)
- Update `edgequake-litellm` Python bridge (pending)

## Lessons/insights
- When guarding env vars in serial tests, enumerate ALL variants the factory checks, not just the most obvious one; CONTENTGEN is checked FIRST in `from_env()`
