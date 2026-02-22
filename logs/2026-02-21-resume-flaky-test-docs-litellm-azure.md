# 2026-02-21 Resume: Fix flaky test, docs, edgequake-litellm Azure

## Actions
- Fixed `test_from_env_fallback_to_mock` in `src/factory.rs`: added `remove_var` for `AZURE_OPENAI_CONTENTGEN_API_KEY`, `AZURE_OPENAI_CONTENTGEN_API_ENDPOINT`, `AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`
- Updated `docs/providers.md` Azure section: 3-constructor table, CONTENTGEN env vars, `from_env_auto()`, programmatic builder, content-filter note, reliable images URLs
- Updated Azure feature row in feature comparison table: Vision now `Y`
- Updated `edgequake-litellm/src/completion.rs`: added `"azure"` to `list_providers()`, updated `parse_provider()` error message
- Created `edgequake-litellm/examples/09_azure_openai.py`: 5 sections (detection, simple chat, JSON mode, streaming, list_providers)
- Updated `edgequake-litellm/examples/README.md`: added example 09 to table and quick-start section

## Decisions
- Root cause of flaky test: `test_from_env_auto_detects_azure_with_contentgen_vars` leaves CONTENTGEN vars populated; fallback test never cleared them (Azure CONTENTGEN checked first in factory::from_env)
- Azure vision = `Y` in feature table since `ImageData::from_url()` + `to_api_url()` routes URLs directly

## Next steps
- No remaining open items from the conversation todo list
- Optionally re-publish edgequake-litellm wheel with updated list_providers

## Lessons/insights
- When guarding serial tests from env pollution, enumerate ALL factory env vars (including CONTENTGEN variants), not just the canonical name
