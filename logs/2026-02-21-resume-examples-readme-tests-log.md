# Task Log — Resume: examples README + tests

## Actions
- Rewrote `examples/README.md` from scratch (130 lines, clean provider-organized layout) via Python script
- Fixed `examples/openai/demo.rs`: removed non-existent `FunctionParameters` import, added `strict` field to `FunctionDefinition`, changed `ToolChoice::Auto(None)` → `ToolChoice::auto()`
- Added empty-string validation to `AzureOpenAIProvider::from_env()` and `from_env_contentgen()` (prevents dotenv from silently succeeding with `.env` credentials)
- Updated `test_supports_json_mode_default_is_false` to use `davinci-002` model (old model without JSON mode support)
- Updated both Azure factory tests to use `set_var("", "")` + restore pattern instead of `remove_var` (dotenvy doesn't override existing vars)

## Decisions
- `gpt-5-mini` supports JSON mode (correct modern behavior); test switched to `davinci-002` for negative testing
- Empty-string validation added at provider level rather than test workaround
- `test_from_config_azure_no_creds` upgraded to `#[serial]` for env-var safety

## Results
- `cargo check --examples` → Finished (0 errors)
- `cargo test --lib` → **1023 passed, 0 failed** (was: 1020 passed, 3 failed)

## Next steps
- Update `docs/providers.md` with new Azure constructor table and env var docs
- Update `edgequake-litellm` Python bridge with Azure support

## Lessons
- `dotenvy::dotenv()` only sets vars NOT already in environment; use `set_var("", "")` to block re-loading from `.env` during isolation tests
- `#[serial]` tests must save+restore env vars when `.env` file exists locally
