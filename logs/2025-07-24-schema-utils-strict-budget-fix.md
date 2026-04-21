# Task Log: Schema Utils & Strict-Budget Fix

## Actions
- Created `src/providers/schema_utils.rs` — shared DRY module with 6 composable utilities: `count_optional_params`, `count_union_type_params`, `check_anthropic_strict_budget`, `strip_keys_recursive`, `ensure_additional_properties_false`, `convert_type_arrays_to_nullable`, `normalize_for_openai_strict`
- Fixed `anthropic.rs` `convert_tools()` — now checks strict-mode budget (20 tools, 24 optional params, 16 union-type params) and strips `strict` from ALL tools when exceeded
- Fixed `bedrock.rs` `build_tool_config_with_aliases()` — now applies schema sanitization (strip unsupported constraints + additionalProperties:false) since Bedrock uses Claude
- Added 3 targeted Anthropic tests for strict-budget enforcement
- Registered `schema_utils` module in `providers/mod.rs`

## Decisions
- OpenAI provider left as-is: `async-openai` crate doesn't expose `.strict()` on FunctionObjectArgs, so strict mode is never sent (safe default)
- Gemini provider left as-is: already has comprehensive sanitization, refactoring to shared utils is optional
- When Anthropic budget exceeded, strip strict from ALL tools (not selective) — matches Anthropic's own guidance and Claude Code's pattern

## Next Steps
- Test with EdgeCrab's 94 tools against `anthropic/claude-sonnet-4.6` to verify the "142 optional parameters" error is resolved
- Consider future work: per-tool `strict` opt-in (like Claude Code) vs. the current default-true approach in `traits.rs`

## Lessons
- Anthropic strict mode has aggregate limits across ALL tools in a request (24 optional params total), not per-tool limits — this is why 94 tools with `strict: true` default fails
- Claude Code gates strict behind 3 conditions: feature flag AND per-tool flag AND model support — a good defensive pattern
