# 2026-03-01 Bedrock E2E Fixes

## Task Logs

- **Actions**: Diagnosed 4 bugs in Bedrock provider; changed default model to `amazon.nova-lite-v1:0`; added inference profile auto-resolution based on region geography; extracted detailed AWS SDK errors via `ProvideErrorMetadata`; fixed factory `block_on` panic with `block_in_place`; fixed blank text ContentBlock in multi-turn tool calling; added 8 unit tests for `resolve_model_id_for_region`; updated CHANGELOG, README, docs/providers.md, e2e tests; committed and pushed to main.
- **Decisions**: Used `amazon.nova-lite-v1:0` as default (no geo-restrictions, works in all regions); auto-prefix bare model IDs with geography code derived from region (us-, eu-, ap-, etc.); store `region` field in `BedrockProvider` struct.
- **Next steps**: Bump version to 0.2.10 and release; verify CI passes on push; consider adding Bedrock embedding support.
- **Lessons/insights**: AWS Bedrock now requires inference profile IDs (e.g., `us.model-id`) for the Converse API — bare model IDs no longer work even for ON_DEMAND models. Anthropic models have geographic restrictions on Bedrock. `SdkError::Display` only prints "service error" — must use `ProvideErrorMetadata` trait for real error details.
