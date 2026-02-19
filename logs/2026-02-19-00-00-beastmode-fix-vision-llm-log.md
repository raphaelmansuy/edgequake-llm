# Task Log: 2026-02-19 – Fix Vision Support (#3)

## Actions
- Fixed `OpenAIProvider::convert_messages()` in `src/providers/openai.rs` to build multipart `ChatCompletionRequestUserMessageContent::Array` when `msg.has_images()` is true; imported `ImageUrl`, `ImageDetail`, `ChatCompletionRequestUserMessageContentPart`, `ChatCompletionRequestMessageContentPartImage/Text` from async-openai 0.24.
- Fixed `AzureOpenAIProvider::convert_messages()` in `src/providers/azure_openai.rs`: changed `AzureMessage.content` from `String` to `serde_json::Value` and built multipart JSON arrays for vision messages.
- Added `build_user_content()` and `parse_image_detail()` helpers to `OpenAIProvider`.
- Added 15+ unit tests in both provider modules (text-only stays string, image present produces array, data URI encoding, detail level parsing).
- Added e2e test `test_openai_provider_vision_chat` that sends a 10×10 red PNG to `gpt-4o` and asserts the model responds with image context.
- Bumped version `0.2.1 → 0.2.2` via `make version-patch`.
- Updated `CHANGELOG.md` with `[0.2.2]` entry.
- Committed, pushed `fix/vision-llm` branch, published `edgequake-llm v0.2.2` to crates.io.

## Decisions
- Used `serde_json::Value` for `AzureMessage.content` to avoid adding a new enum type while keeping JSON serialization correct for both plain text and multipart arrays.
- Used async-openai 0.24's native `ChatCompletionRequestUserMessageContent::Array` (not raw JSON) for `OpenAIProvider` to maintain type safety.
- Vision e2e uses a programmatically generated 10×10 red pixel PNG (not a URL) to avoid network dependencies and ensure determinism.

## Next Steps
- Open PR on GitHub from `fix/vision-llm` → `main`.
- Consider adding e2e vision test for Azure OpenAI (requires Azure credentials).

## Lessons/Insights
- async-openai 0.24 already provides all necessary multipart types; no dep upgrade needed.
- Changing `content: String` to `serde_json::Value` in Azure message struct is a clean, backward-compatible way to support both string and array content without breaking existing tests.
