# Task Log: 2026-03-02 Bedrock Full Model Coverage

## Actions
- Fixed 2 clippy warnings (if_same_then_else, cloned_ref_to_slice_refs)
- Fixed Mistral Pixtral inference profile auto-resolution (added mistral.pixtral to profile families)
- Fixed MiniMax M2 empty response (reasoning model needs higher max_tokens)
- Fixed Writer Palmyra model ID (palmyra-x-004 → palmyra-x4-v1:0)
- Added deepseek.* and writer.* to inference profile families
- Added context_length_for_model entries for Nemotron, MiniMax, Gemma, GLM, GPT-OSS
- Added 10 new e2e tests: MiniMax M2.1, Magistral Small, Devstral 2, Ministral 8B, Gemma 3 4B, Nemotron 30B, Qwen3 Coder, OpenAI GPT OSS, DeepSeek V3.2, Magistral tool calling
- Updated docs/providers.md with comprehensive Bedrock section (12 providers, embedding support)
- Updated CHANGELOG.md with all additions and fixes
- Committed and pushed to main (ef9e2fd)

## Decisions
- MiniMax M2/M2.1 use reasoningContent blocks; need 500+ max_tokens for text output
- Magistral Small doesn't use Converse API tool calling mechanism; uses text-based [TOOL_CALLS] format
- Region-limited tests (9 total) correctly annotated — they fail in eu-west-1 by design

## Next Steps
- Consider adding Claude 4.6, Opus 4.5 e2e tests when subscription allows
- Consider handling reasoningContent blocks in extract_content for thinking support
- Version bump to 0.3.0 for embedding feature release

## Lessons/Insights
- AWS Bedrock inference profiles are model-family-specific, not provider-wide (e.g., mistral.pixtral needs profile but mistral.mistral-large does not)
- MiniMax M2 is a reasoning model that uses SDK_UNKNOWN_MEMBER reasoningContent blocks — low max_tokens exhausts budget on CoT before producing text
- DeepSeek models only available in us-east-1/us-west-2, not eu-west-1

## Test Results
- 1087 unit tests: all pass
- 54 e2e tests: 45 pass in eu-west-1, 9 region-limited (expected)
- Clippy: clean
- Formatting: clean
