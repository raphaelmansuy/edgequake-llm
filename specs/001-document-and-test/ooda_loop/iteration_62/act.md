# OODA-62 Act: FAQ Troubleshooting Expansion Complete

## Changes Made

### Updated: docs/faq.md

Expanded Troubleshooting section from 4 entries to 19 entries:

#### Authentication Errors (3)
- ✅ invalid_api_key - with environment variable check
- ✅ token expired - with refresh commands
- ✅ wrong key format - with provider format table

#### Rate Limiting Issues (3)
- ✅ immediate failures - with RateLimitedProvider example
- ✅ 429 despite limiting - with tier adjustment
- ✅ queue starvation - with configuration tips

#### Token Limit Errors (2)
- ✅ context exceeded - with truncation/chunking code
- ✅ token estimation - with Tokenizer usage

#### Network Errors (3)
- ✅ connection refused - with local provider checks
- ✅ request timeout - with timeout config and retry
- ✅ DNS failures - with resolution testing

#### Provider-Specific Issues (5)
- ✅ Ollama: model not found - with pull commands
- ✅ LMStudio: no response - with UI steps
- ✅ Gemini: permission denied - with gcloud commands
- ✅ Azure: deployment not found - with naming explanation
- ✅ xAI: model does not exist - with correct identifiers

#### Build & Development (3)
- ✅ slow compilation - with CARGO_INCREMENTAL
- ✅ doc warnings - with fix guidance
- ✅ type not in scope - with import example

## Quality Check
- All entries have error message, cause, and solution
- Code examples are runnable Rust snippets
- Cross-references to detailed docs where applicable

## Next Iteration
**OODA-63**: Add performance tuning documentation

---
*FAQ expanded from 15 to 30+ questions*
