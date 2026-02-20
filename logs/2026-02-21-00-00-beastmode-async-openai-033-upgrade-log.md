# 2026-02-21 Beast Mode Log: async-openai 0.33 Upgrade

## Actions
- Created branch `upgrade/async-openai-0-33`
- Updated `Cargo.toml`: `async-openai = { version = "0.33", features = ["chat-completion", "embedding"] }`
- Rewrote `src/providers/openai.rs`: namespaced imports, removed ~150 lines of raw-HTTP workaround, native `max_completion_tokens`, cache/reasoning token extraction
- Fixed `src/error.rs`: `JSONDeserialize(err)` → `JSONDeserialize(err, _content)` (0.33 added 2nd arg)
- Added 12 new unit tests in `openai.rs` (total: 1020 unit tests, all pass)
- Created `tests/e2e_openai_033.rs` with 11 integration tests (10 `#[ignore]`d + 1 unit)
- Updated `CHANGELOG.md` with full 0.2.5 entry
- Bumped version `0.2.4` → `0.2.5` in `Cargo.toml`
- Committed and pushed branch; PR URL: https://github.com/raphaelmansuy/edgequake-llm/pull/new/upgrade/async-openai-0-33

## Decisions
- Used `features = ["chat-completion", "embedding"]` because 0.33 removed all-types default; these features enable the required API and type modules
- Removed `http_client: reqwest::Client` from `OpenAIProvider` struct; workaround needed it, but 0.33 native API doesn't
- E2E tests all use `#[ignore = "Requires OPENAI_API_KEY"]` pattern matching existing test conventions

## Next Steps
- Merge PR after optional CI run
- If published to crates.io: `cargo publish` from main after merge
- Optional: run e2e tests with `OPENAI_API_KEY cargo test --test e2e_openai_033 -- --ignored`

## Lessons/Insights
- async-openai 0.33 has significant breaking changes: type namespacing, feature flags, enum-based tools, 2-arg JSONDeserialize; required reading actual crate source, not just docs
- The `max_completion_tokens` native support in 0.33 completely eliminates the need for raw JSON injection workaround that was issue #13
