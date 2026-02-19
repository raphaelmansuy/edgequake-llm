# Task Log: CI/CD Pipeline Fixes

**Date**: 2026-02-19 14:30  
**Branch**: main (merged from fix/vision-llm)

## Actions
- Diagnosed CI run triggers not firing after pushes (GitHub event processing delay / cached web pages)
- Force-triggered CI by close+reopen of PR #5 and empty commits
- Identified root causes: (1) Cargo.lock gitignored → --locked failed; (2) MSRV 1.75.0 < actual dep min 1.83; (3) rustsec/audit-check@v2 fails on push events with insufficient Check API permissions
- Removed Cargo.lock from .gitignore, committed it (2414 lines)
- Updated rust-version from 1.75.0 → 1.82.0 → 1.83.0 via cargo metadata analysis
- Replaced rustsec/audit-check@v2 with direct `cargo audit` in both ci.yml and publish.yml
- Resolved cherry-pick merge conflicts in openai.rs and azure_openai.rs (formatting-only differences)
- Merged PR #5 (squash) and pushed CI fixes directly to main
- All 7 CI jobs pass on main: Rustfmt, Security audit, Docs, MSRV, Clippy, Test (ubuntu+macos)

## Decisions
- Cargo.lock should be committed even for libraries when CI uses --locked (conventional ≠ always correct for reproducible CI)
- MSRV 1.83.0 is the actual minimum from transitive dependency chain: reqwest→url→idna→idna_adapter→icu_normalizer (2.1.1 requires 1.83)
- wasip2/wit-bindgen require 1.87 but are WASM-only targets, excluded from MSRV calculation
- cargo audit directly is more reliable than rustsec/audit-check@v2 which needs checks:write permission

## Next Steps
- Add CARGO_REGISTRY_TOKEN secret to GitHub repo for automated publishing via publish.yml
- Optionally create `crates-io` GitHub Environment for manual approval gate on publish
- Consider bumping patch version if publishing a new release

## Lessons/Insights
- fetch_webpage returns heavily cached GitHub HTML; use GitHub REST API directly for real-time CI status
- For MSRV enforcement with --locked, the effective MSRV is max(own code MSRV, all non-platform-specific dep MSRVs)
- CI triggers can silently fail to fire for synchronize events; fall back to close+reopen PR to force an opened event
