# OODA-51 Decide

## Decision
Fix vision.rs compilation errors:
1. Add `base64 = "0.22"` to Cargo.toml dependencies
2. Change provider init from `with_model(&api_key, "gpt-4o")` to `new(&api_key).with_model("gpt-4o")`

## Rationale
- base64 is lightweight (no heavy deps), standard for image encoding
- Builder pattern is idiomatic Rust and consistent with library design

## Next Steps
1. Apply both fixes via multi_replace
2. Build to verify
3. Run tests to ensure no regressions
4. Update examples/README.md
5. Commit iteration
