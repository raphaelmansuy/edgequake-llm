# OODA Iteration 01 - Decide

## Priority Actions

1. **Fix 2 failing xAI tests** (HIGH - blocking all-green test suite)
   - Update test assertions to use correct model name `"grok-4-1-fast"` (dashes, not dots)
   - Verify fix with `cargo test`

2. **Run `cargo doc --no-deps`** (MEDIUM - establish baseline)
   - Check for warnings
   - Document any issues

3. **Install coverage tool** (MEDIUM - needed for metric tracking)
   - Install `cargo-tarpaulin`
   - Run initial coverage measurement

4. **Create `./docs/` directory structure** (MEDIUM - foundation for docs)
   - Create placeholder files for all 10 required docs

## Rationale

Fix first, measure second, build third. This follows the principle of ensuring
a solid foundation before building on it.
