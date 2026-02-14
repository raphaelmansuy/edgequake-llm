# OODA-51 Orient

## Analysis
Vision example needed two fixes:
1. **base64 dependency**: Not in Cargo.toml, needed for encoding image bytes
2. **Provider API**: `with_model()` is a builder method on `self`, not a constructor

## Root Causes
- base64 crate is standard for encoding but wasn't added since lib didn't need it
- OpenAIProvider follows builder pattern like other Rust libraries

## Impact
- After adding base64 v0.22 to deps, examples can encode images from bytes
- Fixing provider init to `new().with_model()` chain compiles correctly

## Pattern
Builder pattern: `Provider::new(api_key).with_model(model).with_option(opt)`
