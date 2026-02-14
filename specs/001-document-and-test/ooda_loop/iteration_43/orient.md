# OODA Iteration 43 - Orient

## First Principles: Example Value

### Why Examples Matter

1. **Learning Curve**: Users learn from examples faster than docs
2. **Copy-Paste Ready**: Functional code users can adapt
3. **Integration Proof**: Shows the library works
4. **Feature Discovery**: Exposes capabilities users might miss

### Streaming Example Value

Streaming is critical for:
- **UX**: Real-time feedback for long generations
- **Latency**: Start displaying before completion
- **Memory**: Process large responses incrementally

### Implementation Choice

Used `stream()` method (simple prompt-based) vs `chat_with_tools_stream()`:
- Simpler API for beginners
- Demonstrates core streaming concept
- Can be extended in future examples

### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Example won't compile | Verified with `cargo build --example streaming_chat` |
| API mismatch | Used actual trait method signature |
| Missing deps | Uses only existing crate deps (futures) |
