# Iteration 08 - Orient

## Analysis

The observability subsystem is the most layered feature in edgequake-llm, spanning four distinct components that serve different audiences:

1. **TracingProvider** - developer/ops: distributed tracing for debugging production issues
2. **GenAI Events** - analytics: structured event data for ML observability platforms
3. **InferenceMetrics** - UX: real-time streaming metrics for display layers
4. **Middleware** - aggregate: request-level logging and session-wide counters

No single existing doc covers how these pieces compose together. A unified observability doc with clear separation of concerns is the highest-signal deliverable.

## Key Design Decisions (WHY)
- Privacy-by-default: content not captured without explicit opt-in (GDPR, enterprise requirements)
- Sentinel values for optional fields: tracing-opentelemetry has issues with `Option<T>` in span attributes
- 10 KB truncation on thinking content: prevents trace storage explosion with reasoning models
- AtomicU64 for metrics: lock-free concurrent access without mutex contention
- Provider TTFT priority: network latency should not inflate time-to-first-token measurements

## Risks
- OpenTelemetry GenAI conventions are still evolving; attribute names may change
- Current doc needs to be clear about experimental vs stable attributes
