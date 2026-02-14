# OODA Iteration 05 - Observe

## Mission Re-Read
Re-read `specs/001-document-and-test.md`. Focus: Create `docs/cost-tracking.md`.

## Cost Tracker Module (`src/cost_tracker.rs`, 712 lines)

### Key Types
- `ModelPricing`: Per-million-token pricing (input, output, cached)
- `CostEntry`: Single API call record (model, provider, tokens, cost, timestamp)
- `CostSummary`: Aggregated session stats (by model, provider, operation)
- `SessionCostTracker`: Main tracker with pricing, budget, entries

### Features
- Default pricing for major models (GPT-4o, Claude, Gemini, etc.)
- Budget limits with warning thresholds
- Cache-aware cost calculation
- Breakdowns by model, provider, operation
- Helper functions: format_cost(), format_tokens()

### Default Pricing Table (verified from code)
- GPT-4o: $2.50/$10.00 per 1M tokens
- Claude Sonnet 4.5: $3.00/$15.00 per 1M tokens
- Gemini 2.5 Flash: $0.15/$0.60 per 1M tokens
