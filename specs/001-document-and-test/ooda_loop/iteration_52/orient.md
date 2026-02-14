# OODA-52 Orient

## Analysis
Cost tracking example demonstrates:
- ModelPricing with cache support
- Multiple API call simulations
- Budget management
- Operation type tracking via CostEntry

## API Learnings
- `SessionCostTracker::with_budget(f64)` - constructor with budget
- `record_usage()` - track regular usage
- `record_usage_with_cache()` - track cached calls
- `is_over_budget()` / `is_near_budget()` - budget checks
- `summary()` - aggregated statistics

## Value Added
Cost tracking is critical for enterprise use - this example shows all key patterns.
