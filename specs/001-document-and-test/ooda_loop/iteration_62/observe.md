# OODA-62 Observe: FAQ Troubleshooting Assessment

## Current FAQ State
- 15 existing questions across General, Providers, Caching, Cost Tracking, Rate Limiting, Streaming, Observability, Testing sections
- Small Troubleshooting section with 4 basic entries

## User Pain Points (from error.rs and support patterns)

### Authentication Issues
- Invalid API key formats
- Token expiration
- Wrong provider key

### Rate Limiting Issues
- Not using RateLimitedProvider
- Incorrect tier limits
- Queue starvation

### Token Limit Errors
- Context window exceeded
- No token estimation before sending

### Network Issues
- Connection refused (local providers)
- Timeouts on long requests
- DNS failures

### Provider-Specific Issues
- Ollama: model not found
- LMStudio: no model loaded
- Gemini: permission denied
- Azure: deployment not found
- xAI: incorrect model names

## Enhancement Opportunity
Expand troubleshooting section with specific error messages, causes, and solutions.
