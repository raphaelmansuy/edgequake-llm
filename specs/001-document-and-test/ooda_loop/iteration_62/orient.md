# OODA-62 Orient: Troubleshooting Section Design

## Section Structure

### Error Message → Solution Mapping
Each troubleshooting entry follows:
1. Error message or symptom (header)
2. Cause explanation
3. Concrete solution with code example

### Categories Added

1. **Authentication Errors** (3 entries)
   - invalid_api_key
   - token expired
   - wrong key format

2. **Rate Limiting Issues** (3 entries)
   - immediate 429 failures
   - despite rate limiting
   - queue starvation

3. **Token Limit Errors** (2 entries)
   - context exceeded
   - token estimation

4. **Network Errors** (3 entries)
   - connection refused
   - timeout
   - DNS failures

5. **Provider-Specific Issues** (5 entries)
   - Ollama: model not found
   - LMStudio: no response
   - Gemini: permission denied
   - Azure: deployment not found
   - xAI: model does not exist

### Code Examples
All solutions include runnable Rust code snippets where applicable.

### Cross-References
Link to relevant docs (providers.md, rate-limiting.md) where appropriate.
