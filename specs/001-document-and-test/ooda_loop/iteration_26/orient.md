# OODA Iteration 26 — Orient
## Analysis
TracingProvider is a decorator that adds OpenTelemetry GenAI semantic convention spans. Testing ensures delegation works correctly and constants match spec. The genai_attrs module exposes attribute names for external use. should_capture_content checks EDGECODE_CAPTURE_CONTENT env var (privacy-by-default).
