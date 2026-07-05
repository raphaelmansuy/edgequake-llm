# Provider Attribution Matrix — Official Documentation Cross-Reference

> **Legend**: ✅ Canonical documented mechanism | ⚠️ Pass-through only | ➖ No official app attribution | 🔧 edgequake-llm today
>
> **Updated**: 2026-07-05

Each row maps `ApplicationContext` fields to the **provider-official** attribution mechanism. Links are to primary vendor documentation.

---

## Summary Matrix

| Provider | App ID mechanism | Request correlation | Official doc | edgequake-llm v0.9 |
|----------|------------------|---------------------|--------------|-------------------|
| **OpenAI** | `user` body field + `X-Client-Request-Id` header | `X-Client-Request-Id` → logged server-side | [API Overview — Request IDs](https://developers.openai.com/api/reference/overview) | 🔧 No header API |
| **Azure OpenAI** | `application_name` body field | `x-ms-client-request-id` | [Chat API — application_name](https://learn.microsoft.com/en-us/azure/foundry/openai/latest) | 🔧 No |
| **Anthropic** | Passthrough custom headers (no dedicated app header) | `x-request-id` response only | [Python SDK — default/extra headers](https://platform.claude.com/docs/en/api/sdks/python) | 🔧 `with_extra_headers` |
| **Gemini (AI Studio)** | `x-goog-api-client` append | `x-goog-api-client` | [GCP System Parameters](https://cloud.google.com/apis/docs/system-parameters) | 🔧 `with_extra_headers` |
| **Vertex AI** | Same as Gemini + optional `x-goog-user-project` | Same | [GCP System Parameters](https://cloud.google.com/apis/docs/system-parameters) | 🔧 `with_extra_headers` |
| **OpenRouter** | `HTTP-Referer` + `X-OpenRouter-Title` | N/A | [App Attribution](https://openrouter.ai/docs/app-attribution) | 🔧 Partial (`X-Title` only) |
| **AWS Bedrock** | `requestMetadata` body / `X-Amzn-Bedrock-Request-Metadata` | `x-amzn-requestid` response | [Per-request metadata](https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-request-metadata.html) | 🔧 No |
| **Mistral** | Custom headers via httpx client | Passthrough | [Python SDK README — Custom HTTP Client](https://github.com/mistralai/client-python/blob/main/README.md) | 🔧 `with_extra_headers` |
| **Cohere** | `X-Client-Name` header | Passthrough | [Rerank API — X-Client-Name](https://docs.cohere.com/reference/rerank) | 🔧 No |
| **xAI** | No official app attribution header | `x-grok-conv-id` (cache routing, not app ID) | [Prompt Caching](https://docs.x.ai/developers/advanced-api-usage/prompt-caching) | 🔧 No |
| **NVIDIA NIM/NVCF** | `X-Request-Id` + `traceparent` passthrough | `NVCF-REQID` response | [NIM Logging & Observability](https://docs.nvidia.com/nim/large-language-models/latest/reference/logging-and-observability.html) | 🔧 `with_extra_headers` |
| **OpenAI-compatible** | Provider-dependent passthrough | Passthrough | De facto OpenAI headers | 🔧 `with_extra_headers` + TOML |
| **Ollama / LM Studio** | ➖ Local — no billing attribution | Optional passthrough | N/A | 🔧 No |
| **HuggingFace Inference** | ➖ Passthrough only | Passthrough | [HF Inference API](https://huggingface.co/docs/api-inference) | 🔧 No |
| **VSCode Copilot** | ➖ Hardcoded Copilot headers | Auto `x-request-id` | N/A (closed API) | 🔧 No — do not override |

---

## 1. OpenAI

### Official mechanisms

| Field | Location | Semantics | Doc |
|-------|----------|-----------|-----|
| `X-Client-Request-Id` | Request header | Caller-supplied ID; ASCII ≤512; logged by OpenAI for support lookup | [API Overview](https://developers.openai.com/api/reference/overview) |
| `user` | Request body | End-user identifier for abuse monitoring; stable per user | [Production best practices](https://developers.openai.com/api/docs/guides/production-best-practices) |
| `OpenAI-Organization` / `OpenAI-Project` | Request header | Org/project billing scope (not app attribution) | [API Overview](https://developers.openai.com/api/reference/overview) |

### Recommended `ApplicationContext` mapping

```text
app_id        → NOT sent as header; encode in X-Client-Request-Id prefix OR omit
request_id    → X-Client-Request-Id  (prefer UUID)
end_user_id   → body.user
app_name      → User-Agent suffix (optional, non-standard)
```

**SDK path**: `async-openai` supports `default_headers` on `OpenAIConfig` — inject via client builder in `OpenAIProvider`.

### Edge cases

- `X-Client-Request-Id` >512 chars → **400** from OpenAI; truncate in sanitizer.
- `user` field is for **end-user** abuse tracking, not application — document clearly in SDK.
- API key tracking dashboard (post Dec 2023) attributes by key, not app header.

---

## 2. Azure OpenAI / Microsoft Foundry

### Official mechanisms

| Field | Location | Semantics | Doc |
|-------|----------|-----------|-----|
| `application_name` | Request body | App name for Microsoft attribution; no PII | [Foundry Chat API](https://learn.microsoft.com/en-us/azure/foundry/openai/latest) |
| `end_user_id` | Request body | Entra ID object ID of end user | Same |
| `end_user_tenant_id` | Request body | M365 tenant for multi-tenant apps | Same |
| `x-ms-client-request-id` | Request header | Correlation GUID | [Azure REST headers](https://learn.microsoft.com/en-us/azure/azure-app-configuration/rest-api-headers) |

### Recommended mapping

```text
app_id        → application_name (or prefix: "{app_id}/{request_id}")
app_name      → application_name (preferred over app_id slug)
request_id    → x-ms-client-request-id
end_user_id   → end_user_id body field
tenant_id     → end_user_tenant_id (when Entra tenant known)
```

### Edge cases

- **HTTP 431**: Azure allows ~10 custom headers passthrough today; FAQ states future versions may **stop passthrough** — prefer **body fields** over custom headers for attribution.
- Managed Identity auth: attribution fields independent of auth mode (`AzureCredential` in `azure_openai.rs`).

---

## 3. Anthropic

### Official mechanisms

Anthropic does **not** document a dedicated `X-Application-Id` header. Attribution options:

| Mechanism | Doc |
|-----------|-----|
| `default_headers` / `extra_headers` on SDK | [Python SDK](https://platform.claude.com/docs/en/api/sdks/python) |
| `anthropic-beta` | Beta features only — **not** for app ID | [Beta headers](https://platform.claude.com/docs/en/api/beta-headers) |

Internal telemetry headers (`anthropic-client-version`, etc.) are SDK-managed — **do not set**.

### Recommended mapping

```text
app_id        → extra header "anthropic-app-id" (passthrough; not officially consumed)
              → OR metadata in trace only (see OTEL doc)
request_id    → extra header "x-request-id" (passthrough)
traceparent   → passthrough via extra_headers
```

**Reality check**: Anthropic usage dashboards attribute by API key. Custom headers are for **your** log correlation unless/until Anthropic documents app-level billing tags.

### edgequake-llm today

`AnthropicProvider::with_extra_headers()` merges per-request in `headers()` (~742–763 in `anthropic.rs`). Reserved: `x-api-key`, `anthropic-version`, auth headers.

---

## 4. Google Gemini (AI Studio) & Vertex AI

### Official mechanism: `x-goog-api-client`

Format: space-separated `name/semver` tokens, lowercase names with `-` only.

| Doc | URL |
|-----|-----|
| GCP system parameters | https://cloud.google.com/apis/docs/system-parameters |
| ADK reference implementation | https://github.com/google/adk-go/blob/main/model/gemini/gemini.go |

### Recommended mapping

```text
app_id   → append token "edgequake-app/{app_id}" to x-goog-api-client
           DO NOT replace SDK default; append per google/auth/metrics.py pattern
app_name → not used in header
request_id → passthrough x-request-id (non-standard but harmless)
```

**Append semantics** (from Google auth library):

```text
x-goog-api-client: gl-python/3.12 gccl/1.0 edgequake-app/my-backend
```

### Vertex-specific

- `x-goog-user-project` — billing project when using ADC; separate from app ID.
- ADC / metadata server auth (`GeminiProvider::from_env_vertex_ai`) — attribution headers still apply to outbound HTTP.

### Edge cases

- Invalid semver in custom token → may be ignored by Google telemetry; use safe slug not semver for app_id portion.
- Replacing (not appending) `x-goog-api-client` breaks SDK telemetry — **always merge**.

---

## 5. OpenRouter

### Official mechanisms

| Header | Required? | Purpose | Doc |
|--------|-----------|---------|-----|
| `HTTP-Referer` | **Yes** for app page / rankings | App URL identifier | [App Attribution](https://openrouter.ai/docs/app-attribution) |
| `X-OpenRouter-Title` | Recommended | Display name (`X-Title` legacy) | Same |
| `X-OpenRouter-Categories` | Optional | Marketplace categories | [API Reference](https://openrouter.ai/docs/api/reference/overview) |

### Recommended mapping

```text
app_url  → HTTP-Referer  (REQUIRED for OpenRouter attribution)
app_name → X-OpenRouter-Title
app_id   → embed in URL query or title suffix if URL unavailable
```

### edgequake-llm gap

Current code sets `X-Title` not `X-OpenRouter-Title` (`openrouter.rs:568–571`). **Fix**: prefer `X-OpenRouter-Title`, keep `X-Title` as fallback for backwards compatibility.

### Edge cases

- `localhost` URLs require `X-OpenRouter-Title` or tracking fails.
- Without `HTTP-Referer`, API works but **no leaderboard attribution**.

---

## 6. AWS Bedrock

### Official mechanism: request metadata

| API | Field | Format |
|-----|-------|--------|
| Converse / ConverseStream | `requestMetadata` in body | `{"app": "...", "team": "..."}` string map |
| InvokeModel / InvokeModelWithResponseStream | `X-Amzn-Bedrock-Request-Metadata` header | JSON object string |

Docs:

- [Per-request metadata tagging](https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-request-metadata.html)
- [Cost management overview](https://docs.aws.amazon.com/bedrock/latest/userguide/cost-management.html)
- [What's New — InvokeModel metadata (May 2026)](https://aws.amazon.com/about-aws/whats-new/2026/05/amazon-bedrock-request-level-usage-attribution/)

### Recommended mapping

```text
app_id    → requestMetadata["app"] or ["application_id"]
app_name  → requestMetadata["app_name"]
tenant_id → requestMetadata["tenant_id"]
request_id → NOT metadata; use AWS request ID from response for correlation
```

### Edge cases

- Metadata is **opt-in per call** — not enforced by AWS; gateway must stamp every request.
- SigV4: header must be in `SignedHeaders` — AWS SDK handles when using `request_metadata()` parameter.
- **Not** a Cost Explorer tag — join invocation logs with CUR on `requestId` for cost analysis.
- `BedrockProvider` uses Converse API — inject via `request_metadata` in SDK call, not raw HTTP.

---

## 7. Mistral

### Official mechanism

No documented `X-Application-Id`. Official SDK pattern: custom `httpx.Client` with default headers.

| Doc | URL |
|-----|-----|
| Custom HTTP client | https://github.com/mistralai/client-python/blob/main/README.md |

### Recommended mapping

```text
app_id   → X-Client-Name or X-Mistral-App-Id (passthrough convention)
app_name → X-Client-Name
request_id → x-request-id passthrough
```

`MistralProvider` chat uses inner `OpenAICompatibleProvider` — resolver must inject to **both** inner client and native embedding client.

---

## 8. Cohere

### Official mechanism: `X-Client-Name`

Documented across v1/v2 OpenAPI specs as optional header: *"The name of the project that is making the request."*

| Doc | URL |
|-----|-----|
| Chat / Rerank reference | https://docs.cohere.com/reference/rerank |
| SDK `clientName` | Cohere Java SDK builder pattern in generated clients |

### Recommended mapping

```text
app_id   → X-Client-Name (prefer app_name if set, else app_id)
app_name → X-Client-Name
```

### edgequake-llm gap

`CohereProvider` uses per-request `.header("Authorization")` only — no client name injection.

---

## 9. xAI (Grok)

### Official mechanisms

| Header/Field | Purpose | Doc |
|--------------|---------|-----|
| `x-grok-conv-id` | Cache affinity / sticky routing | [Maximizing Cache Hits](https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits) |
| `prompt_cache_key` | Responses API cache key | Same |

**No official application attribution header.** `XAIProvider` wraps `OpenAICompatibleProvider`.

### Recommended mapping

```text
app_id   → OTEL span attribute + optional passthrough custom header (unsupported officially)
request_id → x-request-id passthrough
end_user_id → N/A
```

Do **not** map `app_id` → `x-grok-conv-id` (changes routing semantics).

---

## 10. NVIDIA NIM / NVCF

### Official mechanisms

| Header | Direction | Purpose | Doc |
|--------|-----------|---------|-----|
| `X-Request-Id` | Request | Correlation; adopted by backend | [NIM Logging](https://docs.nvidia.com/nim/large-language-models/latest/reference/logging-and-observability.html) |
| `traceparent` | Request | W3C trace propagation | Same |
| `NVCF-REQID` | Response | Async poll ID (HTTP 202) | NVCF invocation docs |
| `x-multi-turn-session-id` | Request | NVCF LLM gateway sticky routing | [NVCF LLM Gateway](https://github.com/NVIDIA/nvcf/blob/main/docs/user/llm-gateway.md) |

### Recommended mapping

```text
request_id → X-Request-Id
app_id     → custom header passthrough OR X-Request-Id prefix "{app_id}/{uuid}"
traceparent → passthrough
```

Code reference: `NVIDIA_REQID_HEADER` in `src/providers/nvidia.rs:159`.

---

## 11. OpenAI-Compatible Family

Providers delegating to `OpenAICompatibleProvider`: xAI, HuggingFace, LM Studio, Mistral (chat), Nvidia (chat), POE, etc.

### Default mapping (OpenAI de facto)

```text
request_id → X-Client-Request-Id (OpenAI convention; widely accepted)
app_id     → X-Client-Name or X-Application-Id (passthrough)
traceparent → passthrough
```

Respect `ProviderConfig.headers` from TOML (`openai_compatible.rs:523–531`).

**POE note**: User-Agent customization rejected — keep reqwest default (`openai_compatible.rs:520`).

---

## 12. Local / Self-Hosted (Ollama, LM Studio)

| Attribution | Recommendation |
|-------------|----------------|
| Provider billing | ➖ N/A |
| Operational tracing | Passthrough `X-Request-Id`, `traceparent` to local server if supported |
| App ID | Set `gen_ai.application.id` OTEL span attribute only |

---

## 13. VSCode Copilot

**Do not propagate custom app headers.** `VsCodeCopilotProvider` generates protocol-specific headers (`openai-intent`, `x-request-id`, editor version) in `client.rs:454–499`. Injecting app ID may break Copilot protocol or violate ToS.

Attribution for Copilot workloads: OTEL spans only, tagged `gen_ai.application.id`.

---

## 14. Resolver Priority Rules

When multiple fields map to one header:

1. Per-request `ApplicationContext` > provider TOML override > `EDGEQUAKE_APP_*` env > catalog default.
2. `extra_headers` from caller override resolver output **except** reserved headers.
3. OpenRouter: without `app_url`, emit `AttributionWarning::OpenRouterMissingReferer`.
4. Bedrock: prefer body `requestMetadata` over header when using Converse API.

---

## 15. Compliance & PII

| Provider | Field | PII guidance |
|----------|-------|--------------|
| Azure | `application_name`, `end_user_id` | Explicit "no sensitive personal information" in docs |
| OpenAI | `user` | End-user ID for safety; not for internal app name |
| OTEL Baggage | any | Must not contain PII per W3C privacy considerations |

**Rule**: `app_id` / `app_name` must be **internal slugs**, never emails or customer names.
