# Provider Audit — October 2026

> **Scope**: Gap analysis of edgequake-llm v0.10.8 against official provider catalogs as of **2026-10-01**, focused on model IDs, defaults, context windows, reasoning clamps, and Azure API versioning.
>
> **Predecessor**: [01-PROVIDER-AUDIT-JULY-2026.md](./01-PROVIDER-AUDIT-JULY-2026.md) (feature gaps; fully implemented).
>
> **Method**: Source-code defaults and static registries cross-referenced against live provider documentation on 2026-10-01. A second pass corrected prices, Luna's context, the Gemini 3.6 thinking clamp, and the GPT-6 catalog.

---

## 1. Executive Summary

| Area | Official state on 2026-10-01 | Decision in this crate |
|------|------------------------------|------------------------|
| OpenAI chat default | GPT-6 Astra / GPT-6.1 Sol / GPT-6 Luna are the current lineup | Stay on `gpt-5.6-terra` |
| Why not GPT-6.1 Sol | Sol's page: Chat Completions has no tool calling. Luna: Chat Completions function calling only at effort `none` | Tools stay on Chat Completions |
| OpenAI image | `gpt-image-2.5-flare` (everyday) and `gpt-image-2.5-sunburst` (quality) | Default `gpt-image-2.5-flare` |
| Anthropic | Start with Opus 5.5; speed tier Sonnet 5.5; Fable 5.1 | Default `claude-sonnet-5-5` |
| Gemini | `gemini-3.8-flash` | Default `gemini-3.8-flash` |
| xAI | `grok-4.7` recommended | Default `grok-4.7` |
| Azure API | Foundry `/openai/v1` | Empty `AZURE_OPENAI_API_VERSION` uses v1 |
| Mistral / Cohere | Medium 3.5 and Command A+ still current | No default change |
| `async-openai` | crates.io `0.42.1` (2026-09-28) | Stay on `0.34.0` |

---

## 2. OpenAI

**Sources**

- [Models](https://developers.openai.com/api/docs/models)
- [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
- [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra)
- [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [Image prompting](https://developers.openai.com/api/docs/guides/image-prompting)

| Model | Context | Max output | Input / output per 1M | Effort |
|-------|---------|------------|------------------------|--------|
| `gpt-6-astra` | 1,050,000 | 128,000 | $10 / $50 | `low` `medium` `high` `xhigh` `max` |
| `gpt-6.1-sol` | 1,050,000 | 128,000 | $2 / $10 | same; omitted default `medium`; no `none` |
| `gpt-6-luna` | 1,050,000 | 128,000 | $0.10 / $0.50 | `none`…`max`; omitted default `medium` |
| `gpt-5.6-sol` / `gpt-5.6` | 1,050,000 | 128,000 | $4 / $20 | `none`…`max` |
| `gpt-5.6-terra` | 1,050,000 | 128,000 | $2 / $12 | `none`…`max`; omitted default `medium` |
| `gpt-5.6-luna` | 1,050,000 | 128,000 | $0.20 / $1.20 | `none`…`max` |

The first October pass stored Luna at 400,000 tokens and Sol/Terra/Luna at $5/$30, $2.50/$15, and $1/$6. Those figures came from an upgrade note and a launch list. The live model pages contradict them. The registry now follows the model pages.

Astra's page does not state an omitted-effort default, so `default_when_omitted` is unset for `gpt-6-astra`.

Chat Completions is listed as an endpoint for these models. Tool calling is the constraint: Sol requires the Responses API for tools, and Luna allows Chat Completions function calling only when `reasoning_effort` is `none`. This crate's chat path is Chat Completions, so the default remains `gpt-5.6-terra`.

Image requests for `gpt-image-2.5-*` send `output_format` (`png` when unset). `gpt-image-2` still sends `response_format: b64_json`.

`async-openai` 0.34's `ReasoningEffort` enum is `None | Minimal | Low | Medium | High | Xhigh`. Effort `max` is therefore sent as `xhigh`.

---

## 3. Anthropic

**Source**: [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)

| Item | Value |
|------|-------|
| Default | `claude-sonnet-5-5` |
| Also registered | `claude-fable-5-1`, `claude-opus-5-5`, `claude-opus-5` |
| Context | 1,000,000 for Fable 5.1, Opus 5, Opus 5.5, Sonnet 5.5 |
| Opus 5.5 omitted effort | `medium` |
| Other 5.x omitted effort | `high` |
| `anthropic-version` | `2023-06-01` |
| Haiku 4.5 | Still the fast model. Retirement commitment is not sooner than 2026-10-15 |

Factory Anthropic fallback and OpenRouter default are `claude-sonnet-5-5` / `anthropic/claude-sonnet-5-5`.

---

## 4. Google Gemini

**Source**: [Thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking)

| Model | `minimal` | Default thinking |
|-------|-----------|------------------|
| `gemini-3.8-flash` | Rejected | `medium` |
| `gemini-3.7-flash` | Rejected | `medium` |
| `gemini-3.6-flash` | Supported | `medium` |
| `gemini-3.5-flash` | Supported | `medium` |
| `gemini-3.5-flash-lite` | Supported, and it is the model's default | — |

The first October pass clamped 3.6 the same way as 3.7 and 3.8. That was wrong. Only 3.7 and 3.8 map `minimal` and `none` to `low`.

Chat default is `gemini-3.8-flash`. Context for the 3.5–3.8 Flash profiles is 1,048,576. `auto_preview_suffix` is off for 3.x. Embedding default remains `gemini-embedding-2`. Image default remains `gemini-3.1-flash-image`.

Intro price on the thinking page for 3.8 Flash is $0.75 / $3.75 per 1M through 2026-12-31, then $1.50 / $7.50.

---

## 5. xAI

**Source**: [Models](https://docs.x.ai/developers/models)

`grok-4.7` is the recommended model. Context is 500,000. Under 200k prompt tokens the list price is $2 input / $6 output. Image default is `grok-imagine-image-2.0`.

Effort: 4.5 is `low|medium|high`; 4.6 and 4.7 add `xhigh`. `none` is omitted for 4.5+.

---

## 6. Azure OpenAI

**Source**: [API version lifecycle](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)

Empty `AZURE_OPENAI_API_VERSION` uses `{endpoint}/openai/v1` with no `api-version` query. A dated value keeps the deployment URL. Image generation follows the same split. The v1 image body includes `model`.

The Azure image default deployment name stays `gpt-image-2` because that string is the user's deployment name, not an OpenAI model alias.

---

## 7. Still current

- **Mistral**: overview still leads with Mistral Medium 3.5. Default remains `mistral-medium-3-5`. Mistral Small 4 is listed, but this pass did not change the default.
- **Cohere**: models page still lists `command-a-plus-05-2026` as live. Default unchanged. Embedding default remains `embed-v4.0`.
- **NVIDIA NIM / Hugging Face**: no single canonical successor ID. Defaults unchanged (`nvidia/llama-3.3-nemotron-super-49b-v1`, `meta-llama/Meta-Llama-3.1-70B-Instruct`).

---

## 8. Explicitly deferred

1. OpenAI Responses API as the tool-calling path for GPT-6.
2. Gemini Interactions API.
3. Bumping `async-openai` to 0.42.1 (reqwest 0.13 / thiserror 2).
4. NVIDIA NIM and Hugging Face default refreshes.
