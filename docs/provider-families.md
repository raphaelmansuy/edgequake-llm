# LLM Provider Families: A Deep Comparison

EdgeQuake LLM supports multiple LLM providers, each with unique API patterns,
capabilities, and trade-offs. This document provides an in-depth comparison to
help you choose the right provider for your use case.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM Provider Family Tree                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   OpenAI        │  │   Anthropic     │  │   Google        │          │
│  │   Family        │  │   Family        │  │   Family        │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│           │                    │                    │                    │
│           ▼                    ▼                    ▼                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ • OpenAI        │  │ • Anthropic     │  │ • Gemini        │          │
│  │ • Azure OpenAI  │  │                 │  │ • Vertex AI     │          │
│  │ • OpenRouter    │  │                 │  │                 │          │
│  │ • xAI           │  │                 │  │                 │          │
│  │ • DeepSeek      │  │                 │  │                 │          │
│  │ • Together AI   │  │                 │  │                 │          │
│  │ • Local (LM     │  │                 │  │                 │          │
│  │   Studio,       │  │                 │  │                 │          │
│  │   Ollama)       │  │                 │  │                 │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## API Format Comparison

### OpenAI Family

**Base Format**: Chat Completions API (`/v1/chat/completions`)

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "tools": [...],
  "tool_choice": "auto"
}
```

**Unique Features**:
- `response_format: {type: "json_object"}` for guaranteed JSON output
- `logprobs` for token probability distribution
- `seed` for deterministic outputs
- Fine-tuning API support
- Multimodal (vision) via content arrays

**EdgeQuake Usage**:
```rust
use edgequake_llm::OpenAIProvider;

let provider = OpenAIProvider::from_env()?;
let response = provider.chat(&messages, Some(CompletionOptions {
    temperature: Some(0.7),
    json_mode: Some(true),  // Maps to response_format
    ..Default::default()
})).await?;
```

### Anthropic Family

**Base Format**: Messages API (`/v1/messages`)

```json
{
  "model": "claude-sonnet-4-5-20250929",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "system": "System instruction here",  // Separate from messages
  "max_tokens": 1000,
  "tools": [...],
  "tool_choice": {"type": "auto"}
}
```

**Unique Features**:
- System instruction is a separate top-level field (not in messages array)
- Extended thinking mode (`thinking` content blocks)
- Prompt caching (system + tool definitions cached)
- Multimodal uses `source: {type: "base64", ...}` format
- Content blocks can be text, tool_use, or tool_result

**EdgeQuake Usage**:
```rust
use edgequake_llm::AnthropicProvider;

let provider = AnthropicProvider::from_env()?;
// System message is automatically extracted and placed in `system` field
let response = provider.chat(&[
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("Hello!"),
], None).await?;
```

### Gemini Family

**Base Format**: GenerateContent API

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "..."}]
    },
    {
      "role": "model",  // Note: "model" not "assistant"
      "parts": [{"text": "..."}]
    }
  ],
  "systemInstruction": {
    "parts": [{"text": "System instruction"}]
  },
  "generationConfig": {
    "maxOutputTokens": 1000,
    "temperature": 0.7
  },
  "tools": [{"functionDeclarations": [...]}]
}
```

**Unique Features**:
- Uses `parts` array for content (supports multimodal)
- System instruction via `systemInstruction` field
- Role names differ: `user`, `model` (not `assistant`)
- Context caching API for long system prompts
- Thinking mode for Gemini 2.5+/3.x (though disabled via API currently)
- Image format: `inlineData: {mimeType: "...", data: "base64..."}`

**EdgeQuake Usage**:
```rust
use edgequake_llm::GeminiProvider;

let provider = GeminiProvider::from_env()?;
// Role conversion handled automatically (assistant -> model)
let response = provider.chat(&messages, None).await?;

// List available models
let models = provider.list_models().await?;
```

## Feature Comparison Matrix

| Feature | OpenAI | Anthropic | Gemini |
|---------|--------|-----------|--------|
| Streaming | Yes | Yes | Yes |
| Function Calling | Yes | Yes | Yes |
| JSON Mode | Yes | No* | Yes |
| Vision/Images | Yes | Yes | Yes |
| Extended Thinking | No | Yes | Yes** |
| Prompt Caching | No*** | Yes | Yes |
| Token Counting | Yes | Yes | Yes |
| Embeddings | Yes | No | Yes |

*Anthropic requires structured prompting for JSON
**Gemini thinking currently disabled via REST API
***OpenAI has prompt caching via fine-tuning only

## Image Format Differences

```
┌─────────────────────────────────────────────────────────────────────┐
│                     IMAGE FORMAT COMPARISON                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  OpenAI:                                                             │
│  ┌────────────────────────────────────────┐                         │
│  │ {                                       │                         │
│  │   "type": "image_url",                  │                         │
│  │   "image_url": {                        │                         │
│  │     "url": "data:image/png;base64,...", │                         │
│  │     "detail": "high"                    │                         │
│  │   }                                     │                         │
│  │ }                                       │                         │
│  └────────────────────────────────────────┘                         │
│                                                                      │
│  Anthropic:                                                          │
│  ┌────────────────────────────────────────┐                         │
│  │ {                                       │                         │
│  │   "type": "image",                      │                         │
│  │   "source": {                           │                         │
│  │     "type": "base64",                   │                         │
│  │     "media_type": "image/png",          │                         │
│  │     "data": "..."                       │                         │
│  │   }                                     │                         │
│  │ }                                       │                         │
│  └────────────────────────────────────────┘                         │
│                                                                      │
│  Gemini:                                                             │
│  ┌────────────────────────────────────────┐                         │
│  │ {                                       │                         │
│  │   "inlineData": {                       │                         │
│  │     "mimeType": "image/png",            │                         │
│  │     "data": "..."                       │                         │
│  │   }                                     │                         │
│  │ }                                       │                         │
│  └────────────────────────────────────────┘                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**EdgeQuake Abstraction**:
```rust
use edgequake_llm::traits::ImageData;

// Unified interface - EdgeQuake handles format conversion
let image = ImageData::new(base64_data, "image/png")
    .with_detail("high");  // OpenAI-specific, ignored by others

let message = ChatMessage::user_with_images("What's in this image?", vec![image]);

// Works with any provider
let response = provider.chat(&[message], None).await?;
```

## Tool/Function Calling Differences

### OpenAI Format
```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather for location",
      "parameters": {"type": "object", "properties": {...}}
    }
  }],
  "tool_choice": "auto"  // "auto" | "required" | "none" | {"type": "function", "function": {"name": "..."}}
}
```

### Anthropic Format
```json
{
  "tools": [{
    "name": "get_weather",
    "description": "Get weather for location",
    "input_schema": {"type": "object", "properties": {...}}
  }],
  "tool_choice": {"type": "auto"}  // "auto" | "any" | "none" | {"type": "tool", "name": "..."}
}
```

### Gemini Format
```json
{
  "tools": [{
    "functionDeclarations": [{
      "name": "get_weather",
      "description": "Get weather for location",
      "parameters": {"type": "object", "properties": {...}}
    }]
  }],
  "toolConfig": {
    "functionCallingConfig": {
      "mode": "AUTO",  // "AUTO" | "ANY" | "NONE"
      "allowedFunctionNames": ["get_weather"]
    }
  }
}
```

**EdgeQuake Abstraction**:
```rust
use edgequake_llm::traits::{ToolDefinition, FunctionDefinition, ToolChoice};

let tools = vec![ToolDefinition {
    tool_type: "function".to_string(),
    function: FunctionDefinition {
        name: "get_weather".to_string(),
        description: "Get weather for location".to_string(),
        parameters: serde_json::json!({...}),
        strict: None,
    },
}];

let response = provider.chat_with_tools(
    &messages,
    &tools,
    Some(ToolChoice::auto()),  // Converted to provider-specific format
    None,
).await?;
```

## Best Use Cases

### OpenAI Family
- **Best for**: General-purpose tasks, JSON output, deterministic responses
- **Choose when**: You need structured output (JSON mode) or reproducibility (seed)
- **Providers**: OpenAI, Azure OpenAI (enterprise), xAI, DeepSeek (reasoning)

### Anthropic Family  
- **Best for**: Complex reasoning, long context, conversational AI
- **Choose when**: You need extended thinking, prompt caching, or large context
- **Providers**: Anthropic (Claude models), Ollama (local compatibility)

### Gemini Family
- **Best for**: Multimodal tasks, large context windows (2M tokens)
- **Choose when**: Processing long documents, needing largest context, or using GCP
- **Providers**: Gemini API (Google AI), Vertex AI (enterprise)

## What's Missing from the Unified Interface

### Current Gaps

1. **Provider-Specific Options**
   - OpenAI: `logprobs`, `seed`, `logit_bias`
   - Anthropic: Extended thinking configuration
   - Gemini: Safety settings, context caching API

2. **Streaming Differences**
   - Anthropic: `thinking_delta` events not propagated
   - Gemini: Thought summaries not accessible

3. **Response Metadata**
   - OpenAI: Log probabilities
   - Anthropic: Cache hit/miss statistics
   - Gemini: Cached token counts

### Roadmap

**Phase 1: Provider Extensions**
- Add `LLMProviderOpenAI` trait for OpenAI-specific features
- Add `LLMProviderAnthropic` trait for Anthropic-specific features  
- Add `LLMProviderGemini` trait for Gemini-specific features

**Phase 2: Enhanced Streaming**
- Propagate thinking/reasoning content in `StreamChunk`
- Add cache statistics to response metadata

**Phase 3: Advanced Features**
- OpenAI: Fine-tuning API support
- Anthropic: Prompt caching management API
- Gemini: Context caching API

## Related Documentation

- [providers.md](providers.md) - Provider setup and configuration
- [architecture.md](architecture.md) - System design overview
- [testing.md](testing.md) - Testing strategies for each provider
