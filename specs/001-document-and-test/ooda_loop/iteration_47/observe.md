# OODA Loop - Iteration 47 - OBSERVE

## Date: 2025-01-26

## Current Situation

Following iteration 46 where we added a local_llm example, we continue
adding examples for tool/function calling.

### Mission Deliverables Status

**Examples Progress:**
- ✅ basic_completion.rs - Basic chat completion
- ✅ multi_provider.rs - Provider abstraction
- ✅ streaming_chat.rs - Streaming responses
- ✅ embeddings.rs - Text embeddings
- ✅ reranking.rs - Document reranking
- ✅ local_llm.rs - Ollama/LM Studio
- ➡️ tool_calling.rs - Function calling (this iteration)

**Remaining Examples:**
- chatbot.rs - Interactive chat
- vision.rs - Multimodal

### Observations

1. **ToolDefinition API:**
   - `ToolDefinition::function(name, description, parameters_json)`
   - Parameters must be JSON schema with "type": "object"
   - Supports `additionalProperties: false` for strict mode

2. **ToolChoice API:**
   - `ToolChoice::auto()` - model decides
   - `ToolChoice::required()` - must use tools
   - `ToolChoice::function(name)` - specific function

3. **ChatMessage tool methods:**
   - `ChatMessage::assistant_with_tools()` for tool call responses
   - `ChatMessage::tool_result()` for returning tool outputs

4. **LLMResponse:**
   - `tool_calls: Vec<ToolCall>` contains model's requested calls
   - `finish_reason` indicates "tool_calls" when model wants to call

### Build Status

```
cargo build --examples: PASS (7 examples)
```
