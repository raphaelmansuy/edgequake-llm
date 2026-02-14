# EdgeQuake LLM Examples

This directory contains runnable examples demonstrating various capabilities
of the EdgeQuake LLM library.

## Prerequisites

Before running examples, ensure you have the necessary API keys set as
environment variables:

```bash
# For OpenAI examples
export OPENAI_API_KEY="sk-..."

# For Anthropic (Claude) examples
export ANTHROPIC_API_KEY="sk-ant-..."

# For Google Gemini examples
export GEMINI_API_KEY="..."
# Or for Vertex AI:
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-central1"  # Optional, defaults to us-central1

# For xAI examples
export XAI_API_KEY="xai-..."

# For HuggingFace examples
export HUGGINGFACE_TOKEN="hf_..."

# For local providers (Ollama, LM Studio)
# No API key needed - ensure server is running
```

## Running Examples

All examples can be run using Cargo:

```bash
# Basic completion
cargo run --example basic_completion

# Multi-provider abstraction
cargo run --example multi_provider

# Streaming responses
cargo run --example streaming_chat

# Text embeddings and similarity
cargo run --example embeddings

# Document reranking (no API key needed)
cargo run --example reranking

# Local LLMs (Ollama/LM Studio)
cargo run --example local_llm

# Tool/function calling
cargo run --example tool_calling

# Interactive chatbot
cargo run --example chatbot

# Vision/multimodal image analysis
cargo run --example vision

# Cost tracking and budget management
cargo run --example cost_tracking
```

## Available Examples

### basic_completion.rs

A simple example showing basic chat completion with OpenAI.

**Demonstrates:**
- Creating an OpenAI provider
- Sending a chat message
- Receiving and displaying the response
- Token usage tracking

**Run:**
```bash
export OPENAI_API_KEY="your-key"
cargo run --example basic_completion
```

### multi_provider.rs

Shows how EdgeQuake LLM abstracts multiple providers behind a unified interface.

**Demonstrates:**
- Provider abstraction (`Box<dyn LLMProvider>`)
- Creating multiple provider instances
- Comparing responses from different LLMs
- Graceful handling of missing API keys

**Run:**
```bash
# Set any combination of these (at least one)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

cargo run --example multi_provider
```

### streaming_chat.rs

Demonstrates async streaming responses with real-time output.

**Demonstrates:**
- Setting up a streaming completion
- Processing chunks as they arrive
- Real-time token output
- Proper stream handling and cleanup

**Run:**
```bash
export OPENAI_API_KEY="your-key"
cargo run --example streaming_chat
```

### embeddings.rs

Demonstrates text embedding generation and semantic similarity search.

**Demonstrates:**
- Creating an embedding provider
- Generating embeddings for text documents
- Batch embedding generation
- Cosine similarity calculation
- Basic semantic search ranking

**Run:**
```bash
export OPENAI_API_KEY="your-key"
cargo run --example embeddings
```

### reranking.rs

Demonstrates document reranking to improve search result relevance.

**Demonstrates:**
- BM25 reranking (local, no API key needed)
- Different presets for different use cases
- Reciprocal Rank Fusion (RRF) for combining rankings
- Score-based result ranking

**Run:**
```bash
# No API key required - uses local BM25 algorithm
cargo run --example reranking
```

### local_llm.rs

Demonstrates using local LLM providers (Ollama and LM Studio).

**Demonstrates:**
- Creating Ollama and LM Studio providers
- Checking local server availability
- Unified interface across local providers
- No cloud API keys required

**Run:**
```bash
# No API key required - needs Ollama or LM Studio running locally
# For Ollama: ollama pull llama3.2 && ollama serve
cargo run --example local_llm
```

### tool_calling.rs

Demonstrates function/tool calling with LLM providers.

**Demonstrates:**
- Defining tools with JSON schemas
- Allowing the model to call functions
- Processing tool calls and returning results
- Multi-turn tool calling conversation

**Run:**
```bash
export OPENAI_API_KEY="your-key"
cargo run --example tool_calling
```

### chatbot.rs

Demonstrates an interactive chatbot with conversation history.

**Demonstrates:**
- Multi-turn conversation with memory
- User input handling
- System prompt for personality
- Token usage tracking across turns

**Run:**
```bash
export OPENAI_API_KEY="your-key"
cargo run --example chatbot
```

### vision.rs

Demonstrates multimodal image analysis with vision-capable models.

**Demonstrates:**
- Loading and encoding images as base64
- Creating multimodal messages with ImageData
- Analyzing images with GPT-4V/GPT-4o
- Detail levels (auto/low/high) for quality/cost trade-offs

**Run:**
```bash
export OPENAI_API_KEY="your-key"
cargo run --example vision
```

### cost_tracking.rs

Demonstrates session-level cost tracking and budget management.

**Demonstrates:**
- Setting up a cost tracker with budget limits
- Recording API usage and calculating costs  
- Getting summaries by model, provider, and operation
- Budget alerts and warnings when approaching limits
- Cache savings estimation

**Run:**
```bash
# No API key required - uses simulated costs
cargo run --example cost_tracking
```

## Example Output

### basic_completion

```
EdgeQuake LLM - Basic Completion Example

Sending request to OpenAI...

Response: Paris
Tokens: 15 prompt + 2 completion = 17 total
```

### multi_provider

```
EdgeQuake LLM - Multi-Provider Example

Testing: openai
   Model: gpt-4
   Response: Async/await in Rust provides a way to write...
   Tokens: 52

Testing: anthropic
   Model: claude-sonnet-4-5-20250929
   Response: Async/await is Rust's way of writing...
   Tokens: 48

Testing: gemini
   Model: gemini-2.5-flash
   Response: Async/await allows Rust programs to...
   Tokens: 45
```

## Planned Future Examples

The following examples are planned for future iterations:

- **retry_handling.rs** - Error handling with retry strategies
- **middleware.rs** - Custom middleware (logging, caching)

## Related Documentation

- [Provider Families](../docs/provider-families.md) - Comparison of provider APIs
- [Providers Guide](../docs/providers.md) - Setting up each provider
- [Architecture](../docs/architecture.md) - System design overview
- [Reranking](../docs/reranking.md) - Reranking strategies

## Contributing Examples

When adding new examples:

1. Create a new `.rs` file in this directory
2. Add module documentation at the top explaining the example
3. Include the command to run in the doc comment
4. Update this README with the new example
5. Ensure the example compiles and runs without errors
