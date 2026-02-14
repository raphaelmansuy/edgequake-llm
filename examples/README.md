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

- **streaming.rs** - Real-time streaming responses
- **tool_calling.rs** - Function/tool calling with multiple providers
- **embeddings.rs** - Text embeddings and similarity search
- **reranking.rs** - Document reranking with BM25 and RRF
- **chatbot.rs** - Interactive chatbot with conversation history
- **vision.rs** - Multimodal image analysis
- **local_llm.rs** - Using Ollama and LM Studio
- **cost_tracking.rs** - Cost monitoring and optimization
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
