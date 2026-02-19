"""
01_basic_completion.py — Synchronous completion with various providers.

Run:
    python examples/01_basic_completion.py
"""
from __future__ import annotations

import os
import sys

# Drop-in replacement for litellm
import edgequake_litellm as litellm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def demo(title: str, provider: str, model: str, env_var: str) -> None:
    api_key = os.environ.get(env_var)
    if not api_key:
        print(f"[SKIP] {title}: {env_var} not set")
        return

    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": "What is the capital of France?"}]
    resp = litellm.completion(model, messages)

    print(f"Model   : {resp.model}")
    print(f"Content : {resp.content}")
    print(f"Tokens  : {resp.usage.total_tokens} total "
          f"({resp.usage.prompt_tokens} prompt + {resp.usage.completion_tokens} completion)")


# ---------------------------------------------------------------------------
# Run demos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo("OpenAI GPT-4o mini",   "openai",    "openai/gpt-4o-mini",              "OPENAI_API_KEY")
    demo("Anthropic Claude 3",   "anthropic", "anthropic/claude-3-haiku-20240307","ANTHROPIC_API_KEY")
    demo("Google Gemini Flash",  "gemini",    "gemini/gemini-2.0-flash",          "GEMINI_API_KEY")
    demo("Mistral Small",        "mistral",   "mistral/mistral-small-latest",     "MISTRAL_API_KEY")
    demo("xAI Grok Beta",        "xai",       "xai/grok-beta",                    "XAI_API_KEY")
    demo("OpenRouter Llama Free","openrouter","openrouter/meta-llama/llama-3.1-8b-instruct:free","OPENROUTER_API_KEY")

    print("\nDone.")
