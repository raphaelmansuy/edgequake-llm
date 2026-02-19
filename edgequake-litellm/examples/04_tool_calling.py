"""
04_tool_calling.py — Function / tool calling example.

Run:
    python examples/04_tool_calling.py
"""
from __future__ import annotations

import json
import os

import edgequake_litellm as litellm


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI-compatible schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Paris'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL'",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Simulated tool implementations
# ---------------------------------------------------------------------------

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Simulated weather API."""
    return {
        "city": city,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "Sunny",
    }


def get_stock_price(ticker: str) -> dict:
    """Simulated stock price API."""
    prices = {"AAPL": 189.50, "GOOGL": 175.20, "MSFT": 425.30}
    return {
        "ticker": ticker,
        "price": prices.get(ticker.upper(), 100.00),
        "currency": "USD",
    }


def call_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        result = get_weather(**args)
    elif name == "get_stock_price":
        result = get_stock_price(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try Mistral — also supports tool calling
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            print("Set OPENAI_API_KEY or MISTRAL_API_KEY to run this example.")
            return
        model = "mistral/mistral-small-latest"
    else:
        model = "openai/gpt-4o-mini"

    messages: list[dict] = [
        {"role": "user", "content": "What's the weather in Tokyo and the AAPL stock price?"},
    ]

    print(f"Using model: {model}")
    print(f"User: {messages[0]['content']}\n")

    # First call — let the model decide to use tools
    resp = litellm.completion(model, messages, tools=TOOLS, tool_choice="auto")

    if resp.tool_calls:
        print(f"Model wants to call {len(resp.tool_calls)} tool(s):")
        tool_results = []
        for tc in resp.tool_calls:
            args = json.loads(tc.function_arguments)
            print(f"  → {tc.function_name}({args})")
            result = call_tool(tc.function_name, args)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Second call — provide tool results and get final answer
        messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function_name,
                    "arguments": tc.function_arguments,
                },
            }
            for tc in resp.tool_calls
        ]})
        messages.extend(tool_results)

        final = litellm.completion(model, messages, tools=TOOLS)
        print(f"\nAssistant: {final.content}")
    else:
        print(f"Assistant: {resp.content}")


if __name__ == "__main__":
    main()
