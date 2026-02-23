//! Gemini tool/function calling example.
//!
//! Run: cargo run --example gemini_tool_calling
//! Requires: GEMINI_API_KEY
//!
//! Shows:
//!   1. Single tool definition and invocation
//!   2. Multiple tools (model chooses which to call)
//!   3. Tool choice modes (auto, required, specific function)
//!   4. Multi-step tool use (call → result → continue)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{
    ChatMessage, CompletionOptions, FunctionDefinition, LLMProvider, ToolChoice, ToolDefinition,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GeminiProvider::from_env()?;
    println!(
        "Provider: {} | Model: {} | Function calling: {}",
        LLMProvider::name(&provider),
        LLMProvider::model(&provider),
        if provider.supports_function_calling() {
            "yes"
        } else {
            "no"
        }
    );
    println!("{}", "-".repeat(60));

    // ── 1. Single tool ──────────────────────────────────────────────────
    println!("\n=== 1. Single tool (get_weather) ===");
    {
        let tools = vec![weather_tool()];
        let messages = vec![ChatMessage::user(
            "What's the current weather in San Francisco?",
        )];

        let resp = provider
            .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
            .await?;

        if resp.has_tool_calls() {
            for tc in &resp.tool_calls {
                println!("Tool call: {}({})", tc.name(), tc.arguments());
            }
        } else {
            println!("Text response: {}", resp.content);
        }
    }

    // ── 2. Multiple tools ───────────────────────────────────────────────
    println!("\n=== 2. Multiple tools ===");
    {
        let tools = vec![weather_tool(), calculator_tool(), time_tool()];
        let messages = vec![ChatMessage::user(
            "What time is it in London and what's 42 * 17?",
        )];

        let resp = provider
            .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
            .await?;

        if resp.has_tool_calls() {
            println!("Model chose {} tool(s):", resp.tool_calls.len());
            for tc in &resp.tool_calls {
                println!("  -> {}({})", tc.name(), tc.arguments());
            }
        } else {
            println!("Text: {}", resp.content);
        }
    }

    // ── 3. Forced tool choice ───────────────────────────────────────────
    println!("\n=== 3. Force specific tool ===");
    {
        let tools = vec![weather_tool(), calculator_tool()];
        let messages = vec![ChatMessage::user("Hello, how are you?")];

        // Force the model to call calculator even though the prompt doesn't need it
        let resp = provider
            .chat_with_tools(
                &messages,
                &tools,
                Some(ToolChoice::function("calculate")),
                None,
            )
            .await?;

        if resp.has_tool_calls() {
            for tc in &resp.tool_calls {
                println!("Forced tool: {}({})", tc.name(), tc.arguments());
            }
        } else {
            println!("Text (no tool called): {}", resp.content);
        }
    }

    // ── 4. Multi-step tool use ──────────────────────────────────────────
    println!("\n=== 4. Multi-step: call → mock result → continue ===");
    {
        let tools = vec![weather_tool()];

        // Step 1: Ask about weather
        let messages = vec![ChatMessage::user(
            "What should I wear in Tokyo today? Check the weather first.",
        )];

        let resp = provider
            .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
            .await?;

        if resp.has_tool_calls() {
            println!("Step 1 - Model requests tool:");
            for tc in &resp.tool_calls {
                println!("  -> {}({})", tc.name(), tc.arguments());
            }

            // Step 2: Simulate tool result and send back
            let mock_result = serde_json::json!({
                "temperature": 22,
                "condition": "partly cloudy",
                "humidity": 65,
                "wind_speed": 12
            });

            // Build continuation messages with tool result
            let continuation = vec![
                ChatMessage::user("What should I wear in Tokyo today? Check the weather first."),
                ChatMessage::assistant("Let me check the weather in Tokyo."),
                ChatMessage::user(format!(
                    "The weather API returned: {}",
                    serde_json::to_string_pretty(&mock_result)?
                )),
            ];

            let opts = CompletionOptions {
                max_tokens: Some(150),
                ..Default::default()
            };

            let final_resp = provider.chat(&continuation, Some(&opts)).await?;

            println!("\nStep 2 - Final response with weather context:");
            println!("  {}", final_resp.content);
        } else {
            println!("Direct response: {}", resp.content);
        }
    }

    println!("\n=== Done! Tool calling features demonstrated. ===");
    Ok(())
}

// ── Tool definitions ─────────────────────────────────────────────────────

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a city".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'San Francisco'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit (default: celsius)"
                    }
                },
                "required": ["city"]
            }),
            strict: None,
        },
    }
}

fn calculator_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "calculate".to_string(),
            description: "Evaluate a mathematical expression".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '42 * 17'"
                    }
                },
                "required": ["expression"]
            }),
            strict: None,
        },
    }
}

fn time_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_current_time".to_string(),
            description: "Get the current time in a specific timezone".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, e.g. 'Europe/London'"
                    }
                },
                "required": ["timezone"]
            }),
            strict: None,
        },
    }
}
