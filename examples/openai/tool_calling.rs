//! Tool calling example
//!
//! Demonstrates function/tool calling with LLM providers.
//!
//! Run with: cargo run --example openai_tool_calling
//! Requires: OPENAI_API_KEY environment variable
//!
//! This example shows:
//! - Defining tools for the model to use
//! - Allowing the model to call functions
//! - Processing tool calls and returning results
//! - Multi-turn tool calling conversation

use edgequake_llm::{
    ChatMessage, CompletionOptions, LLMProvider, OpenAIProvider, ToolChoice, ToolDefinition,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Initialize provider
    let provider = OpenAIProvider::new(&api_key);

    println!("🔧 EdgeQuake LLM - Tool Calling Example\n");
    println!("Provider: {}", provider.name());
    println!("Model: {}", provider.model());
    println!("{}", "─".repeat(60));

    // Define available tools
    let tools = vec![
        ToolDefinition::function(
            "get_weather",
            "Get the current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g., 'Paris, France'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"],
                "additionalProperties": false
            }),
        ),
        ToolDefinition::function(
            "get_time",
            "Get the current time for a timezone",
            json!({
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, e.g., 'Europe/Paris'"
                    }
                },
                "required": ["timezone"],
                "additionalProperties": false
            }),
        ),
    ];

    println!("\n📋 Available Tools:");
    for tool in &tools {
        println!("  - {} : {}", tool.function.name, tool.function.description);
    }
    println!();

    // Start conversation
    let mut messages = vec![
        ChatMessage::system("You are a helpful assistant with access to weather and time tools."),
        ChatMessage::user("What's the weather like in Tokyo and what time is it there?"),
    ];

    println!("👤 User: What's the weather like in Tokyo and what time is it there?\n");

    // First completion - model may call tools.
    // Note: gpt-5-mini and o-series models only accept the default temperature (1.0).
    // Omit temperature to stay model-agnostic.
    let options = CompletionOptions {
        ..Default::default()
    };

    let response = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), Some(&options))
        .await?;

    println!(
        "🤖 Assistant thinking... (finish_reason: {:?})",
        response.finish_reason
    );

    // Check if the model wants to call tools
    if !response.tool_calls.is_empty() {
        println!("\n📞 Tool Calls:");

        // Add assistant message with tool calls
        messages.push(ChatMessage::assistant_with_tools(
            response.content.clone(),
            response.tool_calls.clone(),
        ));

        // Process each tool call
        for tool_call in &response.tool_calls {
            println!(
                "  - {}({}) [id: {}]",
                tool_call.function.name, tool_call.function.arguments, tool_call.id
            );

            // Simulate tool execution
            let result = execute_tool(&tool_call.function.name, &tool_call.function.arguments);
            println!("    → Result: {}", result);

            // Add tool result to messages
            messages.push(ChatMessage::tool_result(&tool_call.id, result));
        }

        // Get final response after tool execution
        println!("\n⏳ Getting final response...\n");

        let final_response = provider
            .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), Some(&options))
            .await?;

        println!("🤖 Assistant: {}", final_response.content);
        println!(
            "\nTokens: {} prompt + {} completion = {} total",
            final_response.prompt_tokens,
            final_response.completion_tokens,
            final_response.total_tokens
        );
    } else {
        // Model responded directly without tools
        println!("🤖 Assistant: {}", response.content);
    }

    println!("\n{}", "─".repeat(60));
    println!("✅ Tool calling complete!");

    Ok(())
}

/// Simulate tool execution (in real apps, this would call actual APIs).
fn execute_tool(name: &str, arguments: &str) -> String {
    let args: serde_json::Value = serde_json::from_str(arguments).unwrap_or(json!({}));

    match name {
        "get_weather" => {
            let location = args["location"].as_str().unwrap_or("Unknown");
            let unit = args["unit"].as_str().unwrap_or("celsius");
            // Simulated weather response
            format!(
                r#"{{"location": "{}", "temperature": 22, "unit": "{}", "conditions": "Partly cloudy"}}"#,
                location, unit
            )
        }
        "get_time" => {
            let timezone = args["timezone"].as_str().unwrap_or("UTC");
            // Simulated time response (in real app, use chrono-tz)
            format!(
                r#"{{"timezone": "{}", "time": "14:30", "date": "2025-01-26"}}"#,
                timezone
            )
        }
        _ => format!(r#"{{"error": "Unknown tool: {}"}}"#, name),
    }
}
