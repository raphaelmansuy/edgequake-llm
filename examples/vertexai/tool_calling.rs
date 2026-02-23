//! VertexAI tool calling (function calling) examples.
//!
//! Run: cargo run --example vertexai_tool_calling
//! Requires:
//!   - GOOGLE_CLOUD_PROJECT
//!   - Authenticated via `gcloud auth login` (or GOOGLE_ACCESS_TOKEN)

use edgequake_llm::providers::gemini::GeminiProvider;
use edgequake_llm::traits::{
    ChatMessage, FunctionDefinition, LLMProvider, ToolChoice, ToolDefinition,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GeminiProvider::from_env_vertex_ai()?;
    println!(
        "VertexAI Tool Calling Examples — model: {}\n",
        LLMProvider::model(&provider)
    );

    // ────────────────────────────────────────────────────────────── 1 ──
    // Single tool — weather lookup
    // ────────────────────────────────────────────────────────────── 1 ──
    println!("=== 1. Single tool call ===");

    let weather_tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a given location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'Paris, France'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }),
            strict: None,
        },
    };

    let messages = vec![ChatMessage::user("What's the weather like in Tokyo today?")];
    let resp = provider
        .chat_with_tools(
            &messages,
            std::slice::from_ref(&weather_tool),
            Some(ToolChoice::auto()),
            None,
        )
        .await?;

    if resp.has_tool_calls() {
        for tc in &resp.tool_calls {
            println!("  Function: {}", tc.name());
            println!("  Args:     {}", tc.arguments());
        }
    } else {
        println!("  Text: {}", resp.content);
    }
    println!();

    // ────────────────────────────────────────────────────────────── 2 ──
    // Multiple tools — the model picks the right one
    // ────────────────────────────────────────────────────────────── 2 ──
    println!("=== 2. Multiple tools — model selects ===");

    let calc_tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "calculate".to_string(),
            description: "Evaluate a mathematical expression".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }),
            strict: None,
        },
    };

    let translate_tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "translate".to_string(),
            description: "Translate text from one language to another".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "text":        { "type": "string", "description": "Text to translate" },
                    "source_lang": { "type": "string", "description": "Source language code" },
                    "target_lang": { "type": "string", "description": "Target language code" }
                },
                "required": ["text", "target_lang"]
            }),
            strict: None,
        },
    };

    let tools = vec![
        weather_tool.clone(),
        calc_tool.clone(),
        translate_tool.clone(),
    ];

    // Should choose calculate
    let messages = vec![ChatMessage::user("What is 17 * 23 + 42?")];
    let resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await?;
    print!("  Math question → ");
    if resp.has_tool_calls() {
        for tc in &resp.tool_calls {
            println!("{}({})", tc.name(), tc.arguments());
        }
    } else {
        println!("text: {}", resp.content);
    }

    // Should choose translate
    let messages = vec![ChatMessage::user(
        "How do you say 'good morning' in Japanese?",
    )];
    let resp = provider
        .chat_with_tools(&messages, &tools, Some(ToolChoice::auto()), None)
        .await?;
    print!("  Translate question → ");
    if resp.has_tool_calls() {
        for tc in &resp.tool_calls {
            println!("{}({})", tc.name(), tc.arguments());
        }
    } else {
        println!("text: {}", resp.content);
    }
    println!();

    // ────────────────────────────────────────────────────────────── 3 ──
    // Forced tool choice
    // ────────────────────────────────────────────────────────────── 3 ──
    println!("=== 3. Forced tool choice ===");
    let messages = vec![ChatMessage::user(
        "Tell me something interesting about Mount Fuji.",
    )];
    let resp = provider
        .chat_with_tools(
            &messages,
            std::slice::from_ref(&weather_tool),
            Some(ToolChoice::function("get_weather")),
            None,
        )
        .await?;
    print!("  Forced weather → ");
    if resp.has_tool_calls() {
        for tc in &resp.tool_calls {
            println!("{}({})", tc.name(), tc.arguments());
        }
    } else {
        println!("text: {}", resp.content);
    }
    println!();

    // ────────────────────────────────────────────────────────────── 4 ──
    // Multi-step tool use with result feeding
    // ────────────────────────────────────────────────────────────── 4 ──
    println!("=== 4. Multi-step tool use ===");

    // Step 1: Ask a question that needs a tool
    let messages = vec![ChatMessage::user(
        "I need to know the weather in London to decide what to wear.",
    )];
    let step1 = provider
        .chat_with_tools(
            &messages,
            std::slice::from_ref(&weather_tool),
            Some(ToolChoice::auto()),
            None,
        )
        .await?;

    if step1.has_tool_calls() {
        let tc = &step1.tool_calls[0];
        println!("  Step 1 — Tool call: {}({})", tc.name(), tc.arguments());

        // Step 2: Feed the tool result back
        let tool_result = serde_json::json!({
            "temperature": 14,
            "unit": "celsius",
            "condition": "partly cloudy",
            "humidity": 72,
            "wind_speed_kph": 15
        });

        let mut messages = vec![
            ChatMessage::user("I need to know the weather in London to decide what to wear."),
            ChatMessage::assistant_with_tools("", step1.tool_calls.clone()),
        ];
        messages.push(ChatMessage::tool_result(&tc.id, tool_result.to_string()));

        let step2 = provider
            .chat_with_tools(&messages, &[weather_tool], Some(ToolChoice::auto()), None)
            .await?;
        println!("  Step 2 — Response: {}", step2.content);
    } else {
        println!("  Step 1 — No tool call: {}", step1.content);
    }
    println!();

    // ────────────────────────────────────────────────────────────── 5 ──
    // Tool with complex schema
    // ────────────────────────────────────────────────────────────── 5 ──
    println!("=== 5. Complex tool schema ===");

    let search_tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "search_products".to_string(),
            description: "Search for products in an e-commerce catalog".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books", "home", "sports"],
                        "description": "Product category filter"
                    },
                    "price_range": {
                        "type": "object",
                        "properties": {
                            "min": { "type": "number", "description": "Minimum price" },
                            "max": { "type": "number", "description": "Maximum price" }
                        },
                        "description": "Price range filter"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "price_asc", "price_desc", "rating"],
                        "description": "Sort order"
                    }
                },
                "required": ["query"]
            }),
            strict: None,
        },
    };

    let messages = vec![ChatMessage::user(
        "Find me running shoes under $100, sorted by rating.",
    )];
    let resp = provider
        .chat_with_tools(&messages, &[search_tool], Some(ToolChoice::auto()), None)
        .await?;

    if resp.has_tool_calls() {
        for tc in &resp.tool_calls {
            println!("  Function: {}", tc.name());
            let parsed: serde_json::Value =
                serde_json::from_str(tc.arguments()).unwrap_or_default();
            println!("  Args (pretty):");
            println!(
                "{}",
                serde_json::to_string_pretty(&parsed).unwrap_or_default()
            );
        }
    } else {
        println!("  Text: {}", resp.content);
    }
    println!();

    println!("=== Done! ===");
    Ok(())
}
