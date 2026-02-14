//! Chatbot example
//!
//! Demonstrates an interactive chatbot with conversation history.
//!
//! Run with: cargo run --example chatbot
//! Requires: OPENAI_API_KEY environment variable
//!
//! This example shows:
//! - Multi-turn conversation with memory
//! - User input handling
//! - System prompt for personality
//! - Token usage tracking across turns

use std::io::{self, BufRead, Write};
use edgequake_llm::{ChatMessage, CompletionOptions, LLMProvider, OpenAIProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Initialize provider
    let provider = OpenAIProvider::new(&api_key);

    println!("🤖 EdgeQuake LLM - Chatbot Example");
    println!("═══════════════════════════════════════════════════════════");
    println!("Provider: {} | Model: {}", provider.name(), provider.model());
    println!("Type 'quit' or 'exit' to end the conversation.");
    println!("Type 'clear' to reset conversation history.");
    println!("═══════════════════════════════════════════════════════════\n");

    // Initialize conversation with system prompt
    let mut messages: Vec<ChatMessage> = vec![
        ChatMessage::system(
            "You are a helpful, friendly assistant. Keep responses concise but informative. \
             Use simple language and be conversational."
        ),
    ];

    let options = CompletionOptions {
        max_tokens: Some(500),
        temperature: Some(0.7),
        ..Default::default()
    };

    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;
    let mut turn_count = 0;

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Prompt for user input
        print!("\n👤 You: ");
        stdout.flush()?;

        // Read user input
        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        // Check for exit commands
        match input.to_lowercase().as_str() {
            "quit" | "exit" => {
                println!("\n👋 Goodbye! Session stats:");
                println!("   Turns: {}", turn_count);
                println!("   Total tokens: {} prompt + {} completion",
                    total_prompt_tokens, total_completion_tokens);
                break;
            }
            "clear" => {
                messages.truncate(1); // Keep system prompt
                turn_count = 0;
                println!("🔄 Conversation cleared. Starting fresh!");
                continue;
            }
            "" => continue, // Skip empty input
            _ => {}
        }

        // Add user message
        messages.push(ChatMessage::user(input));

        // Get response
        print!("\n🤖 Assistant: ");
        stdout.flush()?;

        match provider.chat(&messages, Some(&options)).await {
            Ok(response) => {
                println!("{}", response.content);
                
                // Track tokens
                total_prompt_tokens += response.prompt_tokens;
                total_completion_tokens += response.completion_tokens;
                turn_count += 1;

                // Show token usage
                println!("\n   [tokens: {} prompt + {} completion | turn #{}]",
                    response.prompt_tokens, response.completion_tokens, turn_count);

                // Add assistant response to history
                messages.push(ChatMessage::assistant(&response.content));
            }
            Err(e) => {
                println!("Error: {}", e);
                // Remove the user message on error
                messages.pop();
            }
        }
    }

    Ok(())
}
