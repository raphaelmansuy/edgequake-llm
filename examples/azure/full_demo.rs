use chrono::Utc;
use edgequake_llm::traits::{ChatMessage, ImageData, StreamChunk};
use edgequake_llm::{
    AzureOpenAIProvider, CompletionOptions, EmbeddingProvider, LLMProvider, ToolChoice,
    ToolDefinition,
};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    // Create provider from environment (.env is loaded by the provider constructors)
    let provider = match AzureOpenAIProvider::from_env_auto() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create AzureOpenAIProvider: {:?}", e);
            std::process::exit(1);
        }
    };

    println!(
        "Provider: {} | Model: {}",
        LLMProvider::name(&provider),
        LLMProvider::model(&provider)
    );

    // 1) Simple poem generation (completion)
    println!("\n=== Poem generation ===");
    let poem_prompt = "Write a short, evocative four-line poem about winter mornings.";
    match provider
        .complete_with_options(
            poem_prompt,
            &CompletionOptions {
                temperature: Some(0.7),
                max_tokens: Some(120),
                ..Default::default()
            },
        )
        .await
    {
        Ok(resp) => println!("Poem:\n{}\n---\n", resp.content),
        Err(e) => eprintln!("Poem generation failed: {:?}", e),
    }

    // 2) Chat-based image description (textual simulation)
    println!("\n=== Image description (text) ===");
    let messages = vec![
        ChatMessage::system("You are a helpful assistant that describes images."),
        ChatMessage::user(
            "Describe the following photo: a golden retriever playing on a beach at sunset.",
        ),
    ];
    match provider.chat(&messages, None).await {
        Ok(resp) => println!("Description:\n{}\n---\n", resp.content),
        Err(e) => eprintln!("Image description failed: {:?}", e),
    }

    // 3) Embeddings for short texts
    println!("\n=== Embeddings ===");
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "EdgeQuake is a multi-provider LLM abstraction library.".to_string(),
    ];
    match provider.embed(&texts).await {
        Ok(embs) => println!(
            "Got {} embeddings (dim {}).",
            embs.len(),
            embs.first().map(|v| v.len()).unwrap_or(0)
        ),
        Err(e) => eprintln!("Embeddings failed: {:?}", e),
    }

    // 4) Streaming generation
    if provider.supports_streaming() {
        println!("\n=== Streaming generation ===");
        let long_prompt = "Tell a pleasant, unfolding short story in several paragraphs about a lighthouse keeper who rediscovers an old map.";
        match provider.stream(long_prompt).await {
            Ok(mut stream) => {
                while let Some(chunk_res) = stream.next().await {
                    match chunk_res {
                        Ok(chunk) => {
                            if !chunk.is_empty() {
                                print!("{}", chunk);
                            }
                        }
                        Err(e) => {
                            eprintln!("Stream error: {:?}", e);
                            break;
                        }
                    }
                }
                println!("\n---\n");
            }
            Err(e) => eprintln!("Streaming failed: {:?}", e),
        }
    } else {
        println!("Provider does not support streaming.");
    }

    // 5) Tool-calling streaming demo (attempt to force a function call)
    if provider.supports_tool_streaming() {
        println!("\n=== Tool-calling streaming (toy) ===");

        // Define a toy tool that returns the current time (model may call it)
        let tool_params = serde_json::json!({
            "type": "object",
            "properties": { "none": { "type": "string" } }
        });

        let tool = ToolDefinition::function(
            "get_current_time",
            "Returns the current time as ISO string",
            tool_params,
        );

        let chat_msgs = vec![
            ChatMessage::system("You are an assistant that may call a tool to get the current time."),
            ChatMessage::user("Please call the function to get the current time and reply with a friendly message."),
        ];

        match provider
            .chat_with_tools_stream(
                &chat_msgs,
                &[tool],
                Some(ToolChoice::function("get_current_time")),
                None,
            )
            .await
        {
            Ok(mut s) => {
                while let Some(ev) = s.next().await {
                    match ev {
                        Ok(StreamChunk::Content(text)) => print!("{}", text),
                        Ok(StreamChunk::ToolCallDelta {
                            index,
                            id,
                            function_name,
                            function_arguments,
                        }) => {
                            println!(
                                "\n[ToolCallDelta] idx={} id={:?} fn={:?} args={:?}",
                                index, id, function_name, function_arguments
                            );
                            // Respond with a fake tool result if the model asked for it
                            if let Some(id) = id {
                                let tool_result = edgequake_llm::traits::ToolResult::new(
                                    id.clone(),
                                    format!("{{\"time\":\"{}\"}}", Utc::now().to_rfc3339()),
                                );
                                // In a full flow we would send this back via a follow-up chat call; here we just print it
                                println!("[Simulated tool result] {}", tool_result.content);
                            }
                        }
                        Ok(StreamChunk::Finished { reason, .. }) => {
                            println!("\n[Stream finished] reason={}", reason);
                            break;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("Stream error: {:?}", e);
                            break;
                        }
                    }
                }
                println!("\n---\n");
            }
            Err(e) => eprintln!("Tool streaming failed: {:?}", e),
        }
    } else {
        println!("Provider does not support tool streaming.");
    }

    // 6a) Image multimodal: public URL
    // Use Azure's own GitHub sample data — GitHub raw is always reachable from
    // Azure's network, never rate-limits, and landmark images pass content filters.
    println!("\n=== Image multimodal — URL ===");
    {
        let img = ImageData::from_url(
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/landmark.jpg",
        )
        .with_detail("low");
        let img_msg = ChatMessage::user_with_images(
            "Describe what you see in this image in one sentence.",
            vec![img],
        );
        match provider
            .chat(
                &[
                    ChatMessage::system("You are a concise vision assistant."),
                    img_msg,
                ],
                None,
            )
            .await
        {
            Ok(resp) => println!("URL image response: {}\n---\n", resp.content),
            Err(e) => eprintln!("URL image call failed: {:?}", e),
        }
    }

    // 6b) Image multimodal: base64-encoded data
    // Download a different Azure sample image (GitHub raw never 429s our reqwest),
    // base64-encode the real JPEG bytes, and send them directly.
    println!("\n=== Image multimodal — base64 ===");
    {
        use base64::{engine::general_purpose::STANDARD, Engine as _};

        let image_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/landmark.jpg";
        match reqwest::get(image_url).await {
            Ok(response) => match response.bytes().await {
                Ok(bytes) => {
                    let b64 = STANDARD.encode(&bytes);
                    let img = ImageData::new(b64, "image/jpeg").with_detail("low");
                    let img_msg = ChatMessage::user_with_images(
                        "Describe what you see in this image in one sentence.",
                        vec![img],
                    );
                    match provider
                        .chat(
                            &[
                                ChatMessage::system("You are a concise vision assistant."),
                                img_msg,
                            ],
                            None,
                        )
                        .await
                    {
                        Ok(resp) => println!("Base64 image response: {}\n---\n", resp.content),
                        Err(e) => eprintln!("Base64 image chat failed: {:?}", e),
                    }
                }
                Err(e) => eprintln!("Failed to read image bytes: {:?}", e),
            },
            Err(e) => eprintln!("Failed to fetch image for base64 demo: {:?}", e),
        }
    }

    // 7) Content-filter demonstration
    //
    // Azure OpenAI content filters are configured per-deployment in Azure AI Studio.
    // This section shows:
    //   7a) A request that intentionally triggers the default content filter (faces.jpg).
    //   7b) The same request routed through a deployment with content filters disabled,
    //       so the image is described without being blocked.
    //
    // To enable 7b, create a deployment with content filters disabled in Azure AI Studio
    // and set the environment variable:
    //   AZURE_OPENAI_NO_FILTER_DEPLOYMENT_NAME=<your-deployment-name>
    //
    // Azure docs: https://learn.microsoft.com/azure/ai-services/openai/how-to/content-filters
    println!("\n=== Content-filter demonstration ===");

    // Image with human faces — triggers the default responsible-AI content filter.
    let faces_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/faces.jpg";
    let vision_question = "Describe what you see in this image in one sentence.";

    // 7a) Default deployment — expect a content_filter block.
    println!("7a) Default deployment (content filter ON):");
    {
        let img = ImageData::from_url(faces_url).with_detail("low");
        let msg = ChatMessage::user_with_images(vision_question, vec![img]);
        match provider
            .chat(
                &[
                    ChatMessage::system("You are a concise vision assistant."),
                    msg,
                ],
                None,
            )
            .await
        {
            Ok(resp) => println!("    Response (filter did not trigger): {}", resp.content),
            Err(e) => println!("    Blocked as expected: {:?}", e),
        }
    }

    // 7b) Optional deployment with content filter disabled.
    //     Skipped when AZURE_OPENAI_NO_FILTER_DEPLOYMENT_NAME is not set.
    let no_filter_deployment = std::env::var("AZURE_OPENAI_NO_FILTER_DEPLOYMENT_NAME").ok();
    println!(
        "7b) No-filter deployment ({}): {}",
        if no_filter_deployment.is_some() {
            "content filter OFF"
        } else {
            "SKIPPED"
        },
        no_filter_deployment
            .as_deref()
            .unwrap_or("set AZURE_OPENAI_NO_FILTER_DEPLOYMENT_NAME to enable"),
    );
    if let Some(deployment) = no_filter_deployment {
        // Build a fresh provider from the same env vars, then switch deployments.
        match AzureOpenAIProvider::from_env_auto() {
            Ok(unfiltered) => {
                let unfiltered = unfiltered.with_deployment(&deployment);
                let img = ImageData::from_url(faces_url).with_detail("low");
                let msg = ChatMessage::user_with_images(vision_question, vec![img]);
                match unfiltered
                    .chat(
                        &[
                            ChatMessage::system("You are a concise vision assistant."),
                            msg,
                        ],
                        None,
                    )
                    .await
                {
                    Ok(resp) => println!("    Response: {}", resp.content),
                    Err(e) => println!("    Failed: {:?}", e),
                }
            }
            Err(e) => println!("    Could not build no-filter provider: {:?}", e),
        }
    }
    println!("\n---\n");

    println!("Demo complete.");
}
