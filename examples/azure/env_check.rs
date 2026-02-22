use edgequake_llm::AzureOpenAIProvider;

fn main() {
    match AzureOpenAIProvider::from_env_auto() {
        Ok(provider) => {
            println!("Azure provider loaded:\n{:#?}", provider);
        }
        Err(e) => {
            eprintln!("Failed to create AzureOpenAIProvider: {:?}", e);
            std::process::exit(1);
        }
    }
}
