# Iteration 20 - Observe
factory.rs: 1382 lines, 12 existing tests. Missing: create_with_model paths, from_config (Azure error, Mock), create_embedding_provider, create_llm_provider, VSCode parsing, Debug/Clone/Eq traits.

# Iteration 20 - Orient
Factory is the central entry point. Testing create_with_model and from_config ensures all provider routing paths are exercised.

# Iteration 20 - Decide
Add 11 tests: create_with_model (Mock none/some), create_embedding_provider (mock/unknown), create_llm_provider (mock/unknown), from_config (Azure/Mock), provider type traits, VSCode parsing.
