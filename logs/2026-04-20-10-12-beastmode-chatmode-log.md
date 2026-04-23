# Task logs

Actions: pinned edgequake-llm and edgecrab to Rust 1.95.0, added repo toolchain files, and verified Bedrock builds with the latest AWS SDK crates.

Decisions: kept aws-sdk-bedrockruntime at 1.129.0 and aws-config at 1.8.15 because they are already the latest published versions confirmed from upstream docs.

Next steps: optionally commit and tag these toolchain updates if you want them published immediately.

Lessons/insights: disabling the Bedrock SDK service crate default features removed the legacy TLS path and left cargo-audit with warnings only, not blocking vulnerabilities.
