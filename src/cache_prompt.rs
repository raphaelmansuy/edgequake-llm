//! Prompt caching utilities for Anthropic Claude models.
//!
//! # OODA-17: Anthropic Prompt Caching
//!
//! This module provides utilities for leveraging Anthropic's prompt caching feature
//! to reduce costs by 85-90% on repeated context.
//!
//! # Overview
//!
//! Prompt caching allows marking parts of the conversation as cacheable:
//! - System prompts (rarely change)
//! - Large file contexts (repeated across calls)
//! - Recent conversation history (conversation context)
//!
//! Cached tokens are served at 90% discount for subsequent requests.
//!
//! # Usage
//!
//! ```rust
//! use edgequake_llm::cache_prompt::{CachePromptConfig, apply_cache_control};
//! use edgequake_llm::traits::ChatMessage;
//!
//! let config = CachePromptConfig::default();
//! let mut messages = vec![
//!     ChatMessage::system("You are a helpful assistant"),
//!     ChatMessage::user("Large file content here..."),
//! ];
//!
//! apply_cache_control(&mut messages, &config);
//! // Now messages have cache_control set where appropriate
//! ```
//!
//! # See Also
//!
//! - [Anthropic Prompt Caching](https://docs.anthropic.com/claude/docs/prompt-caching)
//! - Aider reference: `base_coder.py`, `sendchat.py`

use crate::traits::{CacheControl, ChatMessage, ChatRole};
use serde::{Deserialize, Serialize};

/// Configuration for automatic prompt cache control marking.
///
/// # Fields
///
/// - `enabled`: Whether to apply cache control (default: true)
/// - `min_content_length`: Minimum message length to auto-cache (default: 1000)
/// - `cache_system_prompt`: Whether to cache system prompts (default: true)
/// - `cache_last_n_messages`: Number of recent user messages to cache (default: 3)
///
/// # Example
///
/// ```rust
/// use edgequake_llm::cache_prompt::CachePromptConfig;
///
/// // Use defaults
/// let config = CachePromptConfig::default();
///
/// // Custom configuration
/// let config = CachePromptConfig {
///     enabled: true,
///     min_content_length: 500,
///     cache_system_prompt: true,
///     cache_last_n_messages: 5,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePromptConfig {
    /// Whether cache control marking is enabled.
    pub enabled: bool,

    /// Minimum content length (in characters) to auto-cache user messages.
    ///
    /// Messages shorter than this threshold are not automatically cached
    /// unless they are system prompts or in the last N messages.
    pub min_content_length: usize,

    /// Whether to cache the system prompt.
    ///
    /// System prompts rarely change and are excellent cache candidates.
    pub cache_system_prompt: bool,

    /// Number of recent user messages to cache.
    ///
    /// Caching recent messages helps with conversation context retention.
    pub cache_last_n_messages: usize,
}

impl Default for CachePromptConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_content_length: 1000,
            cache_system_prompt: true,
            cache_last_n_messages: 3,
        }
    }
}

impl CachePromptConfig {
    /// Create a config with caching disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a config that only caches system prompts.
    pub fn system_only() -> Self {
        Self {
            enabled: true,
            min_content_length: usize::MAX,
            cache_system_prompt: true,
            cache_last_n_messages: 0,
        }
    }

    /// Create an aggressive caching config.
    ///
    /// Caches more content for maximum cost reduction.
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            min_content_length: 100,
            cache_system_prompt: true,
            cache_last_n_messages: 10,
        }
    }
}

/// Statistics about cache usage from an API response.
///
/// Anthropic returns cache statistics in the usage field of responses:
/// - `cache_read_input_tokens`: Tokens served from cache (90% cheaper)
/// - `cache_creation_input_tokens`: Tokens used to create the cache
///
/// # Cost Model
///
/// - Normal input tokens: $0.003 per 1K tokens
/// - Cached input tokens: $0.0003 per 1K tokens (90% discount)
/// - Cache creation has a small overhead but pays off after 2-3 uses
///
/// # Example
///
/// ```rust
/// use edgequake_llm::cache_prompt::CacheStats;
///
/// let stats = CacheStats {
///     input_tokens: 10000,
///     output_tokens: 1000,
///     cache_read_tokens: 8000,
///     cache_creation_tokens: 0,
/// };
///
/// println!("Cache hit rate: {:.0}%", stats.cache_hit_rate() * 100.0);
/// println!("Estimated savings: ${:.4}", stats.savings());
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total input tokens in the request.
    pub input_tokens: u64,

    /// Total output tokens in the response.
    pub output_tokens: u64,

    /// Input tokens served from cache.
    pub cache_read_tokens: u64,

    /// Tokens used to create new cache entries.
    pub cache_creation_tokens: u64,
}

impl CacheStats {
    /// Create new cache stats.
    pub fn new(
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_creation_tokens: u64,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_creation_tokens,
        }
    }

    /// Calculate the cache hit rate as a fraction (0.0 to 1.0).
    ///
    /// Returns 0.0 if there are no input tokens.
    pub fn cache_hit_rate(&self) -> f64 {
        if self.input_tokens == 0 {
            0.0
        } else {
            self.cache_read_tokens as f64 / self.input_tokens as f64
        }
    }

    /// Estimate cost savings in dollars.
    ///
    /// Based on Anthropic Claude pricing (as of 2024):
    /// - Normal input: $0.003 per 1K tokens
    /// - Cached input: $0.0003 per 1K tokens
    ///
    /// Returns the difference between what would have been paid
    /// without caching vs with caching.
    pub fn savings(&self) -> f64 {
        const NORMAL_COST_PER_1K: f64 = 0.003;
        const CACHE_COST_PER_1K: f64 = 0.0003;

        // Cost without caching
        let normal_cost = self.input_tokens as f64 * NORMAL_COST_PER_1K / 1000.0;

        // Cost with caching
        let uncached_tokens = self.input_tokens.saturating_sub(self.cache_read_tokens);
        let cache_cost = self.cache_read_tokens as f64 * CACHE_COST_PER_1K / 1000.0
            + uncached_tokens as f64 * NORMAL_COST_PER_1K / 1000.0;

        normal_cost - cache_cost
    }

    /// Calculate the cost per call with current cache stats.
    pub fn cost_per_call(&self) -> f64 {
        const NORMAL_COST_PER_1K: f64 = 0.003;
        const CACHE_COST_PER_1K: f64 = 0.0003;
        const OUTPUT_COST_PER_1K: f64 = 0.015; // Claude output tokens

        let uncached_tokens = self.input_tokens.saturating_sub(self.cache_read_tokens);

        self.cache_read_tokens as f64 * CACHE_COST_PER_1K / 1000.0
            + uncached_tokens as f64 * NORMAL_COST_PER_1K / 1000.0
            + self.output_tokens as f64 * OUTPUT_COST_PER_1K / 1000.0
    }

    /// Check if caching was effective (hit rate > 50%).
    pub fn is_effective(&self) -> bool {
        self.cache_hit_rate() > 0.5
    }

    /// Merge stats from another request.
    pub fn merge(&mut self, other: &CacheStats) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_creation_tokens += other.cache_creation_tokens;
    }
}

/// Apply cache control to messages based on configuration.
///
/// This function marks messages with `cache_control` hints that providers
/// like Anthropic Claude can use to cache prompt prefixes.
///
/// # Cache Marking Strategy
///
/// 1. **System prompts**: Always cached (if `cache_system_prompt` is true)
/// 2. **Large user messages**: Cached if length > `min_content_length`
/// 3. **Recent user messages**: Last N user messages are cached
///
/// # Arguments
///
/// * `messages` - Mutable slice of messages to apply cache control to
/// * `config` - Configuration controlling which messages to cache
///
/// # Example
///
/// ```rust
/// use edgequake_llm::cache_prompt::{CachePromptConfig, apply_cache_control};
/// use edgequake_llm::traits::ChatMessage;
///
/// let config = CachePromptConfig::default();
/// let mut messages = vec![
///     ChatMessage::system("You are a helpful assistant"),
///     ChatMessage::user("Please analyze this file: ..."),
/// ];
///
/// apply_cache_control(&mut messages, &config);
///
/// assert!(messages[0].cache_control.is_some()); // System prompt cached
/// ```
pub fn apply_cache_control(messages: &mut [ChatMessage], config: &CachePromptConfig) {
    if !config.enabled {
        return;
    }

    // Track user message indices for last-N caching
    let user_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m.role, ChatRole::User))
        .map(|(i, _)| i)
        .collect();

    // Determine which indices should be cached as "last N"
    let last_n_start = user_indices
        .len()
        .saturating_sub(config.cache_last_n_messages);
    let last_n_indices: std::collections::HashSet<usize> =
        user_indices.into_iter().skip(last_n_start).collect();

    for (i, msg) in messages.iter_mut().enumerate() {
        let should_cache = match msg.role {
            ChatRole::System => config.cache_system_prompt,
            ChatRole::User => {
                // Cache if large content OR in last N user messages
                msg.content.len() >= config.min_content_length || last_n_indices.contains(&i)
            }
            _ => false, // Don't cache assistant/tool messages
        };

        if should_cache && msg.cache_control.is_none() {
            msg.cache_control = Some(CacheControl::ephemeral());
        }
    }
}

/// Parse cache statistics from an Anthropic API response.
///
/// Anthropic includes cache stats in the `usage` field:
/// ```json
/// {
///   "usage": {
///     "input_tokens": 10000,
///     "output_tokens": 500,
///     "cache_read_input_tokens": 8000,
///     "cache_creation_input_tokens": 0
///   }
/// }
/// ```
pub fn parse_cache_stats(usage: &serde_json::Value) -> CacheStats {
    CacheStats {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
        cache_read_tokens: usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
        cache_creation_tokens: usage["cache_creation_input_tokens"].as_u64().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CachePromptConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_content_length, 1000);
        assert!(config.cache_system_prompt);
        assert_eq!(config.cache_last_n_messages, 3);
    }

    #[test]
    fn test_disabled_config() {
        let config = CachePromptConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_system_only_config() {
        let config = CachePromptConfig::system_only();
        assert!(config.enabled);
        assert!(config.cache_system_prompt);
        assert_eq!(config.cache_last_n_messages, 0);
        assert_eq!(config.min_content_length, usize::MAX);
    }

    #[test]
    fn test_aggressive_config() {
        let config = CachePromptConfig::aggressive();
        assert!(config.enabled);
        assert_eq!(config.min_content_length, 100);
        assert_eq!(config.cache_last_n_messages, 10);
    }

    #[test]
    fn test_cache_control_disabled() {
        let config = CachePromptConfig::disabled();
        let mut messages = vec![
            ChatMessage::system("System prompt"),
            ChatMessage::user("User message"),
        ];

        apply_cache_control(&mut messages, &config);

        assert!(messages[0].cache_control.is_none());
        assert!(messages[1].cache_control.is_none());
    }

    #[test]
    fn test_cache_system_prompt() {
        let config = CachePromptConfig::default();
        let mut messages = vec![
            ChatMessage::system("You are a helpful assistant"),
            ChatMessage::user("Hello"),
        ];

        apply_cache_control(&mut messages, &config);

        assert!(messages[0].cache_control.is_some());
        assert_eq!(
            messages[0].cache_control.as_ref().unwrap().cache_type,
            "ephemeral"
        );
    }

    #[test]
    fn test_cache_large_messages() {
        let config = CachePromptConfig {
            min_content_length: 100,
            cache_last_n_messages: 0,
            ..Default::default()
        };

        let large_content = "x".repeat(150);
        let small_content = "y".repeat(50);

        let mut messages = vec![
            ChatMessage::system("System"),
            ChatMessage::user(&large_content),
            ChatMessage::user(&small_content),
        ];

        apply_cache_control(&mut messages, &config);

        // System should be cached
        assert!(messages[0].cache_control.is_some());
        // Large message should be cached
        assert!(messages[1].cache_control.is_some());
        // Small message should NOT be cached (last_n is 0)
        assert!(messages[2].cache_control.is_none());
    }

    #[test]
    fn test_cache_last_n_messages() {
        let config = CachePromptConfig {
            min_content_length: usize::MAX, // Disable size-based caching
            cache_last_n_messages: 2,
            cache_system_prompt: false,
            ..Default::default()
        };

        let mut messages = vec![
            ChatMessage::system("System"),
            ChatMessage::user("First"),
            ChatMessage::assistant("Response"),
            ChatMessage::user("Second"),
            ChatMessage::assistant("Response"),
            ChatMessage::user("Third"),
            ChatMessage::user("Fourth"),
        ];

        apply_cache_control(&mut messages, &config);

        // System not cached (disabled)
        assert!(messages[0].cache_control.is_none());
        // First two user messages not cached
        assert!(messages[1].cache_control.is_none());
        assert!(messages[3].cache_control.is_none());
        // Last two user messages cached
        assert!(messages[5].cache_control.is_some()); // Third
        assert!(messages[6].cache_control.is_some()); // Fourth
    }

    #[test]
    fn test_preserves_existing_cache_control() {
        let config = CachePromptConfig::default();
        let mut messages = vec![ChatMessage::system("System")];

        // Pre-set cache control
        messages[0].cache_control = Some(CacheControl::ephemeral());

        apply_cache_control(&mut messages, &config);

        // Should still have cache control
        assert!(messages[0].cache_control.is_some());
    }

    #[test]
    fn test_cache_hit_rate_zero_tokens() {
        let stats = CacheStats::default();
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_hit_rate_full_cache() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 500,
            cache_read_tokens: 10000,
            cache_creation_tokens: 0,
        };
        assert_eq!(stats.cache_hit_rate(), 1.0);
    }

    #[test]
    fn test_cache_hit_rate_partial() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 500,
            cache_read_tokens: 8000,
            cache_creation_tokens: 0,
        };
        assert_eq!(stats.cache_hit_rate(), 0.8);
    }

    #[test]
    fn test_cache_savings() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 500,
            cache_read_tokens: 8000,
            cache_creation_tokens: 0,
        };

        let savings = stats.savings();

        // 8000 tokens saved at 90% discount = 8000 * 0.0027 / 1000 = $0.0216
        assert!(savings > 0.02);
        assert!(savings < 0.03);
    }

    #[test]
    fn test_cache_savings_no_cache() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 500,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        };

        assert_eq!(stats.savings(), 0.0);
    }

    #[test]
    fn test_is_effective() {
        let effective = CacheStats {
            input_tokens: 10000,
            cache_read_tokens: 6000,
            ..Default::default()
        };
        assert!(effective.is_effective());

        let ineffective = CacheStats {
            input_tokens: 10000,
            cache_read_tokens: 4000,
            ..Default::default()
        };
        assert!(!ineffective.is_effective());
    }

    #[test]
    fn test_merge_stats() {
        let mut stats1 = CacheStats {
            input_tokens: 1000,
            output_tokens: 100,
            cache_read_tokens: 500,
            cache_creation_tokens: 200,
        };

        let stats2 = CacheStats {
            input_tokens: 2000,
            output_tokens: 200,
            cache_read_tokens: 1000,
            cache_creation_tokens: 100,
        };

        stats1.merge(&stats2);

        assert_eq!(stats1.input_tokens, 3000);
        assert_eq!(stats1.output_tokens, 300);
        assert_eq!(stats1.cache_read_tokens, 1500);
        assert_eq!(stats1.cache_creation_tokens, 300);
    }

    #[test]
    fn test_parse_cache_stats() {
        let usage = serde_json::json!({
            "input_tokens": 10000,
            "output_tokens": 500,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 100
        });

        let stats = parse_cache_stats(&usage);

        assert_eq!(stats.input_tokens, 10000);
        assert_eq!(stats.output_tokens, 500);
        assert_eq!(stats.cache_read_tokens, 8000);
        assert_eq!(stats.cache_creation_tokens, 100);
    }

    #[test]
    fn test_parse_cache_stats_missing_fields() {
        let usage = serde_json::json!({
            "input_tokens": 5000,
            "output_tokens": 200
        });

        let stats = parse_cache_stats(&usage);

        assert_eq!(stats.input_tokens, 5000);
        assert_eq!(stats.output_tokens, 200);
        assert_eq!(stats.cache_read_tokens, 0);
        assert_eq!(stats.cache_creation_tokens, 0);
    }

    #[test]
    fn test_cost_per_call() {
        let stats = CacheStats {
            input_tokens: 10000,
            output_tokens: 1000,
            cache_read_tokens: 8000,
            cache_creation_tokens: 0,
        };

        let cost = stats.cost_per_call();

        // 8000 cached at $0.0003/1K = $0.0024
        // 2000 normal at $0.003/1K = $0.006
        // 1000 output at $0.015/1K = $0.015
        // Total: $0.0234
        assert!(cost > 0.02);
        assert!(cost < 0.03);
    }

    #[test]
    fn test_cache_stats_serialization() {
        let stats = CacheStats {
            input_tokens: 1000,
            output_tokens: 100,
            cache_read_tokens: 800,
            cache_creation_tokens: 50,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: CacheStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.input_tokens, deserialized.input_tokens);
        assert_eq!(stats.output_tokens, deserialized.output_tokens);
        assert_eq!(stats.cache_read_tokens, deserialized.cache_read_tokens);
        assert_eq!(
            stats.cache_creation_tokens,
            deserialized.cache_creation_tokens
        );
    }
}
