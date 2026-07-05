//! Model name search — exact, substring, and fuzzy matching.
//!
//! Shared by static registry search and live discovery so ranking logic stays DRY.

use serde::{Deserialize, Serialize};

use super::registry::{all_static_models, static_lookup};
use super::types::{CapabilityFilter, DiscoveredModel};

/// How closely a model matched the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelMatchKind {
    /// Exact model ID (case-insensitive).
    ExactId,
    /// Exact display name (case-insensitive).
    ExactName,
    /// `provider/model` composite exact match.
    ExactQualifiedId,
    /// ID or name starts with the query.
    Prefix,
    /// ID, name, or tag contains the query.
    Contains,
    /// Token / similarity fuzzy match.
    Fuzzy,
}

/// A model ranked by search relevance.
#[derive(Debug, Clone)]
pub struct ModelSearchMatch {
    pub model: DiscoveredModel,
    pub score: f64,
    pub match_kind: ModelMatchKind,
}

/// Query for searching models by name or ID.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelSearchQuery {
    /// Search text — model ID, display name, tag, or `provider/model`.
    pub query: String,
    /// Restrict to one provider (canonical ID).
    pub provider: Option<String>,
    /// Enable fuzzy / similarity matching (default: exact + substring only).
    pub fuzzy: bool,
    /// Minimum score in `[0.0, 1.0]` (defaults: `0.80` exact mode, `0.55` fuzzy).
    pub min_score: Option<f64>,
    /// Maximum number of results (default: `20`).
    pub limit: Option<usize>,
    /// Minimum input context window (tokens).
    pub min_context_length: Option<usize>,
    /// Maximum input context window (tokens).
    pub max_context_length: Option<usize>,
    /// Minimum max output tokens.
    pub min_output_tokens: Option<usize>,
    /// Maximum max output tokens.
    pub max_output_tokens: Option<usize>,
    /// Optional capability constraints (AND logic with name match).
    pub capability_filter: Option<CapabilityFilter>,
}

impl ModelSearchQuery {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Default::default()
        }
    }

    pub fn fuzzy(mut self, enabled: bool) -> Self {
        self.fuzzy = enabled;
        self
    }

    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    pub fn with_min_score(mut self, score: f64) -> Self {
        self.min_score = Some(score);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_capability_filter(mut self, filter: CapabilityFilter) -> Self {
        self.capability_filter = Some(filter);
        self
    }

    pub fn with_min_context_length(mut self, tokens: usize) -> Self {
        self.min_context_length = Some(tokens);
        self
    }

    pub fn with_max_context_length(mut self, tokens: usize) -> Self {
        self.max_context_length = Some(tokens);
        self
    }

    pub fn with_min_output_tokens(mut self, tokens: usize) -> Self {
        self.min_output_tokens = Some(tokens);
        self
    }

    pub fn with_max_output_tokens(mut self, tokens: usize) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Merge query-level length constraints with an optional capability filter.
    pub fn effective_filter(&self) -> CapabilityFilter {
        let mut filter = self.capability_filter.clone().unwrap_or_default();
        merge_min_usize(&mut filter.min_context_length, self.min_context_length);
        merge_max_usize(&mut filter.max_context_length, self.max_context_length);
        merge_min_usize(&mut filter.min_output_tokens, self.min_output_tokens);
        merge_max_usize(&mut filter.max_output_tokens, self.max_output_tokens);
        filter
    }

    fn effective_min_score(&self) -> f64 {
        self.min_score
            .unwrap_or(if self.fuzzy { 0.55 } else { 0.80 })
    }

    fn effective_limit(&self) -> usize {
        self.limit.unwrap_or(20)
    }
}

/// Search a model list by name/ID.
pub fn search_models(
    models: impl IntoIterator<Item = DiscoveredModel>,
    query: &ModelSearchQuery,
) -> Vec<ModelSearchMatch> {
    let (parsed_provider, parsed_model) = parse_qualified_query(&query.query);
    let provider_filter = query.provider.as_deref().or(parsed_provider);
    let search_text = parsed_model.unwrap_or(&query.query);

    if search_text.trim().is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    let qualified = parsed_model.is_some();
    for model in models {
        if let Some(provider) = provider_filter {
            if !model.provider.eq_ignore_ascii_case(provider) {
                continue;
            }
        }
        let filter = query.effective_filter();
        if !filter.matches(&model) {
            continue;
        }
        if let Some((score, kind)) = score_model(search_text, &model, query.fuzzy, qualified) {
            if score >= query.effective_min_score() {
                matches.push(ModelSearchMatch {
                    model,
                    score,
                    match_kind: kind,
                });
            }
        }
    }

    matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.model.provider.cmp(&b.model.provider))
            .then_with(|| a.model.id.cmp(&b.model.id))
    });
    matches.truncate(query.effective_limit());
    matches
}

/// Search the offline static registry by name/ID (no network).
pub fn search_static_models(query: &ModelSearchQuery) -> Vec<ModelSearchMatch> {
    search_models(all_static_models(), query)
}

/// Resolve an exact model from the static registry by provider + name/ID.
///
/// Tries exact ID match first, then exact display name (case-insensitive).
pub fn static_lookup_by_name(provider: &str, name_or_id: &str) -> Option<DiscoveredModel> {
    if let Some(model) = static_lookup(provider, name_or_id) {
        return Some(model);
    }

    let models = match provider {
        "openai" => super::registry::openai_models(),
        "anthropic" => super::registry::anthropic_models(),
        "gemini" | "vertexai" => super::registry::gemini_models(),
        "mistral" => super::registry::mistral_models(),
        "xai" => super::registry::xai_models(),
        "cohere" => super::registry::cohere_models(),
        "nvidia" => super::registry::nvidia_models(),
        _ => return None,
    };

    models
        .into_iter()
        .find(|m| m.name.eq_ignore_ascii_case(name_or_id))
}

fn merge_min_usize(target: &mut Option<usize>, value: Option<usize>) {
    if let Some(value) = value {
        *target = Some(target.map(|current| current.max(value)).unwrap_or(value));
    }
}

fn merge_max_usize(target: &mut Option<usize>, value: Option<usize>) {
    if let Some(value) = value {
        *target = Some(target.map(|current| current.min(value)).unwrap_or(value));
    }
}

fn parse_qualified_query(query: &str) -> (Option<&str>, Option<&str>) {
    let trimmed = query.trim();
    if let Some((provider, model)) = trimmed.split_once('/') {
        let provider = provider.trim();
        let model = model.trim();
        if !provider.is_empty() && !model.is_empty() {
            return (Some(provider), Some(model));
        }
    }
    (None, None)
}

fn normalize(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn compact(value: &str) -> String {
    normalize(value)
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

fn score_model(
    query: &str,
    model: &DiscoveredModel,
    fuzzy: bool,
    qualified: bool,
) -> Option<(f64, ModelMatchKind)> {
    let q_norm = normalize(query);
    let q_compact = compact(query);
    if q_norm.is_empty() {
        return None;
    }

    let id_norm = normalize(&model.id);
    let name_norm = normalize(&model.name);
    let id_compact = compact(&model.id);
    let name_compact = compact(&model.name);
    let qualified_id = format!("{} {}", model.provider, model.id);
    let qualified_norm = normalize(&qualified_id);

    if id_norm == q_norm || id_compact == q_compact {
        return Some((1.0, ModelMatchKind::ExactId));
    }
    if name_norm == q_norm || name_compact == q_compact {
        return Some((0.98, ModelMatchKind::ExactName));
    }
    if qualified_norm == q_norm {
        return Some((0.96, ModelMatchKind::ExactQualifiedId));
    }
    if qualified {
        return None;
    }
    if id_norm.starts_with(&q_norm)
        || name_norm.starts_with(&q_norm)
        || id_compact.starts_with(&q_compact)
    {
        return Some((0.92, ModelMatchKind::Prefix));
    }
    if id_norm.contains(&q_norm)
        || name_norm.contains(&q_norm)
        || id_compact.contains(&q_compact)
        || name_compact.contains(&q_compact)
        || model
            .tags
            .iter()
            .any(|tag| normalize(tag).contains(&q_norm))
    {
        return Some((0.85, ModelMatchKind::Contains));
    }

    if !fuzzy {
        return None;
    }

    let token_score = token_overlap_score(&q_norm, &id_norm, &name_norm, &model.tags);
    let similarity = id_norm
        .split_whitespace()
        .chain(name_norm.split_whitespace())
        .map(|part| dice_coefficient(&q_compact, &compact(part)))
        .fold(0.0_f64, f64::max)
        .max(dice_coefficient(&q_compact, &id_compact))
        .max(dice_coefficient(&q_compact, &name_compact))
        .max(token_score);

    if similarity >= 0.55 {
        Some((similarity, ModelMatchKind::Fuzzy))
    } else {
        None
    }
}

fn token_overlap_score(query: &str, id: &str, name: &str, tags: &[String]) -> f64 {
    let tokens: Vec<&str> = query.split_whitespace().collect();
    if tokens.is_empty() {
        return 0.0;
    }
    let haystacks = [id, name];
    let matched = tokens
        .iter()
        .filter(|token| {
            haystacks.iter().any(|h| h.contains(*token))
                || tags.iter().any(|tag| normalize(tag).contains(*token))
        })
        .count();
    (matched as f64 / tokens.len() as f64) * 0.72
}

fn dice_coefficient(a: &str, b: &str) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    let a_bigrams = bigrams(a);
    let b_bigrams = bigrams(b);
    if a_bigrams.is_empty() || b_bigrams.is_empty() {
        return if a.contains(b) || b.contains(a) {
            0.65
        } else {
            0.0
        };
    }

    let overlap = a_bigrams.iter().filter(|bg| b_bigrams.contains(bg)).count();
    (2.0 * overlap as f64) / (a_bigrams.len() + b_bigrams.len()) as f64
}

fn bigrams(value: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::ModelCapabilities;

    fn sample(id: &str, name: &str, provider: &str) -> DiscoveredModel {
        DiscoveredModel {
            id: id.into(),
            name: name.into(),
            provider: provider.into(),
            context_length: 128_000,
            max_output_tokens: 4096,
            capabilities: ModelCapabilities {
                supports_function_calling: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn exact_id_match() {
        let models = vec![sample("gpt-4.1", "GPT-4.1", "openai")];
        let query = ModelSearchQuery::new("gpt-4.1");
        let hits = search_models(models, &query);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].match_kind, ModelMatchKind::ExactId);
        assert!((hits[0].score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn qualified_query_parses_provider() {
        let models = vec![
            sample("gpt-4.1", "GPT-4.1", "openai"),
            sample("gpt-4.1", "GPT-4.1", "azure"),
        ];
        let query = ModelSearchQuery::new("openai/gpt-4.1");
        let hits = search_models(models, &query);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].model.provider, "openai");
    }

    #[test]
    fn substring_search_finds_display_name() {
        let models = vec![sample("claude-opus-4-8", "Claude Opus 4.8", "anthropic")];
        let query = ModelSearchQuery::new("opus 4.8");
        let hits = search_models(models, &query);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].match_kind, ModelMatchKind::Contains);
    }

    #[test]
    fn fuzzy_search_handles_typos() {
        let models = vec![sample("gpt-4.1", "GPT-4.1", "openai")];
        let query = ModelSearchQuery::new("gpt-4l").fuzzy(true);
        let hits = search_models(models, &query);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].match_kind, ModelMatchKind::Fuzzy);
    }

    #[test]
    fn static_search_finds_known_model() {
        let query = ModelSearchQuery::new("claude sonnet").fuzzy(true);
        let hits = search_static_models(&query);
        assert!(hits.iter().any(|m| m.model.id.contains("claude-sonnet")));
    }

    #[test]
    fn search_respects_output_length_bounds() {
        let mut model = sample("big-out", "Big Out", "openai");
        model.max_output_tokens = 32_768;
        let models = vec![model];
        let query = ModelSearchQuery::new("big-out")
            .with_min_output_tokens(16_384)
            .with_max_output_tokens(65_536);
        let hits = search_models(models, &query);
        assert_eq!(hits.len(), 1);

        let mut small = sample("big-out", "Big Out", "openai");
        small.max_output_tokens = 4096;
        let query = ModelSearchQuery::new("big-out").with_min_output_tokens(65_536);
        assert!(search_models(vec![small], &query).is_empty());
    }

    #[test]
    fn search_respects_input_length_bounds() {
        let mut model = sample("big-ctx", "Big Context", "openai");
        model.context_length = 1_000_000;
        let models = vec![model];
        let query = ModelSearchQuery::new("big")
            .with_min_context_length(500_000)
            .with_max_context_length(2_000_000);
        assert_eq!(search_models(models.clone(), &query).len(), 1);

        let mut small = sample("big-ctx", "Big Context", "openai");
        small.context_length = 1_000_000;
        let query = ModelSearchQuery::new("big").with_max_context_length(500_000);
        assert!(search_models(vec![small], &query).is_empty());
    }

    #[test]
    fn static_lookup_by_name_falls_back_to_display_name() {
        let model = static_lookup_by_name("openai", "GPT-4.1");
        assert!(model.is_some());
        assert_eq!(model.unwrap().id, "gpt-4.1");
    }
}
