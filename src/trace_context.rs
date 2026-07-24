//! Lightweight task-local / span context for LLM calls (EdgeCrab integration).

use std::future::Future;

use tracing::Instrument;

/// Run `fut` inside a span annotated with conversation session + platform.
///
/// EdgeCrab wraps each provider attempt so GenAI traces can correlate by session.
pub async fn with_trace_context<F, T>(session_id: &str, platform: &str, fut: F) -> T
where
    F: Future<Output = T>,
{
    let span = tracing::info_span!(
        "edgequake.llm.call",
        session_id = session_id,
        platform = platform,
    );
    fut.instrument(span).await
}
