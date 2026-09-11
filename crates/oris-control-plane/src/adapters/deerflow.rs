//! DeerFlow Adapter (§3.2).
//!
//! DeerFlow is a multi-agent research/workflow framework. This adapter wraps
//! it (does **NOT** modify DeerFlow core) and implements
//! [`PluggableMemoryBackend`](crate::adapters::PluggableMemoryBackend).
//!
//! The adapter handles:
//! - **Context length limiting** — truncation to `context_limit` tokens.
//! - **Protocol conversion** — DeerFlow ↔ Memory API JSON.
//! - **Timeout** — configurable per-operation via [`TimeoutConfig`].
//! - **Degradation** — returns partial/empty results on failure.
//!
//! # Architecture
//!
//! ```text
//! DeerFlow Core (unmodified)
//!      │
//!      ▼
//! DeerFlowAdapter ──► Memory API Gateway (#26)
//!      │                      ▲
//!      │                      │
//!      └── IdentityResolver ──┘
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use tokio::time::timeout;
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use oris_memory_contract::memory_types::{MemoryItem, Scope};

use crate::adapters::{
    AdapterError, CandidateMemoryRequest, MemoryApiClient, MemoryBackendContract,
    MemoryQueryFilters, PluggableMemoryBackend, ProvenanceInfo, TimeoutConfig,
};
use crate::identity::IdentityResolver;
use crate::shared_task::SharedTaskManager;

/// DeerFlow adapter — implements `PluggableMemoryBackend` for DeerFlow.
///
/// Wraps DeerFlow's memory interface without modifying its core. All memory
/// operations go through the Memory API Gateway.
pub struct DeerFlowAdapter {
    /// HTTP client for the Memory API Gateway.
    memory_api_client: Arc<dyn MemoryApiClient>,
    /// Identity resolver reference — resolves tokens to canonical identities.
    identity_resolver: Arc<IdentityResolver>,
    /// Optional shared task manager for structured handoff via task_id.
    shared_task_manager: Option<Arc<SharedTaskManager>>,
    /// Memory backend contract — declares supported capabilities.
    memory_backend_contract: MemoryBackendContract,
    /// Maximum context length in tokens (hard cap).
    context_limit: usize,
    /// Timeout configuration for adapter operations.
    timeout_config: TimeoutConfig,
}

impl DeerFlowAdapter {
    /// Create a new DeerFlow adapter.
    pub fn new(
        memory_api_client: Arc<dyn MemoryApiClient>,
        identity_resolver: Arc<IdentityResolver>,
    ) -> Self {
        Self {
            memory_api_client,
            identity_resolver,
            shared_task_manager: None,
            memory_backend_contract: MemoryBackendContract::default(),
            context_limit: 4096,
            timeout_config: TimeoutConfig::default(),
        }
    }

    /// Attach a shared task manager for structured handoff.
    pub fn with_shared_task_manager(mut self, mgr: Arc<SharedTaskManager>) -> Self {
        self.shared_task_manager = Some(mgr);
        self
    }

    /// Override the context token limit (hard cap for injection).
    pub fn with_context_limit(mut self, limit: usize) -> Self {
        self.context_limit = limit;
        self
    }

    /// Override the timeout configuration.
    pub fn with_timeout_config(mut self, config: TimeoutConfig) -> Self {
        self.timeout_config = config;
        self
    }

    /// Override the memory backend contract.
    pub fn with_contract(mut self, contract: MemoryBackendContract) -> Self {
        self.memory_backend_contract = contract;
        self
    }

    /// Truncate context text to fit within the token budget.
    ///
    /// Returns `(limited_text, was_truncated)`.
    fn limit_context(&self, text: &str, budget: usize) -> (String, bool) {
        let limit = budget.min(self.context_limit);
        let estimated_tokens = text.len() / 4;
        if estimated_tokens <= limit {
            return (text.to_string(), false);
        }
        // Truncate to approximately `limit * 4` characters.
        let char_limit = limit * 4;
        let truncated = if text.len() > char_limit {
            let mut end = char_limit;
            // Try to truncate at a character boundary to avoid panicking.
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            text[..end].to_string()
        } else {
            text.to_string()
        };
        (truncated, true)
    }

    /// Complete a structured handoff via task_id using [`SharedTaskManager`].
    ///
    /// Updates the task status to "completed" and invalidates the cache.
    pub async fn complete_handoff(&self, task_id: Uuid) -> Result<(), AdapterError> {
        let mgr = self
            .shared_task_manager
            .as_ref()
            .ok_or_else(|| AdapterError::Handoff("no shared task manager configured".into()))?;

        mgr.update_status(task_id, "completed")
            .await
            .map_err(|e| AdapterError::Handoff(e.to_string()))?;

        let _ = mgr.invalidate_cache(task_id).await;
        Ok(())
    }

    /// Get a reference to the identity resolver.
    pub fn identity_resolver(&self) -> &IdentityResolver {
        &self.identity_resolver
    }
}

#[async_trait]
impl PluggableMemoryBackend for DeerFlowAdapter {
    /// Inject necessary context at task start.
    ///
    /// Reads user context via the Memory API and truncates to `budget` tokens.
    /// On timeout or API error, degrades to an empty string.
    #[instrument(skip(self))]
    async fn inject_context(&self, task_id: Uuid, budget: usize) -> String {
        if !self.memory_backend_contract.supports_injection {
            return String::new();
        }

        let result = timeout(
            self.timeout_config.injection_timeout,
            self.memory_api_client
                .read_user_context("system", Some(task_id)),
        )
        .await;

        match result {
            Ok(Ok(raw)) => {
                let (limited, compressed) = self.limit_context(&raw, budget);
                if compressed {
                    debug!(
                        task_id = %task_id,
                        budget,
                        "context was compressed to fit budget"
                    );
                }
                limited
            }
            Ok(Err(e)) => {
                warn!(error = %e, task_id = %task_id, "context injection degraded");
                String::new()
            }
            Err(_) => {
                warn!(
                    task_id = %task_id,
                    timeout = ?self.timeout_config.injection_timeout,
                    "context injection timed out"
                );
                String::new()
            }
        }
    }

    /// Submit results and memory candidates at task end.
    ///
    /// Each candidate is submitted with a timeout. On timeout, returns
    /// [`AdapterError::Timeout`]. On API error, propagates the error.
    #[instrument(skip(self, results, memory_candidates))]
    async fn submit_results(
        &self,
        task_id: Uuid,
        results: &str,
        memory_candidates: &[CandidateMemoryRequest],
    ) -> Result<(), AdapterError> {
        if !self.memory_backend_contract.supports_submission {
            return Ok(());
        }
        let _ = results; // results text is logged for audit; not submitted separately

        for candidate in memory_candidates {
            match timeout(
                self.timeout_config.api_timeout,
                self.memory_api_client.submit_candidate(candidate),
            )
            .await
            {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    warn!(error = %e, task_id = %task_id, "candidate submission failed");
                    return Err(e);
                }
                Err(_) => {
                    warn!(
                        task_id = %task_id,
                        timeout = ?self.timeout_config.api_timeout,
                        "candidate submission timed out"
                    );
                    return Err(AdapterError::Timeout(self.timeout_config.api_timeout));
                }
            }
        }

        // Complete the handoff if a task manager is configured.
        if let Some(mgr) = &self.shared_task_manager {
            mgr.update_status(task_id, "completed")
                .await
                .map_err(|e| AdapterError::Handoff(e.to_string()))?;
        }

        Ok(())
    }

    /// Pluggable memory query.
    ///
    /// On timeout or API error, degrades to an empty vector.
    #[instrument(skip(self))]
    async fn query_memory(&self, query: &str, filters: &MemoryQueryFilters) -> Vec<MemoryItem> {
        if !self.memory_backend_contract.supports_query {
            return Vec::new();
        }

        let result = timeout(
            self.timeout_config.api_timeout,
            self.memory_api_client.query_memories(query, filters),
        )
        .await;

        match result {
            Ok(Ok(items)) => items,
            Ok(Err(e)) => {
                warn!(error = %e, "memory query degraded");
                Vec::new()
            }
            Err(_) => {
                warn!(
                    timeout = ?self.timeout_config.api_timeout,
                    "memory query timed out"
                );
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::test_support::{
        make_memory_item, sample_context_json, MockMemoryApiClient,
    };
    use crate::identity::{IdentityResolver, NoopIamClient};
    use std::time::Duration;
    use uuid::Uuid;

    fn make_adapter(mock: MockMemoryApiClient) -> DeerFlowAdapter {
        let resolver = IdentityResolver::new(Arc::new(NoopIamClient::new("acme")));
        DeerFlowAdapter::new(Arc::new(mock), Arc::new(resolver))
    }

    // ── inject_context tests ──

    #[tokio::test]
    async fn inject_context_returns_context_string() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![]);
        let adapter = make_adapter(mock);

        let ctx = adapter.inject_context(Uuid::new_v4(), 100).await;

        assert!(!ctx.is_empty());
        assert!(ctx.contains("Canonical User Context"));
    }

    #[tokio::test]
    async fn inject_context_truncates_to_budget() {
        let long_text = "x".repeat(1000);
        let mock = MockMemoryApiClient::new(&long_text, Uuid::new_v4(), vec![]);
        let adapter = make_adapter(mock);

        let ctx = adapter.inject_context(Uuid::new_v4(), 10).await;

        // 10 tokens ≈ 40 chars; with context_limit of 4096 the budget (10) wins.
        assert!(ctx.len() <= 40);
    }

    #[tokio::test]
    async fn inject_context_degrades_on_timeout() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![])
            .with_delay(200);
        let adapter = make_adapter(mock).with_timeout_config(TimeoutConfig {
            injection_timeout: Duration::from_millis(50),
            ..Default::default()
        });

        let ctx = adapter.inject_context(Uuid::new_v4(), 100).await;

        assert!(ctx.is_empty(), "timeout should degrade to empty string");
    }

    #[tokio::test]
    async fn inject_context_degrades_on_api_error() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![])
            .with_read_error();
        let adapter = make_adapter(mock);

        let ctx = adapter.inject_context(Uuid::new_v4(), 100).await;

        assert!(ctx.is_empty(), "API error should degrade to empty string");
    }

    #[tokio::test]
    async fn inject_context_returns_empty_when_disabled() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![]);
        let contract = MemoryBackendContract {
            supports_injection: false,
            ..Default::default()
        };
        let adapter = make_adapter(mock).with_contract(contract);

        let ctx = adapter.inject_context(Uuid::new_v4(), 100).await;

        assert!(ctx.is_empty(), "disabled injection returns empty");
    }

    // ── submit_results tests ──

    #[tokio::test]
    async fn submit_results_submits_all_candidates() {
        let id = Uuid::new_v4();
        let mock = MockMemoryApiClient::new(&sample_context_json(), id, vec![]);
        let adapter = make_adapter(mock);

        let candidates = vec![
            CandidateMemoryRequest {
                content: "fact 1".into(),
                scope: Scope::Personal,
                provenance: ProvenanceInfo::from_agent("deerflow"),
                confidence: None,
            },
            CandidateMemoryRequest {
                content: "fact 2".into(),
                scope: Scope::Team,
                provenance: ProvenanceInfo::from_agent("deerflow"),
                confidence: Some(0.9),
            },
        ];

        let result = adapter
            .submit_results(Uuid::new_v4(), "results text", &candidates)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn submit_results_propagates_timeout_error() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![])
            .with_delay(200);
        let adapter = make_adapter(mock).with_timeout_config(TimeoutConfig {
            api_timeout: Duration::from_millis(50),
            ..Default::default()
        });

        let candidates = vec![CandidateMemoryRequest {
            content: "fact".into(),
            scope: Scope::Personal,
            provenance: ProvenanceInfo::from_agent("deerflow"),
            confidence: None,
        }];

        let result = adapter
            .submit_results(Uuid::new_v4(), "results", &candidates)
            .await;

        assert!(matches!(result, Err(AdapterError::Timeout(_))));
    }

    #[tokio::test]
    async fn submit_results_propagates_api_error() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![])
            .with_submit_error();
        let adapter = make_adapter(mock);

        let candidates = vec![CandidateMemoryRequest {
            content: "fact".into(),
            scope: Scope::Personal,
            provenance: ProvenanceInfo::from_agent("deerflow"),
            confidence: None,
        }];

        let result = adapter
            .submit_results(Uuid::new_v4(), "results", &candidates)
            .await;

        assert!(matches!(result, Err(AdapterError::MemoryApi(_))));
    }

    #[tokio::test]
    async fn submit_results_skips_when_disabled() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![]);
        let contract = MemoryBackendContract {
            supports_submission: false,
            ..Default::default()
        };
        let adapter = make_adapter(mock).with_contract(contract);

        let candidates = vec![CandidateMemoryRequest {
            content: "fact".into(),
            scope: Scope::Personal,
            provenance: ProvenanceInfo::from_agent("deerflow"),
            confidence: None,
        }];

        let result = adapter
            .submit_results(Uuid::new_v4(), "results", &candidates)
            .await;

        assert!(result.is_ok(), "disabled submission is a no-op");
    }

    #[tokio::test]
    async fn submit_results_succeeds_with_empty_candidates() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![]);
        let adapter = make_adapter(mock);

        let result = adapter.submit_results(Uuid::new_v4(), "results", &[]).await;

        assert!(result.is_ok());
    }

    // ── query_memory tests ──

    #[tokio::test]
    async fn query_memory_returns_items() {
        let items = vec![make_memory_item("fact one"), make_memory_item("fact two")];
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), items.clone());
        let adapter = make_adapter(mock);

        let results = adapter
            .query_memory("test query", &MemoryQueryFilters::default())
            .await;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content.as_deref(), Some("fact one"));
    }

    #[tokio::test]
    async fn query_memory_degrades_on_timeout() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![])
            .with_delay(200);
        let adapter = make_adapter(mock).with_timeout_config(TimeoutConfig {
            api_timeout: Duration::from_millis(50),
            ..Default::default()
        });

        let results = adapter
            .query_memory("test query", &MemoryQueryFilters::default())
            .await;

        assert!(results.is_empty(), "timeout should degrade to empty vec");
    }

    #[tokio::test]
    async fn query_memory_degrades_on_api_error() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![])
            .with_query_error();
        let adapter = make_adapter(mock);

        let results = adapter
            .query_memory("test query", &MemoryQueryFilters::default())
            .await;

        assert!(results.is_empty(), "API error should degrade to empty vec");
    }

    #[tokio::test]
    async fn query_memory_returns_empty_when_disabled() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![]);
        let contract = MemoryBackendContract {
            supports_query: false,
            ..Default::default()
        };
        let adapter = make_adapter(mock).with_contract(contract);

        let results = adapter
            .query_memory("test query", &MemoryQueryFilters::default())
            .await;

        assert!(results.is_empty(), "disabled query returns empty");
    }

    // ── handoff tests ──

    #[tokio::test]
    async fn complete_handoff_fails_without_task_manager() {
        let mock = MockMemoryApiClient::new(&sample_context_json(), Uuid::new_v4(), vec![]);
        let adapter = make_adapter(mock);

        let result = adapter.complete_handoff(Uuid::new_v4()).await;

        assert!(matches!(result, Err(AdapterError::Handoff(_))));
    }

    // ── context limiting tests ──

    #[tokio::test]
    async fn limit_context_no_truncation_when_within_budget() {
        let short_text = "short context";
        let mock = MockMemoryApiClient::new(short_text, Uuid::new_v4(), vec![]);
        let adapter = make_adapter(mock);

        let ctx = adapter.inject_context(Uuid::new_v4(), 100).await;

        assert_eq!(ctx, short_text);
    }

    #[tokio::test]
    async fn limit_context_respects_context_limit_cap() {
        // Budget is very large but context_limit caps at adapter's configured limit.
        let long_text = "y".repeat(1000);
        let mock = MockMemoryApiClient::new(&long_text, Uuid::new_v4(), vec![]);
        let adapter = make_adapter(mock).with_context_limit(5);

        let ctx = adapter.inject_context(Uuid::new_v4(), 99999).await;

        // context_limit = 5 tokens → ~20 chars.
        assert!(ctx.len() <= 20);
    }
}
