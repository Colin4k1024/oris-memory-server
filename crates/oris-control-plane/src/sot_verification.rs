//! Source of Truth Verification — ensures enterprise memory does not replace
//! authoritative Source of Truth systems for high-risk decisions.
//!
//! Architecture doc §0.5: "Don't use enterprise memory to replace Source of
//! Truth." High-risk decisions must verify memories against SoT systems
//! (ERP/MES/PLM/QMS/EAM/SCADA/Historian) before acting on them.
//!
//! # Flow
//!
//! 1. Check the memory's source type and temporal validity (`valid_to`).
//! 2. Query the relevant enterprise SoT systems via a read-only
//!    [`SoTAdapter`] interface.
//! 3. Record the verification result to audit and version trails (when
//!    recorders are configured).
//! 4. Low-confidence memories are never marked as safe for production
//!    control, regardless of SoT match outcome.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use oris_memory_store::memory_types::{AuthorityLevel, MemoryItem};

// ──────────────────────────── Errors ────────────────────────────

/// Errors that can occur during Source of Truth verification.
#[derive(Debug, Error)]
pub enum SoTError {
    /// The SoT adapter returned an error (network, auth, timeout, etc.).
    #[error("SoT adapter error: {0}")]
    Adapter(String),

    /// The memory item was not found.
    #[error("memory {0} not found")]
    MemoryNotFound(Uuid),

    /// The memory item has expired (`valid_to` is in the past).
    #[error("memory {0} has expired (valid_to: {1})")]
    Expired(Uuid, DateTime<Utc>),

    /// Serialization failure.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Audit recording failed.
    #[error("audit recording failed: {0}")]
    AuditRecording(String),

    /// Version recording failed.
    #[error("version recording failed: {0}")]
    VersionRecording(String),
}

// ──────────────────────────── Core Types ────────────────────────────

/// Identifies which enterprise Source of Truth system to query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SoTSystem {
    Erp,
    Mes,
    Plm,
    Qms,
    Eam,
    Scada,
    Historian,
}

impl SoTSystem {
    /// Human-readable identifier used in logs and audit trails.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Erp => "erp",
            Self::Mes => "mes",
            Self::Plm => "plm",
            Self::Qms => "qms",
            Self::Eam => "eam",
            Self::Scada => "scada",
            Self::Historian => "historian",
        }
    }
}

/// A single read-only query against a Source of Truth system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoTQuery {
    pub system: SoTSystem,
    pub entity_id: String,
    pub field: String,
    pub timestamp: Option<DateTime<Utc>>,
}

impl SoTQuery {
    /// Convenience constructor.
    pub fn new(system: SoTSystem, entity_id: impl Into<String>, field: impl Into<String>) -> Self {
        Self {
            system,
            entity_id: entity_id.into(),
            field: field.into(),
            timestamp: None,
        }
    }

    /// Composite key used by [`MockSoTAdapter`] to look up canned responses.
    fn mock_key(&self) -> String {
        format!("{}:{}:{}", self.system.as_str(), self.entity_id, self.field)
    }
}

/// The outcome of comparing a memory's value against a SoT system's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchResult {
    /// SoT value matches the memory value.
    Matched,
    /// SoT value differs from the memory value.
    Mismatched,
    /// The entity/field was not found in the SoT system.
    NotFound,
    /// The SoT system was unavailable (timeout, error, maintenance).
    SystemUnavailable,
}

/// A response from a single SoT system query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoTResponse {
    pub system: SoTSystem,
    pub entity_id: String,
    pub field: String,
    /// The value returned by the SoT system (serialised as a string).
    pub value: String,
    pub verified_at: DateTime<Utc>,
    pub match_result: MatchResult,
}

/// Overall verification status across all queried SoT systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    /// All SoT queries returned `Matched`.
    Verified,
    /// Some queries matched; others returned `NotFound` or `SystemUnavailable`.
    PartiallyVerified,
    /// All queries returned `SystemUnavailable` or no queries were issued.
    Unverified,
    /// At least one query returned `Mismatched`.
    Conflicting,
}

/// The complete result of verifying a memory item against SoT systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub memory_id: Uuid,
    pub sot_responses: Vec<SoTResponse>,
    pub overall: VerificationStatus,
    pub verified_at: DateTime<Utc>,
    /// Whether this memory may safely drive production-control decisions.
    /// False when the memory is low-confidence, expired, or not fully
    /// verified — enforcing the §0.5 constraint.
    pub safe_for_production_control: bool,
}

impl VerificationResult {
    /// A human-readable change-reason string for version snapshots.
    pub fn version_reason(&self) -> String {
        format!(
            "SoT verification: {:?} ({} responses, safe={})",
            self.overall,
            self.sot_responses.len(),
            self.safe_for_production_control
        )
    }

    /// A human-readable action string for audit entries.
    pub fn audit_action(&self) -> String {
        format!("sot_verification_{:?}", self.overall).to_lowercase()
    }
}

// ──────────────────────────── Validity Checker ────────────────────────────

/// Checks whether a memory item is temporally valid and needs re-verification.
///
/// A memory is "expired" when its `valid_to` timestamp is in the past.
/// A memory "needs re-verification" when it is expired or has never been
/// verified (`last_verified_at` is `None`).
pub struct ValidityChecker;

/// Minimum confidence for a memory to be considered for production control.
const PRODUCTION_CONFIDENCE_THRESHOLD: f32 = 0.7;

impl ValidityChecker {
    /// Returns `true` if the memory's `valid_to` is in the past.
    pub fn is_expired(item: &MemoryItem) -> bool {
        match item.valid_to {
            Some(dt) => dt < Utc::now(),
            None => false,
        }
    }

    /// Returns `true` if the memory should be re-verified against SoT.
    ///
    /// Triggers when the memory is expired or has never been verified.
    pub fn needs_reverification(item: &MemoryItem) -> bool {
        if Self::is_expired(item) {
            return true;
        }
        item.last_verified_at.is_none()
    }

    /// Returns `true` if the memory's confidence is below the production
    /// control threshold.
    pub fn is_low_confidence(item: &MemoryItem) -> bool {
        item.confidence < PRODUCTION_CONFIDENCE_THRESHOLD
    }

    /// The confidence threshold used for production-control gating.
    pub fn confidence_threshold() -> f32 {
        PRODUCTION_CONFIDENCE_THRESHOLD
    }
}

// ──────────────────────────── SoT Adapter ────────────────────────────

/// Read-only async interface for querying external Source of Truth systems.
///
/// Implementations wrap ERP/MES/PLM/QMS/EAM/SCADA/Historian connectors.
/// The interface is intentionally read-only — memory must never *write*
/// to a Source of Truth system.
#[async_trait]
pub trait SoTAdapter: Send + Sync {
    /// Query a single field for a single entity from a SoT system.
    ///
    /// The implementation is responsible for comparing the SoT value
    /// against the expected memory value and setting `match_result`
    /// accordingly.  If the system is unreachable, return
    /// [`SoTError::Adapter`].
    async fn query(&self, query: &SoTQuery) -> Result<SoTResponse, SoTError>;
}

/// In-memory mock adapter for unit testing.
///
/// Pre-populate it with canned [`SoTResponse`]s keyed by
/// `"{system}:{entity_id}:{field}"`.  Queries without a matching key
/// return [`SoTError::Adapter`] (simulating an unavailable system).
pub struct MockSoTAdapter {
    responses: Mutex<HashMap<String, SoTResponse>>,
}

impl MockSoTAdapter {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(HashMap::new()),
        }
    }

    /// Insert a canned response for a given query key.
    pub fn insert(&self, query: &SoTQuery, response: SoTResponse) -> &Self {
        self.responses
            .lock()
            .expect("mock adapter mutex poisoned")
            .insert(query.mock_key(), response);
        self
    }

    /// Convenience: insert a canned response built from a query + value +
    /// match result.
    pub fn with(
        &self,
        query: &SoTQuery,
        value: impl Into<String>,
        match_result: MatchResult,
    ) -> &Self {
        let response = SoTResponse {
            system: query.system,
            entity_id: query.entity_id.clone(),
            field: query.field.clone(),
            value: value.into(),
            verified_at: Utc::now(),
            match_result,
        };
        self.insert(query, response)
    }
}

impl Default for MockSoTAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SoTAdapter for MockSoTAdapter {
    async fn query(&self, query: &SoTQuery) -> Result<SoTResponse, SoTError> {
        let key = query.mock_key();
        match self.responses.lock().expect("mutex poisoned").get(&key) {
            Some(resp) => Ok(resp.clone()),
            None => Err(SoTError::Adapter(format!(
                "no mock response for key: {key}"
            ))),
        }
    }
}

// ──────────────────────────── Recording Traits ────────────────────────────

/// Optional audit-trail recorder.  When set on the verifier, the
/// verification result is persisted after each `verify()` call.
#[async_trait]
pub trait AuditRecorder: Send + Sync {
    async fn record(&self, result: &VerificationResult) -> Result<(), SoTError>;
}

/// Optional version-trail recorder.  When set on the verifier, a version
/// snapshot reason is recorded after each `verify()` call.
#[async_trait]
pub trait VersionRecorder: Send + Sync {
    async fn record(&self, memory_id: Uuid, reason: &str) -> Result<(), SoTError>;
}

// ──────────────────────────── Source of Truth Verifier ────────────────────────────

/// Verifies enterprise memory items against Source of Truth systems.
///
/// Holds an [`SoTAdapter`] for read-only queries and optional
/// [`AuditRecorder`] / [`VersionRecorder`] hooks.  The verifier is
/// stateless beyond these injected collaborators.
pub struct SourceOfTruthVerifier {
    adapter: Arc<dyn SoTAdapter>,
    audit_recorder: Option<Arc<dyn AuditRecorder>>,
    version_recorder: Option<Arc<dyn VersionRecorder>>,
}

impl SourceOfTruthVerifier {
    /// Create a verifier backed by the given SoT adapter.
    pub fn new(adapter: Arc<dyn SoTAdapter>) -> Self {
        Self {
            adapter,
            audit_recorder: None,
            version_recorder: None,
        }
    }

    /// Attach an audit recorder.  When set, `verify()` persists the result.
    pub fn with_audit_recorder(mut self, recorder: Arc<dyn AuditRecorder>) -> Self {
        self.audit_recorder = Some(recorder);
        self
    }

    /// Attach a version recorder.  When set, `verify()` records a snapshot.
    pub fn with_version_recorder(mut self, recorder: Arc<dyn VersionRecorder>) -> Self {
        self.version_recorder = Some(recorder);
        self
    }

    /// Verify a memory item against the given SoT queries.
    ///
    /// # Steps
    /// 1. **Source & validity check** — L0 memories (already SoT) skip
    ///    verification.  Expired memories are still queried but flagged.
    /// 2. **SoT query** — each query is sent to the adapter.
    /// 3. **Record** — if recorders are set, the result is persisted to
    ///    audit and version trails.
    /// 4. **Production-control gate** — `safe_for_production_control` is
    ///    set to `false` when the memory is low-confidence, expired, or
    ///    not fully verified.
    #[instrument(skip(self, item, queries), fields(memory_id = %item.memory_id))]
    pub async fn verify(
        &self,
        item: &MemoryItem,
        queries: &[SoTQuery],
    ) -> Result<VerificationResult, SoTError> {
        let memory_id = item.memory_id;

        // 1. Source & validity check.
        // L0 memories are already Source of Truth — no external verification
        // needed.
        if item.authority_level == AuthorityLevel::L0SourceOfTruth {
            debug!(
                "memory {} is L0 (source of truth) — skipping verification",
                memory_id
            );
            let result = VerificationResult {
                memory_id,
                sot_responses: Vec::new(),
                overall: VerificationStatus::Verified,
                verified_at: Utc::now(),
                safe_for_production_control: !ValidityChecker::is_low_confidence(item)
                    && !ValidityChecker::is_expired(item),
            };
            self.record_result(&result).await?;
            return Ok(result);
        }

        let is_expired = ValidityChecker::is_expired(item);
        if is_expired {
            warn!(
                "memory {} has expired (valid_to: {:?}) — verifying but flagging",
                memory_id, item.valid_to
            );
        }

        // 2. Query each SoT system via the adapter.
        let mut responses: Vec<SoTResponse> = Vec::with_capacity(queries.len());
        for query in queries {
            match self.adapter.query(query).await {
                Ok(resp) => responses.push(resp),
                Err(SoTError::Adapter(msg)) => {
                    // Adapter failure → treat as system unavailable.
                    warn!(
                        "SoT adapter error for {} on {}:{} — {}",
                        query.system.as_str(),
                        query.entity_id,
                        query.field,
                        msg
                    );
                    responses.push(SoTResponse {
                        system: query.system,
                        entity_id: query.entity_id.clone(),
                        field: query.field.clone(),
                        value: String::new(),
                        verified_at: Utc::now(),
                        match_result: MatchResult::SystemUnavailable,
                    });
                }
                Err(e) => return Err(e),
            }
        }

        // 3. Aggregate overall status.
        let overall = aggregate_status(&responses);

        // 4. Production-control gate.
        let safe_for_production_control = overall == VerificationStatus::Verified
            && !ValidityChecker::is_low_confidence(item)
            && !is_expired;

        let result = VerificationResult {
            memory_id,
            sot_responses: responses,
            overall,
            verified_at: Utc::now(),
            safe_for_production_control,
        };

        // 5. Record to audit and version trails (if configured).
        self.record_result(&result).await?;

        Ok(result)
    }

    /// Persist the result to audit and version trails, if recorders are set.
    async fn record_result(&self, result: &VerificationResult) -> Result<(), SoTError> {
        if let Some(ref audit) = self.audit_recorder {
            audit
                .record(result)
                .await
                .map_err(|e| SoTError::AuditRecording(e.to_string()))?;
        }
        if let Some(ref version) = self.version_recorder {
            version
                .record(result.memory_id, &result.version_reason())
                .await
                .map_err(|e| SoTError::VersionRecording(e.to_string()))?;
        }
        Ok(())
    }
}

/// Determine the overall verification status from individual responses.
fn aggregate_status(responses: &[SoTResponse]) -> VerificationStatus {
    if responses.is_empty() {
        return VerificationStatus::Unverified;
    }

    let mut matched = 0;
    let mut mismatched = 0;

    for r in responses {
        match r.match_result {
            MatchResult::Matched => matched += 1,
            MatchResult::Mismatched => mismatched += 1,
            MatchResult::NotFound | MatchResult::SystemUnavailable => {}
        }
    }

    if mismatched > 0 {
        VerificationStatus::Conflicting
    } else if matched == responses.len() {
        VerificationStatus::Verified
    } else if matched > 0 {
        VerificationStatus::PartiallyVerified
    } else {
        VerificationStatus::Unverified
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oris_memory_store::memory_types::{
        AuthorityLevel, MemoryItem, MemoryStatus, MemoryType, PrivacyClass, Scope, SourceType,
    };
    use serde_json::json;
    use uuid::Uuid;

    // ── Helpers ──

    /// Build a `MemoryItem` with sensible test defaults.
    fn test_memory_item(
        authority: AuthorityLevel,
        confidence: f32,
        valid_to: Option<DateTime<Utc>>,
    ) -> MemoryItem {
        MemoryItem {
            memory_id: Uuid::new_v4(),
            tenant_id: "tenant-1".to_string(),
            memory_type: MemoryType::Semantic,
            scope: Scope::Personal,
            subject_type: Some("user".to_string()),
            subject_id: Some("user-123".to_string()),
            entity_refs: vec![json!({"type": "person", "id": "user-123"})],
            content: Some("test content".to_string()),
            structured_payload: Some(json!({"key": "value"})),
            embedding: None,
            source_type: SourceType::UserExplicit,
            source_reference: Some("ref-1".to_string()),
            evidence_refs: vec![json!({"ref": "evidence-1"})],
            confidence,
            authority_level: authority,
            importance: 0.9,
            observed_at: Some(Utc::now()),
            valid_from: Some(Utc::now()),
            valid_to,
            privacy_class: PrivacyClass::Internal,
            acl: json!({"read": ["user-123"]}),
            retention_policy: Some("default".to_string()),
            status: MemoryStatus::Active,
            version: 3,
            derived_from: vec![json!("parent-uuid")],
            created_by_user: Some("user-123".to_string()),
            created_by_agent: None,
            last_verified_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// In-memory audit recorder that captures results for assertion.
    struct CapturingAuditRecorder {
        recorded: Mutex<Vec<VerificationResult>>,
    }

    impl CapturingAuditRecorder {
        fn new() -> Self {
            Self {
                recorded: Mutex::new(Vec::new()),
            }
        }
        fn results(&self) -> Vec<VerificationResult> {
            self.recorded.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl AuditRecorder for CapturingAuditRecorder {
        async fn record(&self, result: &VerificationResult) -> Result<(), SoTError> {
            self.recorded.lock().unwrap().push(result.clone());
            Ok(())
        }
    }

    /// In-memory version recorder that captures (memory_id, reason) pairs.
    struct CapturingVersionRecorder {
        recorded: Mutex<Vec<(Uuid, String)>>,
    }

    impl CapturingVersionRecorder {
        fn new() -> Self {
            Self {
                recorded: Mutex::new(Vec::new()),
            }
        }
        fn pairs(&self) -> Vec<(Uuid, String)> {
            self.recorded.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl VersionRecorder for CapturingVersionRecorder {
        async fn record(&self, memory_id: Uuid, reason: &str) -> Result<(), SoTError> {
            self.recorded
                .lock()
                .unwrap()
                .push((memory_id, reason.to_string()));
            Ok(())
        }
    }

    // ── 1. SoTSystem tests ──

    #[test]
    fn sot_system_as_str_covers_all_variants() {
        assert_eq!(SoTSystem::Erp.as_str(), "erp");
        assert_eq!(SoTSystem::Mes.as_str(), "mes");
        assert_eq!(SoTSystem::Plm.as_str(), "plm");
        assert_eq!(SoTSystem::Qms.as_str(), "qms");
        assert_eq!(SoTSystem::Eam.as_str(), "eam");
        assert_eq!(SoTSystem::Scada.as_str(), "scada");
        assert_eq!(SoTSystem::Historian.as_str(), "historian");
    }

    // ── 2. SoTQuery mock_key ──

    #[test]
    fn sot_query_mock_key_format() {
        let q = SoTQuery::new(SoTSystem::Erp, "ORD-123", "status");
        assert_eq!(q.mock_key(), "erp:ORD-123:status");
    }

    // ── 3. aggregate_status logic ──

    #[test]
    fn aggregate_status_all_matched_is_verified() {
        let responses = vec![
            SoTResponse {
                system: SoTSystem::Erp,
                entity_id: "e1".into(),
                field: "f".into(),
                value: "v".into(),
                verified_at: Utc::now(),
                match_result: MatchResult::Matched,
            },
            SoTResponse {
                system: SoTSystem::Mes,
                entity_id: "e1".into(),
                field: "f".into(),
                value: "v".into(),
                verified_at: Utc::now(),
                match_result: MatchResult::Matched,
            },
        ];
        assert_eq!(aggregate_status(&responses), VerificationStatus::Verified);
    }

    // ── 4. aggregate_status with mismatch ──

    #[test]
    fn aggregate_status_mismatch_is_conflicting() {
        let responses = vec![
            SoTResponse {
                system: SoTSystem::Erp,
                entity_id: "e1".into(),
                field: "f".into(),
                value: "v".into(),
                verified_at: Utc::now(),
                match_result: MatchResult::Matched,
            },
            SoTResponse {
                system: SoTSystem::Mes,
                entity_id: "e1".into(),
                field: "f".into(),
                value: "w".into(),
                verified_at: Utc::now(),
                match_result: MatchResult::Mismatched,
            },
        ];
        assert_eq!(
            aggregate_status(&responses),
            VerificationStatus::Conflicting
        );
    }

    // ── 5. aggregate_status partial ──

    #[test]
    fn aggregate_status_partial_when_some_not_found() {
        let responses = vec![
            SoTResponse {
                system: SoTSystem::Erp,
                entity_id: "e1".into(),
                field: "f".into(),
                value: "v".into(),
                verified_at: Utc::now(),
                match_result: MatchResult::Matched,
            },
            SoTResponse {
                system: SoTSystem::Plm,
                entity_id: "e1".into(),
                field: "f".into(),
                value: "".into(),
                verified_at: Utc::now(),
                match_result: MatchResult::NotFound,
            },
        ];
        assert_eq!(
            aggregate_status(&responses),
            VerificationStatus::PartiallyVerified
        );
    }

    // ── 6. aggregate_status empty ──

    #[test]
    fn aggregate_status_empty_is_unverified() {
        assert_eq!(aggregate_status(&[]), VerificationStatus::Unverified);
    }

    // ── 7. aggregate_status all unavailable ──

    #[test]
    fn aggregate_status_all_unavailable_is_unverified() {
        let responses = vec![
            SoTResponse {
                system: SoTSystem::Erp,
                entity_id: "e1".into(),
                field: "f".into(),
                value: "".into(),
                verified_at: Utc::now(),
                match_result: MatchResult::SystemUnavailable,
            },
            SoTResponse {
                system: SoTSystem::Mes,
                entity_id: "e1".into(),
                field: "f".into(),
                value: "".into(),
                verified_at: Utc::now(),
                match_result: MatchResult::SystemUnavailable,
            },
        ];
        assert_eq!(aggregate_status(&responses), VerificationStatus::Unverified);
    }

    // ── 8. ValidityChecker::is_expired ──

    #[test]
    fn validity_checker_is_expired_when_past() {
        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, Some(Utc::now()));
        assert!(ValidityChecker::is_expired(&item));
    }

    // ── 9. ValidityChecker not expired ──

    #[test]
    fn validity_checker_not_expired_when_future() {
        let future = Utc::now() + chrono::Duration::days(30);
        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, Some(future));
        assert!(!ValidityChecker::is_expired(&item));
    }

    #[test]
    fn validity_checker_not_expired_when_none() {
        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, None);
        assert!(!ValidityChecker::is_expired(&item));
    }

    // ── 10. ValidityChecker::needs_reverification ──

    #[test]
    fn validity_checker_needs_reverification_when_never_verified() {
        let future = Utc::now() + chrono::Duration::days(30);
        let mut item = test_memory_item(AuthorityLevel::L2Verified, 0.9, Some(future));
        item.last_verified_at = None;
        assert!(ValidityChecker::needs_reverification(&item));
    }

    // ── 11. ValidityChecker::is_low_confidence ──

    #[test]
    fn validity_checker_low_confidence_threshold() {
        let future = Utc::now() + chrono::Duration::days(30);
        let low = test_memory_item(AuthorityLevel::L3Inferred, 0.5, Some(future));
        let high = test_memory_item(AuthorityLevel::L2Verified, 0.9, Some(future));
        assert!(ValidityChecker::is_low_confidence(&low));
        assert!(!ValidityChecker::is_low_confidence(&high));
    }

    // ── 12. MockSoTAdapter returns canned response ──

    #[tokio::test]
    async fn mock_adapter_returns_canned_response() {
        let mock = MockSoTAdapter::new();
        let query = SoTQuery::new(SoTSystem::Erp, "ORD-1", "status");
        mock.with(&query, "shipped", MatchResult::Matched);

        let resp = mock.query(&query).await.unwrap();
        assert_eq!(resp.system, SoTSystem::Erp);
        assert_eq!(resp.entity_id, "ORD-1");
        assert_eq!(resp.field, "status");
        assert_eq!(resp.value, "shipped");
        assert_eq!(resp.match_result, MatchResult::Matched);
    }

    // ── 13. MockSoTAdapter errors on unknown query ──

    #[tokio::test]
    async fn mock_adapter_errors_on_unknown_query() {
        let mock = MockSoTAdapter::new();
        let query = SoTQuery::new(SoTSystem::Mes, "WO-1", "quantity");
        let result = mock.query(&query).await;
        assert!(matches!(result, Err(SoTError::Adapter(_))));
    }

    // ── 14. Verifier skips L0 (source of truth) ──

    #[tokio::test]
    async fn verifier_skips_l0_source_of_truth() {
        let mock = Arc::new(MockSoTAdapter::new());
        let verifier = SourceOfTruthVerifier::new(mock.clone());
        let item = test_memory_item(AuthorityLevel::L0SourceOfTruth, 0.95, None);
        let queries = vec![SoTQuery::new(SoTSystem::Erp, "ORD-1", "status")];

        let result = verifier.verify(&item, &queries).await.unwrap();
        assert_eq!(result.overall, VerificationStatus::Verified);
        assert!(result.sot_responses.is_empty());
        assert!(result.safe_for_production_control);
    }

    // ── 15. Verifier: all matched → Verified + safe ──

    #[tokio::test]
    async fn verifier_all_matched_is_verified_and_safe() {
        let mock = Arc::new(MockSoTAdapter::new());
        let q1 = SoTQuery::new(SoTSystem::Erp, "ORD-1", "status");
        let q2 = SoTQuery::new(SoTSystem::Mes, "WO-1", "quantity");
        mock.with(&q1, "shipped", MatchResult::Matched);
        mock.with(&q2, "100", MatchResult::Matched);

        let verifier = SourceOfTruthVerifier::new(mock);
        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, None);
        let result = verifier.verify(&item, &[q1, q2]).await.unwrap();

        assert_eq!(result.overall, VerificationStatus::Verified);
        assert_eq!(result.sot_responses.len(), 2);
        assert!(result.safe_for_production_control);
    }

    // ── 16. Verifier: mismatch → Conflicting + unsafe ──

    #[tokio::test]
    async fn verifier_mismatch_is_conflicting_and_unsafe() {
        let mock = Arc::new(MockSoTAdapter::new());
        let q1 = SoTQuery::new(SoTSystem::Erp, "ORD-1", "status");
        mock.with(&q1, "cancelled", MatchResult::Mismatched);

        let verifier = SourceOfTruthVerifier::new(mock);
        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, None);
        let result = verifier.verify(&item, &[q1]).await.unwrap();

        assert_eq!(result.overall, VerificationStatus::Conflicting);
        assert!(!result.safe_for_production_control);
    }

    // ── 17. Verifier: low confidence → unsafe even when verified ──

    #[tokio::test]
    async fn verifier_low_confidence_unsafe_even_when_verified() {
        let mock = Arc::new(MockSoTAdapter::new());
        let q1 = SoTQuery::new(SoTSystem::Erp, "ORD-1", "status");
        mock.with(&q1, "shipped", MatchResult::Matched);

        let verifier = SourceOfTruthVerifier::new(mock);
        let item = test_memory_item(AuthorityLevel::L3Inferred, 0.5, None);
        let result = verifier.verify(&item, &[q1]).await.unwrap();

        assert_eq!(result.overall, VerificationStatus::Verified);
        assert!(
            !result.safe_for_production_control,
            "low-confidence memory must not be safe for production control"
        );
    }

    // ── 18. Verifier: adapter failure → SystemUnavailable ──

    #[tokio::test]
    async fn verifier_adapter_failure_becomes_system_unavailable() {
        let mock = Arc::new(MockSoTAdapter::new());
        // No canned response → adapter error
        let q1 = SoTQuery::new(SoTSystem::Erp, "ORD-1", "status");

        let verifier = SourceOfTruthVerifier::new(mock);
        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, None);
        let result = verifier.verify(&item, &[q1]).await.unwrap();

        assert_eq!(result.sot_responses.len(), 1);
        assert_eq!(
            result.sot_responses[0].match_result,
            MatchResult::SystemUnavailable
        );
        assert_eq!(result.overall, VerificationStatus::Unverified);
    }

    // ── 19. Verifier: expired memory → not safe ──

    #[tokio::test]
    async fn verifier_expired_memory_not_safe_even_when_verified() {
        let mock = Arc::new(MockSoTAdapter::new());
        let q1 = SoTQuery::new(SoTSystem::Erp, "ORD-1", "status");
        mock.with(&q1, "shipped", MatchResult::Matched);

        let verifier = SourceOfTruthVerifier::new(mock);
        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, Some(Utc::now()));
        let result = verifier.verify(&item, &[q1]).await.unwrap();

        assert_eq!(result.overall, VerificationStatus::Verified);
        assert!(
            !result.safe_for_production_control,
            "expired memory must not be safe for production control"
        );
    }

    // ── 20. Verifier: records to audit and version ──

    #[tokio::test]
    async fn verifier_records_to_audit_and_version() {
        let mock = Arc::new(MockSoTAdapter::new());
        let q1 = SoTQuery::new(SoTSystem::Erp, "ORD-1", "status");
        mock.with(&q1, "shipped", MatchResult::Matched);

        let audit = Arc::new(CapturingAuditRecorder::new());
        let version = Arc::new(CapturingVersionRecorder::new());

        let verifier = SourceOfTruthVerifier::new(mock)
            .with_audit_recorder(audit.clone())
            .with_version_recorder(version.clone());

        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, None);
        let result = verifier.verify(&item, &[q1]).await.unwrap();

        assert_eq!(audit.results().len(), 1);
        assert_eq!(audit.results()[0].memory_id, result.memory_id);

        assert_eq!(version.pairs().len(), 1);
        assert_eq!(version.pairs()[0].0, result.memory_id);
        assert!(version.pairs()[0].1.contains("verification"));
    }

    // ── 21. Verifier: no queries → Unverified ──

    #[tokio::test]
    async fn verifier_no_queries_is_unverified() {
        let mock = Arc::new(MockSoTAdapter::new());
        let verifier = SourceOfTruthVerifier::new(mock);
        let item = test_memory_item(AuthorityLevel::L2Verified, 0.9, None);

        let result = verifier.verify(&item, &[]).await.unwrap();
        assert_eq!(result.overall, VerificationStatus::Unverified);
        assert!(result.sot_responses.is_empty());
        assert!(!result.safe_for_production_control);
    }

    // ── 22. VerificationResult helpers ──

    #[test]
    fn verification_result_version_reason_and_audit_action() {
        let result = VerificationResult {
            memory_id: Uuid::new_v4(),
            sot_responses: vec![],
            overall: VerificationStatus::Verified,
            verified_at: Utc::now(),
            safe_for_production_control: true,
        };
        let reason = result.version_reason();
        assert!(reason.contains("Verified"));
        assert!(reason.contains("safe=true"));

        let action = result.audit_action();
        assert!(action.contains("verified"));
    }

    // ── 23. SoTError display ──

    #[test]
    fn sot_error_display_messages() {
        let id = Uuid::new_v4();
        let e = SoTError::MemoryNotFound(id);
        assert!(format!("{e}").contains("not found"));

        let e = SoTError::Adapter("connection refused".into());
        assert!(format!("{e}").contains("connection refused"));

        let e = SoTError::AuditRecording("timeout".into());
        assert!(format!("{e}").contains("audit"));
        assert!(format!("{e}").contains("timeout"));
    }

    // ── 24. SoTSystem serde round-trip ──

    #[test]
    fn sot_system_serde_roundtrip() {
        let json = serde_json::to_string(&SoTSystem::Erp).unwrap();
        assert_eq!(json, "\"erp\"");
        let back: SoTSystem = serde_json::from_str(&json).unwrap();
        assert_eq!(back, SoTSystem::Erp);
    }
}
