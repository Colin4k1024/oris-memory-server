//! Shared Task Memory — cross-agent collaboration context.
//!
//! Implements the shared task model from architecture doc §6.  A task
//! context is shared **by reference** — all participating agents read and
//! write through the same `task_id`.  Concurrent writes use optimistic
//! concurrency (version checks in the store layer).  Handoff between agents
//! requires a distributed lock to prevent races.
//!
//! Hot-context (Redis) is a non-authoritative read-through cache.  Any write
//! invalidates the cached entry so stale data is never served.

use uuid::Uuid;

use oris_memory_store::memory_types::SharedTaskContext;
use oris_memory_store::postgres::task_repo::{TaskRepo, TaskRepoError};
use oris_memory_store::postgres::Pool;
use oris_memory_store::redis::hot_context::{HotContextRepo, DEFAULT_TTL_SECS};
use oris_memory_store::redis::session::SessionRepo;

/// Manages shared task context for cross-agent collaboration.
pub struct SharedTaskManager {
    task_repo: TaskRepo,
    hot_context: Option<HotContextRepo>,
    session: Option<SessionRepo>,
}

/// Handoff request when transferring task ownership between agents.
#[derive(Debug, Clone)]
pub struct TaskHandoffRequest {
    pub task_id: Uuid,
    pub from_agent: String,
    pub to_agent: String,
    pub reason: String,
    pub pending_steps: Vec<serde_json::Value>,
}

// ── Error ───────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SharedTaskError {
    #[error("{0}")]
    Repo(#[from] TaskRepoError),
    #[error("cache error: {0}")]
    Cache(String),
    #[error("lock error: {0}")]
    Lock(String),
    #[error("invalid handoff: {0}")]
    InvalidHandoff(String),
    #[error("task not found")]
    NotFound,
}

// ── TaskHandoffRequest ──────────────────────────────────────────

impl TaskHandoffRequest {
    /// Validate the handoff request before attempting the lock.
    pub fn validate(&self) -> Result<(), SharedTaskError> {
        if self.from_agent.is_empty() || self.to_agent.is_empty() {
            return Err(SharedTaskError::InvalidHandoff(
                "from_agent and to_agent must not be empty".into(),
            ));
        }
        if self.from_agent == self.to_agent {
            return Err(SharedTaskError::InvalidHandoff(format!(
                "from_agent and to_agent must differ (both are '{agent}')",
                agent = self.from_agent
            )));
        }
        Ok(())
    }

    /// Redis lock resource key for this handoff.
    pub fn lock_resource(&self) -> String {
        format!("task:{}", self.task_id)
    }

    /// Lock holder identifier (the releasing agent).
    pub fn lock_holder(&self) -> &str {
        &self.from_agent
    }
}

// ── Manager ─────────────────────────────────────────────────────

impl SharedTaskManager {
    /// Create a new manager backed by the given PostgreSQL pool.
    pub fn new(pool: Pool) -> Self {
        Self {
            task_repo: TaskRepo::new(pool),
            hot_context: None,
            session: None,
        }
    }

    /// Attach a hot-context (Redis) repository.  Builder-style.
    pub fn with_hot_context(mut self, repo: HotContextRepo) -> Self {
        self.hot_context = Some(repo);
        self
    }

    /// Attach a session (Redis) repository for distributed locks.  Builder-style.
    pub fn with_session(mut self, repo: SessionRepo) -> Self {
        self.session = Some(repo);
        self
    }

    /// Create a new shared task context.
    pub async fn create_task(&self, task: &SharedTaskContext) -> Result<Uuid, SharedTaskError> {
        let task_id = self.task_repo.insert(task).await?;
        Ok(task_id)
    }

    /// Get task context, hot context cache first.
    pub async fn get_context(
        &self,
        task_id: Uuid,
    ) -> Result<Option<SharedTaskContext>, SharedTaskError> {
        let key = task_id.to_string();

        // 1. Try hot-context cache.
        if let Some(ref hc) = self.hot_context {
            match hc.get_task_context(&key).await {
                Ok(Some(data)) => {
                    if let Ok(task) = serde_json::from_slice::<SharedTaskContext>(&data) {
                        return Ok(Some(task));
                    }
                    // Deserialization failed — fall through to DB.
                }
                Ok(None) => {} // cache miss — fall through
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        task_id = %task_id,
                        "hot context read failed, falling back to DB"
                    );
                }
            }
        }

        // 2. Fall back to PostgreSQL.
        let task = self.task_repo.get_by_id(task_id).await?;

        // 3. Populate cache (best-effort).
        if let Some(ref task) = task {
            self.populate_cache(task_id, task).await;
        }

        Ok(task)
    }

    /// Add findings to a task (concurrent-safe via store-layer version check).
    pub async fn add_findings(
        &self,
        task_id: Uuid,
        findings: Vec<serde_json::Value>,
    ) -> Result<(), SharedTaskError> {
        self.task_repo.add_findings(task_id, &findings).await?;

        self.invalidate_cache_best_effort(task_id).await;
        Ok(())
    }

    /// Handoff task from one agent to another.
    ///
    /// 1. Acquire distributed lock (`SessionRepo`)
    /// 2. Update task ownership
    /// 3. Append pending steps as findings for the incoming owner
    /// 4. Invalidate hot context
    /// 5. Release lock
    pub async fn handoff(&self, req: TaskHandoffRequest) -> Result<(), SharedTaskError> {
        req.validate()?;

        let session = self.session.as_ref().ok_or_else(|| {
            SharedTaskError::InvalidHandoff(
                "distributed lock (SessionRepo) is required for handoff".into(),
            )
        })?;

        let resource = req.lock_resource();
        let holder = req.lock_holder();
        let lock_ttl = 30_u64;

        // 1. Acquire distributed lock.
        let acquired = session
            .acquire_lock(&resource, holder, lock_ttl)
            .await
            .map_err(|e| SharedTaskError::Lock(e.to_string()))?;

        if !acquired {
            return Err(SharedTaskError::Lock(format!(
                "could not acquire lock for task {task_id} — another handoff in progress",
                task_id = req.task_id
            )));
        }

        // 2-4. Perform handoff; lock is released regardless of outcome.
        let result = self.perform_handoff(&req).await;

        // 5. Release lock (best-effort; log on failure).
        if let Err(e) = session.release_lock(&resource, holder).await {
            tracing::warn!(
                error = %e,
                resource = %resource,
                "failed to release handoff lock"
            );
        }

        result
    }

    /// Update task status.
    pub async fn update_status(&self, task_id: Uuid, status: &str) -> Result<(), SharedTaskError> {
        self.task_repo.update_status(task_id, status).await?;

        self.invalidate_cache_best_effort(task_id).await;
        Ok(())
    }

    /// Invalidate task hot-context cache.
    pub async fn invalidate_cache(&self, task_id: Uuid) -> Result<(), SharedTaskError> {
        if let Some(ref hc) = self.hot_context {
            hc.invalidate_task_context(&task_id.to_string())
                .await
                .map_err(|e| SharedTaskError::Cache(e.to_string()))?;
        }
        Ok(())
    }

    // ── private helpers ──────────────────────────────────────────

    /// Best-effort cache population — logs and swallows Redis errors.
    async fn populate_cache(&self, task_id: Uuid, task: &SharedTaskContext) {
        if let Some(ref hc) = self.hot_context {
            if let Ok(data) = serde_json::to_vec(task) {
                if let Err(e) = hc
                    .set_task_context(&task_id.to_string(), &data, DEFAULT_TTL_SECS)
                    .await
                {
                    tracing::warn!(
                        error = %e,
                        task_id = %task_id,
                        "failed to populate task hot context"
                    );
                }
            }
        }
    }

    /// Best-effort cache invalidation — logs and swallows Redis errors.
    async fn invalidate_cache_best_effort(&self, task_id: Uuid) {
        if let Some(ref hc) = self.hot_context {
            if let Err(e) = hc.invalidate_task_context(&task_id.to_string()).await {
                tracing::warn!(
                    error = %e,
                    task_id = %task_id,
                    "failed to invalidate task hot context"
                );
            }
        }
    }

    /// Perform the handoff after the lock is acquired.
    async fn perform_handoff(&self, req: &TaskHandoffRequest) -> Result<(), SharedTaskError> {
        // 2. Update task ownership.
        self.task_repo.handoff(req.task_id, &req.to_agent).await?;

        // 3. Append pending steps as findings for the incoming owner.
        if !req.pending_steps.is_empty() {
            self.task_repo
                .add_findings(req.task_id, &req.pending_steps)
                .await?;
        }

        // 4. Invalidate hot context.
        self.invalidate_cache_best_effort(req.task_id).await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handoff(from: &str, to: &str) -> TaskHandoffRequest {
        TaskHandoffRequest {
            task_id: Uuid::new_v4(),
            from_agent: from.to_string(),
            to_agent: to.to_string(),
            reason: "load balance".into(),
            pending_steps: vec![serde_json::json!({"step": "review"})],
        }
    }

    #[test]
    fn handoff_validate_accepts_different_agents() {
        let req = make_handoff("agent-a", "agent-b");
        assert!(req.validate().is_ok());
    }

    #[test]
    fn handoff_validate_rejects_same_agent() {
        let req = make_handoff("agent-a", "agent-a");
        let err = req.validate().unwrap_err();
        assert!(matches!(err, SharedTaskError::InvalidHandoff(_)));
        assert!(err.to_string().contains("must differ"));
    }

    #[test]
    fn handoff_validate_rejects_empty_from_agent() {
        let mut req = make_handoff("", "agent-b");
        req.from_agent = String::new();
        let err = req.validate().unwrap_err();
        assert!(matches!(err, SharedTaskError::InvalidHandoff(_)));
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn handoff_validate_rejects_empty_to_agent() {
        let mut req = make_handoff("agent-a", "agent-b");
        req.to_agent = String::new();
        assert!(req.validate().is_err());
    }

    #[test]
    fn handoff_lock_resource_format() {
        let req = make_handoff("agent-a", "agent-b");
        assert_eq!(req.lock_resource(), format!("task:{}", req.task_id));
    }

    #[test]
    fn handoff_lock_holder_is_from_agent() {
        let req = make_handoff("agent-a", "agent-b");
        assert_eq!(req.lock_holder(), "agent-a");
    }
}
