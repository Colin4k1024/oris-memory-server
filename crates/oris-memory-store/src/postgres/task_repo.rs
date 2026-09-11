//! CRUD for the `shared_task_context` table.

use crate::memory_types::{PrivacyClass, SharedTaskContext};
use sqlx::{PgPool, Row};
use uuid::Uuid;

pub struct TaskRepo {
    pool: PgPool,
}

impl TaskRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new shared task context.
    pub async fn insert(&self, task: &SharedTaskContext) -> Result<Uuid, TaskRepoError> {
        let task_id = if task.task_id == Uuid::nil() {
            Uuid::new_v4()
        } else {
            task.task_id
        };
        sqlx::query(
            r#"INSERT INTO shared_task_context (
                task_id, parent_task_id, initiator_user_id, organization_scope, goal,
                constraints, success_criteria, entities, business_refs, current_findings,
                evidence_refs, decisions, assumptions, completed_steps, pending_steps,
                current_owner_agent, participant_agents, artifact_refs, source_system_refs,
                status, version, expires_at, acl, privacy_class, audit_ref
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17, $18, $19, $20, 1, $21, $22, $23, $24
            )"#,
        )
        .bind(task_id)
        .bind(task.parent_task_id)
        .bind(&task.initiator_user_id)
        .bind(&task.organization_scope)
        .bind(&task.goal)
        .bind(serde_json::to_value(&task.constraints).unwrap_or_default())
        .bind(serde_json::to_value(&task.success_criteria).unwrap_or_default())
        .bind(serde_json::to_value(&task.entities).unwrap_or_default())
        .bind(serde_json::to_value(&task.business_refs).unwrap_or_default())
        .bind(serde_json::to_value(&task.current_findings).unwrap_or_default())
        .bind(serde_json::to_value(&task.evidence_refs).unwrap_or_default())
        .bind(serde_json::to_value(&task.decisions).unwrap_or_default())
        .bind(serde_json::to_value(&task.assumptions).unwrap_or_default())
        .bind(serde_json::to_value(&task.completed_steps).unwrap_or_default())
        .bind(serde_json::to_value(&task.pending_steps).unwrap_or_default())
        .bind(&task.current_owner_agent)
        .bind(serde_json::to_value(&task.participant_agents).unwrap_or_default())
        .bind(serde_json::to_value(&task.artifact_refs).unwrap_or_default())
        .bind(serde_json::to_value(&task.source_system_refs).unwrap_or_default())
        .bind(&task.status)
        .bind(task.expires_at)
        .bind(&task.acl)
        .bind(task.privacy_class.as_str())
        .bind(&task.audit_ref)
        .execute(&self.pool)
        .await?;
        Ok(task_id)
    }

    /// Fetch a task context by task_id.
    pub async fn get_by_id(
        &self,
        task_id: Uuid,
    ) -> Result<Option<SharedTaskContext>, TaskRepoError> {
        let row = sqlx::query(r#"SELECT * FROM shared_task_context WHERE task_id = $1"#)
            .bind(task_id)
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| map_row_to_task(&r)).transpose()
    }

    /// Append findings to a task (atomic JSONB array append).
    pub async fn add_findings(
        &self,
        task_id: Uuid,
        findings: &[serde_json::Value],
    ) -> Result<(), TaskRepoError> {
        let findings_json = serde_json::to_value(findings).unwrap_or_default();
        sqlx::query(
            r#"UPDATE shared_task_context
               SET current_findings = current_findings || $2::jsonb,
                   version = version + 1, updated_at = NOW()
               WHERE task_id = $1"#,
        )
        .bind(task_id)
        .bind(&findings_json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Update task status.
    pub async fn update_status(&self, task_id: Uuid, status: &str) -> Result<(), TaskRepoError> {
        sqlx::query(
            r#"UPDATE shared_task_context SET status = $2, updated_at = NOW()
               WHERE task_id = $1"#,
        )
        .bind(task_id)
        .bind(status)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Transfer task ownership to a new agent (handoff).
    pub async fn handoff(&self, task_id: Uuid, new_owner_agent: &str) -> Result<(), TaskRepoError> {
        sqlx::query(
            r#"UPDATE shared_task_context
               SET current_owner_agent = $2, version = version + 1, updated_at = NOW()
               WHERE task_id = $1"#,
        )
        .bind(task_id)
        .bind(new_owner_agent)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

fn map_row_to_task(row: &sqlx::postgres::PgRow) -> Result<SharedTaskContext, TaskRepoError> {
    Ok(SharedTaskContext {
        task_id: row.try_get("task_id")?,
        parent_task_id: row.try_get("parent_task_id")?,
        initiator_user_id: row.try_get("initiator_user_id")?,
        organization_scope: row.try_get("organization_scope")?,
        goal: row.try_get("goal")?,
        constraints: row
            .try_get::<serde_json::Value, _>("constraints")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        success_criteria: row
            .try_get::<serde_json::Value, _>("success_criteria")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        entities: row
            .try_get::<serde_json::Value, _>("entities")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        business_refs: row
            .try_get::<serde_json::Value, _>("business_refs")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        current_findings: row
            .try_get::<serde_json::Value, _>("current_findings")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        evidence_refs: row
            .try_get::<serde_json::Value, _>("evidence_refs")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        decisions: row
            .try_get::<serde_json::Value, _>("decisions")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        assumptions: row
            .try_get::<serde_json::Value, _>("assumptions")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        completed_steps: row
            .try_get::<serde_json::Value, _>("completed_steps")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        pending_steps: row
            .try_get::<serde_json::Value, _>("pending_steps")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        current_owner_agent: row.try_get("current_owner_agent")?,
        participant_agents: row
            .try_get::<serde_json::Value, _>("participant_agents")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        artifact_refs: row
            .try_get::<serde_json::Value, _>("artifact_refs")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        source_system_refs: row
            .try_get::<serde_json::Value, _>("source_system_refs")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        status: row.try_get("status")?,
        version: row.try_get("version")?,
        expires_at: row.try_get("expires_at")?,
        acl: row.try_get("acl").unwrap_or_default(),
        privacy_class: match row
            .try_get::<String, _>("privacy_class")
            .unwrap_or_default()
            .as_str()
        {
            "public" => PrivacyClass::Public,
            "confidential" => PrivacyClass::Confidential,
            "restricted" => PrivacyClass::Restricted,
            _ => PrivacyClass::Internal,
        },
        audit_ref: row.try_get("audit_ref")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum TaskRepoError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
}
