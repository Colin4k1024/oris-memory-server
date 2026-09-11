//! Retention policy engine and privacy classification.
//!
//! Each [`RetentionPolicy`] maps a `(Scope, MemoryType)` pair to a lifecycle
//! schedule expressed in days.  The [`PolicyEngine`] is a pure in-memory
//! lookup — no database access — and answers questions such as "should this
//! memory be archived by now?" and "should it be deleted?".

use chrono::{DateTime, Utc};
use oris_memory_store::memory_types::{MemoryItem, MemoryType, Scope};

/// A single retention policy entry.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    pub policy_id: String,
    pub scope: Scope,
    pub memory_type: MemoryType,
    /// Time-to-live in days.  `None` means the memory never expires.
    pub ttl_days: Option<u32>,
    /// Archive the memory after this many days from creation.
    pub archive_after_days: Option<u32>,
    /// Hard-delete the memory after this many days from creation.
    pub delete_after_days: Option<u32>,
}

/// In-memory retention policy lookup engine.
pub struct PolicyEngine {
    policies: Vec<RetentionPolicy>,
}

impl PolicyEngine {
    /// Create a new engine from a list of policies.
    pub fn new(policies: Vec<RetentionPolicy>) -> Self {
        Self { policies }
    }

    /// Returns a slice of all registered policies.
    pub fn policies(&self) -> &[RetentionPolicy] {
        &self.policies
    }

    /// Find the policy matching the given scope and memory type.
    ///
    /// If multiple policies match, the first one inserted wins.
    pub fn get_policy(&self, scope: Scope, memory_type: MemoryType) -> Option<&RetentionPolicy> {
        self.policies
            .iter()
            .find(|p| p.scope == scope && p.memory_type == memory_type)
    }

    /// Returns `true` if the memory item should be archived based on its
    /// creation time and the matching policy's `archive_after_days`.
    pub fn should_archive(&self, item: &MemoryItem, now: DateTime<Utc>) -> bool {
        let Some(policy) = self.get_policy(item.scope, item.memory_type) else {
            return false;
        };
        match policy.archive_after_days {
            Some(days) => {
                let threshold = item.created_at + chrono::Duration::days(days as i64);
                now >= threshold
            }
            None => false,
        }
    }

    /// Returns `true` if the memory item should be hard-deleted based on its
    /// creation time and the matching policy's `delete_after_days`.
    pub fn should_delete(&self, item: &MemoryItem, now: DateTime<Utc>) -> bool {
        let Some(policy) = self.get_policy(item.scope, item.memory_type) else {
            return false;
        };
        match policy.delete_after_days {
            Some(days) => {
                let threshold = item.created_at + chrono::Duration::days(days as i64);
                now >= threshold
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use oris_memory_store::memory_types::{
        AuthorityLevel, MemoryItem, MemoryStatus, PrivacyClass, SourceType,
    };
    use serde_json::Value;
    use uuid::Uuid;

    fn make_item(scope: Scope, mt: MemoryType, created: DateTime<Utc>) -> MemoryItem {
        MemoryItem {
            memory_id: Uuid::new_v4(),
            tenant_id: "acme".into(),
            memory_type: mt,
            scope,
            subject_type: None,
            subject_id: None,
            entity_refs: vec![],
            content: Some("test".into()),
            structured_payload: None,
            embedding: None,
            source_type: SourceType::UserExplicit,
            source_reference: None,
            evidence_refs: vec![],
            confidence: 0.5,
            authority_level: AuthorityLevel::L2Verified,
            importance: 0.5,
            observed_at: None,
            valid_from: None,
            valid_to: None,
            privacy_class: PrivacyClass::Internal,
            acl: Value::Null,
            retention_policy: None,
            status: MemoryStatus::Active,
            version: 1,
            derived_from: vec![],
            created_by_user: Some("alice".into()),
            created_by_agent: None,
            last_verified_at: None,
            created_at: created,
            updated_at: created,
        }
    }

    #[test]
    fn should_archive_after_policy_days() {
        let engine = PolicyEngine::new(vec![RetentionPolicy {
            policy_id: "p1".into(),
            scope: Scope::Team,
            memory_type: MemoryType::Semantic,
            ttl_days: None,
            archive_after_days: Some(30),
            delete_after_days: None,
        }]);
        let created = Utc::now() - chrono::Duration::days(31);
        let item = make_item(Scope::Team, MemoryType::Semantic, created);
        assert!(engine.should_archive(&item, Utc::now()));
    }

    #[test]
    fn should_not_archive_before_policy_days() {
        let engine = PolicyEngine::new(vec![RetentionPolicy {
            policy_id: "p1".into(),
            scope: Scope::Team,
            memory_type: MemoryType::Semantic,
            ttl_days: None,
            archive_after_days: Some(30),
            delete_after_days: None,
        }]);
        let created = Utc::now() - chrono::Duration::days(10);
        let item = make_item(Scope::Team, MemoryType::Semantic, created);
        assert!(!engine.should_archive(&item, Utc::now()));
    }

    #[test]
    fn should_delete_after_policy_days() {
        let engine = PolicyEngine::new(vec![RetentionPolicy {
            policy_id: "p1".into(),
            scope: Scope::Team,
            memory_type: MemoryType::Semantic,
            ttl_days: None,
            archive_after_days: None,
            delete_after_days: Some(90),
        }]);
        let created = Utc::now() - chrono::Duration::days(91);
        let item = make_item(Scope::Team, MemoryType::Semantic, created);
        assert!(engine.should_delete(&item, Utc::now()));
    }

    #[test]
    fn no_policy_means_no_action() {
        let engine = PolicyEngine::new(vec![]);
        let created = Utc::now() - chrono::Duration::days(365);
        let item = make_item(Scope::Personal, MemoryType::Episodic, created);
        assert!(!engine.should_archive(&item, Utc::now()));
        assert!(!engine.should_delete(&item, Utc::now()));
    }
}
