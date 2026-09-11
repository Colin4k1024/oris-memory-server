//! Hot context materialization via Redis.
//!
//! Redis is a **non-authoritative** layer: all data can be lost and rebuilt
//! from PostgreSQL. This module caches the "hot context" — the canonical user
//! profile, active task contexts, and permission snapshots — so that the most
//! frequent read paths don't hit the database.
//!
//! Key layout:
//! - User context:  `oris:hotctx:{tenant}:{user}`
//! - Task context:   `oris:task:{task_id}`
//! - Permissions:    `oris:perm:{tenant}:{user}`
//!
//! Default TTL is 5 minutes (300 s). Callers may override per entry.

use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use redis::RedisError;

/// Default TTL for hot-context entries (5 minutes).
pub const DEFAULT_TTL_SECS: u64 = 300;

fn user_context_key(tenant_id: &str, user_id: &str) -> String {
    format!("oris:hotctx:{tenant_id}:{user_id}")
}

fn task_context_key(task_id: &str) -> String {
    format!("oris:task:{task_id}")
}

fn permissions_key(tenant_id: &str, user_id: &str) -> String {
    format!("oris:perm:{tenant_id}:{user_id}")
}

/// Redis-backed hot context repository.
///
/// Caches canonical user profiles, active task contexts, and permission
/// snapshots with short TTLs. All data is non-authoritative and can be
/// rebuilt from PostgreSQL on a cache miss.
pub struct HotContextRepo {
    conn: ConnectionManager,
}

impl HotContextRepo {
    /// Create a new `HotContextRepo` from a [`ConnectionManager`].
    pub fn new(conn: ConnectionManager) -> Self {
        Self { conn }
    }

    /// Fetch the materialized user context blob.
    ///
    /// Returns `Ok(None)` on cache miss; the caller should rebuild from
    /// PostgreSQL and call [`set_user_context`](Self::set_user_context).
    pub async fn get_user_context(
        &self,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<Option<Vec<u8>>, RedisError> {
        let key = user_context_key(tenant_id, user_id);
        let mut conn = self.conn.clone();
        conn.get::<_, Option<Vec<u8>>>(&key).await
    }

    /// Store the materialized user context blob with the given TTL.
    pub async fn set_user_context(
        &self,
        user_id: &str,
        tenant_id: &str,
        data: &[u8],
        ttl_secs: u64,
    ) -> Result<(), RedisError> {
        let key = user_context_key(tenant_id, user_id);
        let mut conn = self.conn.clone();
        conn.set_ex(&key, data, ttl_secs).await
    }

    /// Invalidate the cached user context.
    pub async fn invalidate_user_context(
        &self,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<(), RedisError> {
        let key = user_context_key(tenant_id, user_id);
        let mut conn = self.conn.clone();
        conn.del::<_, ()>(&key).await
    }

    /// Fetch the cached task context blob.
    pub async fn get_task_context(&self, task_id: &str) -> Result<Option<Vec<u8>>, RedisError> {
        let key = task_context_key(task_id);
        let mut conn = self.conn.clone();
        conn.get::<_, Option<Vec<u8>>>(&key).await
    }

    /// Store the task context blob with the given TTL.
    pub async fn set_task_context(
        &self,
        task_id: &str,
        data: &[u8],
        ttl_secs: u64,
    ) -> Result<(), RedisError> {
        let key = task_context_key(task_id);
        let mut conn = self.conn.clone();
        conn.set_ex(&key, data, ttl_secs).await
    }

    /// Invalidate the cached task context.
    pub async fn invalidate_task_context(&self, task_id: &str) -> Result<(), RedisError> {
        let key = task_context_key(task_id);
        let mut conn = self.conn.clone();
        conn.del::<_, ()>(&key).await
    }

    /// Fetch the cached permission snapshot.
    pub async fn get_permissions(
        &self,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<Option<Vec<u8>>, RedisError> {
        let key = permissions_key(tenant_id, user_id);
        let mut conn = self.conn.clone();
        conn.get::<_, Option<Vec<u8>>>(&key).await
    }

    /// Store the permission snapshot with the given TTL.
    pub async fn set_permissions(
        &self,
        user_id: &str,
        tenant_id: &str,
        data: &[u8],
        ttl_secs: u64,
    ) -> Result<(), RedisError> {
        let key = permissions_key(tenant_id, user_id);
        let mut conn = self.conn.clone();
        conn.set_ex(&key, data, ttl_secs).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ttl_is_5_minutes() {
        assert_eq!(DEFAULT_TTL_SECS, 300);
    }

    #[test]
    fn user_context_key_format() {
        assert_eq!(
            user_context_key("acme", "user-42"),
            "oris:hotctx:acme:user-42"
        );
    }

    #[test]
    fn task_context_key_format() {
        assert_eq!(task_context_key("task-99"), "oris:task:task-99");
    }

    #[test]
    fn permissions_key_format() {
        assert_eq!(permissions_key("acme", "user-42"), "oris:perm:acme:user-42");
    }
}
