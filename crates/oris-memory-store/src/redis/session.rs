//! Session state and distributed locks via Redis.
//!
//! Key layout:
//! - Lock:     `oris:lock:{resource}`
//! - Session:  `oris:session:{session_id}`
//!
//! Locks use the `SET NX EX` pattern for acquisition and an atomic
//! Lua compare-and-delete for release, preventing a holder from deleting
//! a lock it no longer owns.

use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use redis::RedisError;

fn lock_key(resource: &str) -> String {
    format!("oris:lock:{resource}")
}

fn session_key(session_id: &str) -> String {
    format!("oris:session:{session_id}")
}

/// Redis-backed session state and distributed lock repository.
pub struct SessionRepo {
    conn: ConnectionManager,
}

impl SessionRepo {
    /// Create a new `SessionRepo` from a [`ConnectionManager`].
    pub fn new(conn: ConnectionManager) -> Self {
        Self { conn }
    }

    /// Attempt to acquire a distributed lock.
    ///
    /// Uses `SET resource holder NX EX ttl`. Returns `true` if the lock was
    /// acquired, `false` if another holder already owns it.
    pub async fn acquire_lock(
        &self,
        resource: &str,
        holder: &str,
        ttl_secs: u64,
    ) -> Result<bool, RedisError> {
        let key = lock_key(resource);
        let mut conn = self.conn.clone();
        // SET key holder NX EX ttl — returns "OK" on success, nil if key exists.
        let result: Option<String> = redis::cmd("SET")
            .arg(&key)
            .arg(holder)
            .arg("NX")
            .arg("EX")
            .arg(ttl_secs)
            .query_async(&mut conn)
            .await?;
        Ok(result.is_some())
    }

    /// Release a distributed lock.
    ///
    /// Atomically checks that the current holder matches before deleting the
    /// key via a Lua script. Returns `true` if the lock was released,
    /// `false` if the caller did not own it (expired or stolen).
    pub async fn release_lock(&self, resource: &str, holder: &str) -> Result<bool, RedisError> {
        let key = lock_key(resource);
        let script = r#"
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("DEL", KEYS[1])
            else
                return 0
            end
        "#;
        let mut conn = self.conn.clone();
        let result: i32 = redis::cmd("EVAL")
            .arg(script)
            .arg(1) // number of keys
            .arg(&key)
            .arg(holder)
            .query_async(&mut conn)
            .await?;
        Ok(result == 1)
    }

    /// Fetch session state by session ID.
    ///
    /// Returns `Ok(None)` when the session does not exist or has expired.
    pub async fn get_session(&self, session_id: &str) -> Result<Option<Vec<u8>>, RedisError> {
        let key = session_key(session_id);
        let mut conn = self.conn.clone();
        conn.get::<_, Option<Vec<u8>>>(&key).await
    }

    /// Store session state with the given TTL.
    pub async fn set_session(
        &self,
        session_id: &str,
        data: &[u8],
        ttl_secs: u64,
    ) -> Result<(), RedisError> {
        let key = session_key(session_id);
        let mut conn = self.conn.clone();
        conn.set_ex(&key, data, ttl_secs).await
    }

    /// Delete a session.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), RedisError> {
        let key = session_key(session_id);
        let mut conn = self.conn.clone();
        conn.del::<_, ()>(&key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_key_format() {
        assert_eq!(lock_key("resource-1"), "oris:lock:resource-1");
    }

    #[test]
    fn session_key_format() {
        assert_eq!(session_key("sess-abc"), "oris:session:sess-abc");
    }

    #[test]
    fn lock_key_with_nested_path() {
        assert_eq!(lock_key("task:42:embedding"), "oris:lock:task:42:embedding");
    }
}
