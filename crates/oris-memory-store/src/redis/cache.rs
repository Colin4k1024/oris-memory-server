//! Result caching via Redis.
//!
//! Generic key-value cache for search results and expensive computations.
//! All keys are namespaced under the `oris:cache:` prefix so that
//! [`CacheRepo::invalidate_pattern`] only touches cache entries.
//!
//! Uses `SCAN` (not `KEYS`) for pattern invalidation to avoid blocking the
//! Redis server.

use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use redis::RedisError;

/// Prefix applied to every cache key.
const CACHE_PREFIX: &str = "oris:cache";

fn prefixed_key(key: &str) -> String {
    format!("{CACHE_PREFIX}:{key}")
}

/// Redis-backed result cache.
///
/// Stores opaque byte blobs under the `oris:cache:` namespace with
/// caller-specified TTLs. Pattern-based invalidation uses `SCAN` to avoid
/// blocking Redis.
pub struct CacheRepo {
    conn: ConnectionManager,
}

impl CacheRepo {
    /// Create a new `CacheRepo` from a [`ConnectionManager`].
    pub fn new(conn: ConnectionManager) -> Self {
        Self { conn }
    }

    /// Fetch a cached value by key.
    ///
    /// Returns `Ok(None)` on cache miss.
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, RedisError> {
        let redis_key = prefixed_key(key);
        let mut conn = self.conn.clone();
        conn.get::<_, Option<Vec<u8>>>(&redis_key).await
    }

    /// Store a value with the given TTL (seconds).
    pub async fn set(&self, key: &str, value: &[u8], ttl_secs: u64) -> Result<(), RedisError> {
        let redis_key = prefixed_key(key);
        let mut conn = self.conn.clone();
        conn.set_ex(&redis_key, value, ttl_secs).await
    }

    /// Delete a single cache entry.
    pub async fn delete(&self, key: &str) -> Result<(), RedisError> {
        let redis_key = prefixed_key(key);
        let mut conn = self.conn.clone();
        conn.del::<_, ()>(&redis_key).await
    }

    /// Invalidate all cache entries matching a glob-style pattern.
    ///
    /// Uses `SCAN` to avoid blocking Redis. The pattern is automatically
    /// prefixed with `oris:cache:`, so callers should pass the suffix
    /// pattern only (e.g. `"search:*"`).
    pub async fn invalidate_pattern(&self, pattern: &str) -> Result<(), RedisError> {
        let full_pattern = prefixed_key(pattern);
        let mut conn = self.conn.clone();
        let mut cursor: u64 = 0;
        loop {
            let (next_cursor, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&full_pattern)
                .arg("COUNT")
                .arg(100u64)
                .query_async(&mut conn)
                .await?;
            if !batch.is_empty() {
                conn.del::<_, ()>(&batch[..]).await?;
            }
            if next_cursor == 0 {
                break;
            }
            cursor = next_cursor;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixed_key_adds_namespace() {
        assert_eq!(prefixed_key("foo"), "oris:cache:foo");
    }

    #[test]
    fn prefixed_key_empty_string() {
        assert_eq!(prefixed_key(""), "oris:cache:");
    }

    #[test]
    fn prefixed_key_with_segments() {
        assert_eq!(
            prefixed_key("search:tenant:abc"),
            "oris:cache:search:tenant:abc"
        );
    }
}
