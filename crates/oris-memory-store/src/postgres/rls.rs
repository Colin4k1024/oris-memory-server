//! Row Level Security management.
//!
//! RLS provides database-level tenant isolation as the last line of defense.
//! The application must set `app.tenant_id` via `set_config()` on each
//! connection before any query.

use sqlx::{PgPool, Row};

pub struct RlsManager;

impl RlsManager {
    /// Set the tenant context for the current connection (local transaction scope).
    pub async fn set_tenant(pool: &PgPool, tenant_id: &str) -> Result<(), sqlx::Error> {
        sqlx::query("SELECT set_config('app.tenant_id', $1, true)")
            .bind(tenant_id)
            .execute(pool)
            .await?;
        Ok(())
    }

    /// Reset the tenant context (clears RLS filter).
    pub async fn reset_tenant(pool: &PgPool) -> Result<(), sqlx::Error> {
        sqlx::query("SELECT set_config('app.tenant_id', '', true)")
            .execute(pool)
            .await?;
        Ok(())
    }

    /// Verify that RLS is enabled on the memory_item table.
    pub async fn is_enabled(pool: &PgPool) -> Result<bool, sqlx::Error> {
        let row = sqlx::query(
            r#"SELECT relrowsecurity FROM pg_class
               WHERE relname = 'memory_item'"#,
        )
        .fetch_optional(pool)
        .await?;

        Ok(row
            .and_then(|r| r.try_get::<bool, _>("relrowsecurity").ok())
            .unwrap_or(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rls_manager_exists() {
        // Compile-time check that the type is usable
        let _ = std::marker::PhantomData::<RlsManager>;
    }
}
