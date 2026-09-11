//! API Gateway middleware — enforces the Memory API contract on every request.
//!
//! Every memory API request must carry tenant, user, agent, purpose, task,
//! and trace headers. This middleware extracts and validates those headers,
//! resolves the identity via [`IdentityResolver`], verifies the tenant match,
//! and injects the [`ResolvedIdentity`](crate::identity::ResolvedIdentity)
//! into request extensions for downstream handlers.
//!
//! See §5.1 + §10.2 of the architecture spec:
//!
//! > 所有调用必须传递 tenant、user、agent、purpose、task、trace，
//! > 不能只用技术服务账号绕过用户权限。
//!
//! # Contract
//!
//! | Header          | Required | Notes                                        |
//! |-----------------|----------|----------------------------------------------|
//! | `X-Tenant-ID`   | yes      | non-empty tenant identifier                  |
//! | `X-User-ID`     | yes      | non-empty user identifier                    |
//! | `X-Agent-ID`    | no       | `None` for direct user access                |
//! | `X-Purpose`     | yes      | e.g. `"memory_read"`, `"task_execution"`     |
//! | `X-Task-ID`     | no       | parsed as UUID; required for cross-agent ops |
//! | `X-Trace-ID`    | yes      | auto-generated (UUID v7) if missing          |
//! | `Authorization` | yes      | `Bearer <token>` forwarded to `IdentityResolver` |

use std::sync::Arc;

use axum::extract::{Request as AxumRequest, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;

use tracing::debug;

use uuid::Uuid;

use crate::identity::IdentityResolver;

// ──────────────────────────── Header Names ────────────────────────────

const HEADER_TENANT_ID: &str = "x-tenant-id";
const HEADER_USER_ID: &str = "x-user-id";
const HEADER_AGENT_ID: &str = "x-agent-id";
const HEADER_PURPOSE: &str = "x-purpose";
const HEADER_TASK_ID: &str = "x-task-id";
const HEADER_TRACE_ID: &str = "x-trace-id";
const HEADER_AUTHORIZATION: &str = "authorization";

// ──────────────────────────── ApiContract ────────────────────────────

/// Extracted API contract from request headers.
///
/// Built by [`ApiContract::from_headers`]. After a successful
/// [`validate`](Self::validate) all required fields are guaranteed non-empty.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ApiContract {
    /// `X-Tenant-ID` — tenant identifier (required, non-empty).
    pub tenant_id: String,
    /// `X-User-ID` — user identifier (required, non-empty).
    pub user_id: String,
    /// `X-Agent-ID` — agent making the request (optional).
    pub agent_id: Option<String>,
    /// `X-Purpose` — stated purpose, e.g. `"memory_read"`, `"task_execution"`.
    pub purpose: String,
    /// `X-Task-ID` — task context (optional, parsed as UUID).
    pub task_id: Option<Uuid>,
    /// `X-Trace-ID` — trace identifier (auto-generated if missing).
    pub trace_id: String,
    /// `Authorization: Bearer <token>` — token for identity resolution.
    pub token: String,
}

impl ApiContract {
    /// Extract the contract from HTTP request headers.
    ///
    /// Required headers that are absent yield [`GatewayError::MissingHeader`];
    /// required headers present but empty or whitespace-only yield
    /// [`GatewayError::EmptyHeader`]. If `X-Trace-ID` is missing or blank a
    /// UUID v7 is generated automatically.
    pub fn from_headers(headers: &axum::http::HeaderMap) -> Result<Self, GatewayError> {
        // Helper: fetch a required header as a trimmed, non-empty String.
        let require = |name: &'static str| -> Result<String, GatewayError> {
            match headers.get(name) {
                None => Err(GatewayError::MissingHeader(name)),
                Some(val) => {
                    let raw = val.to_str().map_err(|_| GatewayError::EmptyHeader(name))?;
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        Err(GatewayError::EmptyHeader(name))
                    } else {
                        Ok(trimmed.to_string())
                    }
                }
            }
        };

        let tenant_id = require(HEADER_TENANT_ID)?;
        let user_id = require(HEADER_USER_ID)?;
        let purpose = require(HEADER_PURPOSE)?;

        // Agent ID is optional — empty/whitespace values become None.
        let agent_id = headers
            .get(HEADER_AGENT_ID)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string);

        // Task ID is optional; parse to Uuid when present (ignore malformed).
        let task_id = headers
            .get(HEADER_TASK_ID)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<Uuid>().ok());

        // Trace ID is required — auto-generate a UUID v7 if missing/blank.
        let trace_id = match headers.get(HEADER_TRACE_ID) {
            Some(val) => {
                let raw = val.to_str().unwrap_or("").trim();
                if raw.is_empty() {
                    Uuid::now_v7().to_string()
                } else {
                    raw.to_string()
                }
            }
            None => Uuid::now_v7().to_string(),
        };

        // Authorization: Bearer <token> (strip prefix, accept raw token).
        let token = match headers.get(HEADER_AUTHORIZATION) {
            None => return Err(GatewayError::MissingHeader("Authorization")),
            Some(val) => {
                let raw = val
                    .to_str()
                    .map_err(|_| GatewayError::EmptyHeader("Authorization"))?;
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    return Err(GatewayError::EmptyHeader("Authorization"));
                }
                let after_scheme = trimmed
                    .strip_prefix("Bearer ")
                    .or_else(|| trimmed.strip_prefix("bearer "))
                    .unwrap_or(trimmed);
                let token = after_scheme.trim();
                if token.is_empty() {
                    return Err(GatewayError::EmptyHeader("Authorization"));
                }
                token.to_string()
            }
        };

        Ok(Self {
            tenant_id,
            user_id,
            agent_id,
            purpose,
            task_id,
            trace_id,
            token,
        })
    }

    /// Validate that all required fields are present and non-empty.
    ///
    /// Defensive double-check after [`from_headers`](Self::from_headers).
    /// Returns `Ok(())` if the contract is well-formed.
    pub fn validate(&self) -> Result<(), GatewayError> {
        if self.tenant_id.is_empty() {
            return Err(GatewayError::EmptyHeader(HEADER_TENANT_ID));
        }
        if self.user_id.is_empty() {
            return Err(GatewayError::EmptyHeader(HEADER_USER_ID));
        }
        if self.purpose.is_empty() {
            return Err(GatewayError::EmptyHeader(HEADER_PURPOSE));
        }
        if self.token.is_empty() {
            return Err(GatewayError::EmptyHeader("Authorization"));
        }
        Ok(())
    }
}

// ──────────────────────────── GatewayError ────────────────────────────

/// Errors returned by the gateway middleware.
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    /// A required header was absent from the request.
    #[error("missing required header: {0}")]
    MissingHeader(&'static str),
    /// A required header was present but empty or whitespace-only.
    #[error("empty header value: {0}")]
    EmptyHeader(&'static str),
    /// Identity resolution via [`IdentityResolver`] failed.
    #[error("identity resolution failed: {0}")]
    IdentityResolution(String),
    /// The resolved identity does not match the requested tenant.
    #[error("unauthorized: {0}")]
    Unauthorized(String),
}

impl GatewayError {
    /// Map the error to the appropriate HTTP status code.
    ///
    /// Missing/empty headers → `400 Bad Request`.
    /// Identity/auth failures → `401 Unauthorized`.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::MissingHeader(_) | Self::EmptyHeader(_) => StatusCode::BAD_REQUEST,
            Self::IdentityResolution(_) | Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
        }
    }
}

// ──────────────────────────── Middleware ────────────────────────────

/// Axum middleware that enforces the API contract on every request.
///
/// # Steps
///
/// 1. Extract [`ApiContract`] from request headers.
/// 2. Validate required fields.
/// 3. Resolve identity via [`IdentityResolver::resolve`].
/// 4. Verify the resolved tenant matches `X-Tenant-ID`.
/// 5. Inject [`ResolvedIdentity`](crate::identity::ResolvedIdentity) and
///    [`ApiContract`] into request extensions.
/// 6. Call the next handler.
///
/// On any error the middleware short-circuits with the appropriate HTTP
/// status code and a human-readable error message.
pub async fn enforce_contract(
    State(resolver): State<Arc<IdentityResolver>>,
    mut request: AxumRequest,
    next: Next,
) -> Result<Response, (StatusCode, String)> {
    // 1. Extract contract from headers.
    let contract = match ApiContract::from_headers(request.headers()) {
        Ok(c) => c,
        Err(e) => {
            debug!(error = %e, "gateway: contract extraction failed");
            return Err((e.status_code(), e.to_string()));
        }
    };

    // 2. Validate required fields.
    if let Err(e) = contract.validate() {
        debug!(error = %e, "gateway: contract validation failed");
        return Err((e.status_code(), e.to_string()));
    }

    // 3. Resolve identity.
    let resolved = match resolver
        .resolve(
            &contract.token,
            contract.agent_id.as_deref(),
            &contract.purpose,
        )
        .await
    {
        Ok(id) => id,
        Err(e) => {
            let gw_err = GatewayError::IdentityResolution(e.to_string());
            debug!(error = %gw_err, "gateway: identity resolution failed");
            return Err((gw_err.status_code(), gw_err.to_string()));
        }
    };

    // 4. Verify tenant matches.
    if resolved.organization_id != contract.tenant_id {
        let gw_err = GatewayError::Unauthorized(format!(
            "tenant mismatch: header '{}' vs resolved '{}'",
            contract.tenant_id, resolved.organization_id
        ));
        debug!(error = %gw_err, "gateway: tenant mismatch");
        return Err((gw_err.status_code(), gw_err.to_string()));
    }

    debug!(
        trace_id = %contract.trace_id,
        tenant = %contract.tenant_id,
        user = %contract.user_id,
        purpose = %contract.purpose,
        "gateway: contract enforced, forwarding request",
    );

    // 5. Inject resolved identity + contract into extensions.
    request.extensions_mut().insert(resolved);
    request.extensions_mut().insert(contract);

    // 6. Call next handler.
    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;

    // ── Test helpers ──

    /// Build a header set with all required fields populated.
    fn valid_headers() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-tenant-id", "acme-corp".parse().unwrap());
        h.insert("x-user-id", "user-42".parse().unwrap());
        h.insert("x-purpose", "memory_read".parse().unwrap());
        h.insert("x-trace-id", "trace-abc-123".parse().unwrap());
        h.insert("authorization", "Bearer secret-token".parse().unwrap());
        h
    }

    // ── Header extraction ──

    #[test]
    fn extracts_all_required_headers() {
        let contract = ApiContract::from_headers(&valid_headers()).unwrap();
        assert_eq!(contract.tenant_id, "acme-corp");
        assert_eq!(contract.user_id, "user-42");
        assert_eq!(contract.purpose, "memory_read");
        assert_eq!(contract.trace_id, "trace-abc-123");
        assert_eq!(contract.token, "secret-token");
        assert!(contract.agent_id.is_none());
        assert!(contract.task_id.is_none());
    }

    #[test]
    fn extracts_optional_agent_and_task_id() {
        let mut h = valid_headers();
        h.insert("x-agent-id", "agent-007".parse().unwrap());
        h.insert(
            "x-task-id",
            "550e8400-e29b-41d4-a716-446655440000".parse().unwrap(),
        );
        let contract = ApiContract::from_headers(&h).unwrap();
        assert_eq!(contract.agent_id.as_deref(), Some("agent-007"));
        assert_eq!(
            contract.task_id.unwrap().to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
    }

    // ── Validation ──

    #[test]
    fn validates_complete_contract() {
        let contract = ApiContract::from_headers(&valid_headers()).unwrap();
        assert!(contract.validate().is_ok());
    }

    // ── Auto trace_id ──

    #[test]
    fn auto_generates_trace_id_when_missing() {
        let mut h = valid_headers();
        h.remove("x-trace-id");
        let contract = ApiContract::from_headers(&h).unwrap();
        assert!(!contract.trace_id.is_empty());
        assert!(Uuid::parse_str(&contract.trace_id).is_ok());
    }

    #[test]
    fn auto_generates_trace_id_when_blank() {
        let mut h = valid_headers();
        h.insert("x-trace-id", "   ".parse().unwrap());
        let contract = ApiContract::from_headers(&h).unwrap();
        assert!(!contract.trace_id.is_empty());
        assert_ne!(contract.trace_id, "   ");
    }

    #[test]
    fn auto_generated_trace_ids_are_unique() {
        let mut h = valid_headers();
        h.remove("x-trace-id");
        let c1 = ApiContract::from_headers(&h).unwrap();
        let c2 = ApiContract::from_headers(&h).unwrap();
        assert_ne!(c1.trace_id, c2.trace_id);
    }

    // ── Error cases: missing headers ──

    #[test]
    fn missing_tenant_id_returns_missing_header_error() {
        let mut h = valid_headers();
        h.remove("x-tenant-id");
        let err = ApiContract::from_headers(&h).unwrap_err();
        assert!(matches!(err, GatewayError::MissingHeader("x-tenant-id")));
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn missing_user_id_returns_missing_header_error() {
        let mut h = valid_headers();
        h.remove("x-user-id");
        let err = ApiContract::from_headers(&h).unwrap_err();
        assert!(matches!(err, GatewayError::MissingHeader("x-user-id")));
    }

    #[test]
    fn missing_purpose_returns_missing_header_error() {
        let mut h = valid_headers();
        h.remove("x-purpose");
        let err = ApiContract::from_headers(&h).unwrap_err();
        assert!(matches!(err, GatewayError::MissingHeader("x-purpose")));
    }

    #[test]
    fn missing_authorization_returns_missing_header_error() {
        let mut h = valid_headers();
        h.remove("authorization");
        let err = ApiContract::from_headers(&h).unwrap_err();
        assert!(matches!(err, GatewayError::MissingHeader("Authorization")));
    }

    // ── Error cases: empty headers ──

    #[test]
    fn empty_tenant_id_returns_empty_header_error() {
        let mut h = valid_headers();
        h.insert("x-tenant-id", "  ".parse().unwrap());
        let err = ApiContract::from_headers(&h).unwrap_err();
        assert!(matches!(err, GatewayError::EmptyHeader("x-tenant-id")));
    }

    #[test]
    fn empty_user_id_returns_empty_header_error() {
        let mut h = valid_headers();
        h.insert("x-user-id", "".parse().unwrap());
        let err = ApiContract::from_headers(&h).unwrap_err();
        assert!(matches!(err, GatewayError::EmptyHeader("x-user-id")));
    }

    // ── Bearer token extraction ──

    #[test]
    fn strips_bearer_prefix() {
        let mut h = valid_headers();
        h.insert("authorization", "Bearer my-token".parse().unwrap());
        let contract = ApiContract::from_headers(&h).unwrap();
        assert_eq!(contract.token, "my-token");
    }

    #[test]
    fn strips_bearer_prefix_case_insensitive() {
        let mut h = valid_headers();
        h.insert("authorization", "bearer lower-token".parse().unwrap());
        let contract = ApiContract::from_headers(&h).unwrap();
        assert_eq!(contract.token, "lower-token");
    }

    #[test]
    fn accepts_raw_token_without_bearer_prefix() {
        let mut h = valid_headers();
        h.insert("authorization", "raw-token".parse().unwrap());
        let contract = ApiContract::from_headers(&h).unwrap();
        assert_eq!(contract.token, "raw-token");
    }

    // ── Task ID / Agent ID edge cases ──

    #[test]
    fn ignores_malformed_task_id() {
        let mut h = valid_headers();
        h.insert("x-task-id", "not-a-uuid".parse().unwrap());
        let contract = ApiContract::from_headers(&h).unwrap();
        assert!(contract.task_id.is_none());
    }

    #[test]
    fn empty_agent_id_treated_as_none() {
        let mut h = valid_headers();
        h.insert("x-agent-id", "  ".parse().unwrap());
        let contract = ApiContract::from_headers(&h).unwrap();
        assert!(contract.agent_id.is_none());
    }

    // ── Status codes ──

    #[test]
    fn status_codes_are_correct() {
        assert_eq!(
            GatewayError::MissingHeader("x").status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            GatewayError::EmptyHeader("x").status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            GatewayError::IdentityResolution("x".to_string()).status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            GatewayError::Unauthorized("x".to_string()).status_code(),
            StatusCode::UNAUTHORIZED
        );
    }

    // ── Middleware integration tests ──

    use axum::body::Body;
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    use crate::identity::NoopIamClient;

    fn test_resolver() -> Arc<IdentityResolver> {
        Arc::new(IdentityResolver::new(Arc::new(NoopIamClient::new(
            "acme-corp",
        ))))
    }

    fn test_router(resolver: Arc<IdentityResolver>) -> Router {
        Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(from_fn_with_state(resolver, enforce_contract))
    }

    #[tokio::test]
    async fn middleware_passes_valid_request() {
        let app = test_router(test_resolver());
        let request = axum::http::Request::builder()
            .uri("/test")
            .header("x-tenant-id", "acme-corp")
            .header("x-user-id", "user-42")
            .header("x-purpose", "memory_read")
            .header("authorization", "Bearer user-42")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn middleware_rejects_missing_headers() {
        let app = test_router(test_resolver());
        let request = axum::http::Request::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn middleware_rejects_tenant_mismatch() {
        let app = test_router(test_resolver());
        let request = axum::http::Request::builder()
            .uri("/test")
            .header("x-tenant-id", "wrong-tenant")
            .header("x-user-id", "user-42")
            .header("x-purpose", "memory_read")
            .header("authorization", "Bearer user-42")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
