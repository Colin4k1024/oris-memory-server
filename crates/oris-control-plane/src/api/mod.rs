//! API layer for the Enterprise Context & Memory Service.
//!
//! Contains the §26 API-gateway middleware ([`gateway`]), the V1 REST
//! endpoint handlers ([`routes`]), and request/response DTOs ([`extractors`]).
//! Legacy experience-repository types remain in [`request`] and [`response`].

pub mod extractors;
pub mod gateway;
pub mod request;
pub mod response;
pub mod routes;
pub mod user_memory;

pub use extractors::{ApiError, RequestContext};
pub use gateway::{enforce_contract, ApiContract, GatewayError};
pub use request::FetchQuery;
pub use response::{ErrorResponse, FetchResponse, HealthResponse, SyncAudit};
pub use routes::{v1_router, AppState};
