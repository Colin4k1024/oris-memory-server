//! oris-experience-repo
//!
//! HTTP API server for Oris Experience Repository.
//!
//! Provides a REST API for external agents to query and contribute experiences
//! (genes and capsules) to the Oris experience pool.

pub mod adapters;
pub mod api;
pub mod canonical_user;
pub mod client;
pub mod compliance;
pub mod conflict_resolver;
pub mod context_assembler;
pub mod context_package;
pub mod context_router;
pub mod control_plane;
pub mod decision;
pub mod degradation;
pub mod embedding;
pub mod encryption;
pub mod engine;
pub mod entity;
pub mod error;
pub mod eval;
pub mod governance;
pub mod identity;
pub mod key_service;
pub mod mcp;
pub mod mcp_tools;
pub mod memory_control_plane;
pub mod migration;
pub mod network_types;
pub mod oen;
pub mod outbox_worker;
pub mod poison_guard;
pub mod pre_filter;
pub mod rerank;
pub mod resilience;
pub mod server;
pub mod shared_task;
pub mod skill_projection;
pub mod slo_monitoring;
pub mod vector_search;
pub mod write_pipeline;

pub use client::ExperienceRepoClient;
pub use control_plane::{ExperienceControlPlane, ExperienceSearchQuery, ExperienceSearchResult};
pub use engine::{
    CircuitBreaker, EngineError, EngineHealth, EngineQuery, EngineRegistry, EngineResult,
    EngineWriteItem, MemoryEngine, NoopEngine,
};
pub use error::ExperienceRepoError;
pub use key_service::{KeyServiceError, KeyStore};
pub use network_types::{NetworkPublishError, NetworkPublisher};
pub use oen::{
    Ed25519Signature, EnvelopeManifest, MessageType, NetworkAsset, OenEnvelope, OenError,
    OenVerifier,
};
pub use oris_memory_contract::{CapsuleV1, ExperienceBundleV1, GeneV1, UsageReceiptV1};
pub use poison_guard::{PoisonGuard, SafetyVerdict, ScanReport, SourceType};
pub use server::{ExperienceRepoServer, ServerConfig};
pub mod red_team_tests;
