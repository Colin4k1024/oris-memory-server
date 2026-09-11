//! Inlined OEN (Oris Evolution Network) protocol types.
//!
//! These types were previously provided by the `oris-evolution-network` crate.
//! They are inlined here so that `oris-experience-repo` no longer depends on the
//! `oris-evolution` crate chain. The `NetworkAsset` enum wraps the canonical V1
//! contract types (`GeneV1`, `CapsuleV1`) instead of the legacy evolution types,
//! and the `EvolutionEvent` variant has been removed (it is unused by this
//! crate's core API).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use oris_experience_contract::{CapsuleV1, GeneV1};

pub type Ed25519Signature = String;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageType {
    Publish,
    Fetch,
    Report,
    Revoke,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NetworkAsset {
    Gene { gene: GeneV1 },
    Capsule { capsule: CapsuleV1 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvolutionEnvelope {
    pub protocol: String,
    pub protocol_version: String,
    pub message_type: MessageType,
    pub message_id: String,
    pub sender_id: String,
    pub timestamp: String,
    pub assets: Vec<NetworkAsset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest: Option<EnvelopeManifest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<Ed25519Signature>,
    pub content_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvelopeManifest {
    pub publisher: String,
    pub sender_id: String,
    pub asset_ids: Vec<String>,
    pub asset_hash: String,
}

/// Errors returned by [`NetworkPublisher::publish_envelope`].
#[derive(Debug, Error)]
pub enum NetworkPublishError {
    #[error("HTTP error: {0}")]
    Http(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("signing key not configured — cannot produce Ed25519 signature")]
    SigningKeyNotConfigured,
}

/// Abstraction for publishing an [`EvolutionEnvelope`] to a remote endpoint.
///
/// Implement this trait to inject a custom publish strategy. Failures should be
/// treated as non-fatal by callers — log a warning and continue without
/// aborting the promotion path.
#[async_trait]
pub trait NetworkPublisher: Send + Sync {
    /// Publish an evolution envelope to the remote network endpoint.
    async fn publish_envelope(
        &self,
        envelope: &EvolutionEnvelope,
    ) -> Result<(), NetworkPublishError>;
}
