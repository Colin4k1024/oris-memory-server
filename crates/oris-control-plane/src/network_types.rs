//! Network publishing types for the Oris Evolution Network.
//!
//! The envelope types (`OenEnvelope`, `MessageType`, `NetworkAsset`,
//! `EnvelopeManifest`, `Ed25519Signature`) have been **unified** into the
//! [`crate::oen`] module (issue #11). This file now only retains the
//! `NetworkPublisher` trait and `NetworkPublishError` error enum.

use async_trait::async_trait;
use thiserror::Error;

use crate::oen::OenEnvelope;

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

/// Abstraction for publishing an [`OenEnvelope`] to a remote endpoint.
///
/// Implement this trait to inject a custom publish strategy. Failures should be
/// treated as non-fatal by callers — log a warning and continue without
/// aborting the promotion path.
#[async_trait]
pub trait NetworkPublisher: Send + Sync {
    /// Publish an OEN envelope to the remote network endpoint.
    async fn publish_envelope(&self, envelope: &OenEnvelope) -> Result<(), NetworkPublishError>;
}
