//! OEN (Oris Evolution Network) Envelope handling module.
//!
//! This module is the **canonical home** for all OEN protocol types:
//! - `OenEnvelope` — the unified envelope structure (issue #11)
//! - `MessageType`, `NetworkAsset`, `EnvelopeManifest`, `Ed25519Signature`
//! - `OenVerifier` — with hardened cache (issue #12)
//! - `OenError`

mod error;
mod verifier;

pub use error::OenError;
pub use verifier::{
    Ed25519Signature, EnvelopeManifest, MessageType, NetworkAsset, OenEnvelope, OenVerifier,
};
