//! OEN Envelope verifier for validating signed envelopes.
//!
//! This module is the **canonical home** for the unified OEN Envelope type,
//! replacing the former dual `EvolutionEnvelope` / `OenEnvelope` definitions
//! that existed in `network_types.rs` and here respectively (issue #11).
//!
//! The cache has also been hardened (issue #12):
//! - TTL reduced from 300 s → 60 s
//! - Cache key now includes `public_key_hex` so key rotation invalidates old entries
//! - New `clear_cache_for_sender` method for revocation scenarios

use std::time::Duration;

use base64::Engine;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::error::OenError;
use oris_memory_contract::{CapsuleV1, GeneV1};

/// Type alias for an Ed25519 signature (hex or base64 string).
pub type Ed25519Signature = String;

// ──────────────────────────── MessageType ────────────────────────────

/// OEN message types — unified across the former `network_types::MessageType`
/// (Publish / Fetch / Report / Revoke) and `oen::MessageType` (Publish / Fetch /
/// Feedback) into a single enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    Publish,
    Fetch,
    Report,
    Revoke,
    Feedback,
}

impl std::fmt::Display for MessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageType::Publish => write!(f, "Publish"),
            MessageType::Fetch => write!(f, "Fetch"),
            MessageType::Report => write!(f, "Report"),
            MessageType::Revoke => write!(f, "Revoke"),
            MessageType::Feedback => write!(f, "Feedback"),
        }
    }
}

// ──────────────────────────── NetworkAsset ────────────────────────────

/// A typed asset carried inside an envelope. Wraps the canonical V1 contract
/// types (`GeneV1`, `CapsuleV1`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NetworkAsset {
    Gene { gene: GeneV1 },
    Capsule { capsule: CapsuleV1 },
}

// ──────────────────────────── EnvelopeManifest ────────────────────────

/// Manifest summarising the envelope contents for integrity verification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvelopeManifest {
    pub publisher: String,
    pub sender_id: String,
    pub asset_ids: Vec<String>,
    pub asset_hash: String,
}

// ──────────────────────────── OenEnvelope ────────────────────────────

/// Unified OEN Envelope — the **single canonical envelope structure** for the
/// Oris Evolution Network protocol.
///
/// Former `EvolutionEnvelope` fields (`protocol`, `protocol_version`,
/// `message_id`, `assets`, `content_hash`, `manifest`) are now part of this
/// struct. All of them use `#[serde(default)]` so that legacy JSON payloads
/// containing only the original five core fields continue to deserialize
/// without error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OenEnvelope {
    // ── Core fields (always present, required) ──
    /// Sender identifier (agent ID).
    pub sender_id: String,
    /// Message type.
    pub message_type: MessageType,
    /// The payload (Gene or Capsule data, or arbitrary JSON).
    pub payload: serde_json::Value,
    /// Ed25519 signature (base64 encoded).
    pub signature: String,
    /// Timestamp (RFC3339 format).
    pub timestamp: String,

    // ── Protocol metadata (from former EvolutionEnvelope, optional) ──
    #[serde(default)]
    pub protocol: String,
    #[serde(default)]
    pub protocol_version: String,
    #[serde(default)]
    pub message_id: String,

    // ── Typed assets (from former EvolutionEnvelope, optional) ──
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<NetworkAsset>,

    // ── Content hash (from former EvolutionEnvelope, optional) ──
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content_hash: String,

    // ── Manifest (from former EvolutionEnvelope, optional) ──
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest: Option<EnvelopeManifest>,
}

impl OenEnvelope {
    /// Create a minimal envelope with only the core fields.
    /// Protocol metadata fields default to empty values.
    pub fn minimal(
        sender_id: impl Into<String>,
        message_type: MessageType,
        payload: serde_json::Value,
        signature: impl Into<String>,
        timestamp: impl Into<String>,
    ) -> Self {
        Self {
            sender_id: sender_id.into(),
            message_type,
            payload,
            signature: signature.into(),
            timestamp: timestamp.into(),
            protocol: String::new(),
            protocol_version: String::new(),
            message_id: String::new(),
            assets: Vec::new(),
            content_hash: String::new(),
            manifest: None,
        }
    }

    /// Parse an envelope from JSON.
    pub fn from_json(json: &str) -> Result<Self, OenError> {
        serde_json::from_str(json).map_err(|e| OenError::ParseError(e.to_string()))
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> Result<String, OenError> {
        serde_json::to_string(self).map_err(|e| OenError::ParseError(e.to_string()))
    }
}

// ──────────────────────────── OenVerifier ────────────────────────────

/// Cache entry with verification timestamp.
struct CacheEntry {
    at: std::time::Instant,
}

impl Default for CacheEntry {
    fn default() -> Self {
        Self {
            at: std::time::Instant::now(),
        }
    }
}

/// OEN Envelope verifier with hardened caching support.
///
/// Security improvements (issue #12):
/// - Signature cache TTL reduced to **60 s** (was 300 s)
/// - Cache key includes `public_key_hex` so a rotated key invalidates old entries
/// - `clear_cache_for_sender` method for immediate invalidation on key revocation
#[derive(Clone)]
pub struct OenVerifier {
    /// Maximum age of envelope timestamp (in seconds).
    timestamp_tolerance_secs: i64,
    /// Signature cache TTL (in seconds).
    signature_ttl_secs: u64,
    /// Signature cache (`cache_key -> CacheEntry`).
    signature_cache:
        std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, CacheEntry>>>,
}

impl OenVerifier {
    /// Create a new verifier with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a verifier with custom timestamp tolerance.
    pub fn with_timestamp_tolerance(mut self, secs: i64) -> Self {
        self.timestamp_tolerance_secs = secs;
        self
    }

    /// Create a verifier with custom signature cache TTL.
    pub fn with_signature_ttl(mut self, secs: u64) -> Self {
        self.signature_ttl_secs = secs;
        self
    }

    /// Clear all cached signature verifications for a specific sender.
    ///
    /// Call this when a sender's public key is revoked or rotated to ensure
    /// no stale "verified" entries remain for that sender.
    pub async fn clear_cache_for_sender(&self, sender_id: &str) {
        let mut cache = self.signature_cache.lock().await;
        let prefix = format!("{sender_id}:");
        cache.retain(|key, _| !key.starts_with(&prefix));
    }

    /// Clear the entire signature cache.
    pub async fn clear_cache(&self) {
        let mut cache = self.signature_cache.lock().await;
        cache.clear();
    }

    /// Verify an OEN Envelope.
    ///
    /// This verifies:
    /// 1. The message type is Publish
    /// 2. The sender_id matches the expected agent
    /// 3. The timestamp is within the tolerance window
    /// 4. The Ed25519 signature is valid (with hardened cache)
    pub async fn verify_envelope(
        &self,
        envelope: &OenEnvelope,
        expected_agent_id: &str,
        public_key_hex: &str,
    ) -> Result<(), OenError> {
        // 1. Check message type
        if envelope.message_type != MessageType::Publish {
            return Err(OenError::InvalidMessageType {
                expected: MessageType::Publish.to_string(),
                actual: envelope.message_type.to_string(),
            });
        }

        // 2. Check sender_id matches
        if envelope.sender_id != expected_agent_id {
            return Err(OenError::SenderMismatch {
                expected: expected_agent_id.to_string(),
                actual: envelope.sender_id.clone(),
            });
        }

        // 3. Check timestamp
        let timestamp: DateTime<Utc> = envelope
            .timestamp
            .parse()
            .map_err(|_| OenError::ParseError("invalid timestamp format".to_string()))?;

        let now = Utc::now();
        let diff = (now - timestamp).num_seconds().abs();

        if diff > self.timestamp_tolerance_secs {
            return Err(OenError::TimestampExpired {
                seconds: diff,
                max: self.timestamp_tolerance_secs,
            });
        }

        // 4. Verify signature (with cache)
        // Cache key includes public_key_hex so that key rotation invalidates old entries.
        let cache_key = format!(
            "{}:{}:{}",
            envelope.sender_id, public_key_hex, envelope.signature
        );

        {
            let cache = self.signature_cache.lock().await;
            if let Some(entry) = cache.get(&cache_key) {
                if entry.at.elapsed() < Duration::from_secs(self.signature_ttl_secs) {
                    return Ok(());
                }
            }
        }

        // Verify the signature
        let payload_bytes = serde_json::to_vec(&envelope.payload)
            .map_err(|e| OenError::ParseError(e.to_string()))?;

        let signature_bytes =
            base64_decode(&envelope.signature).map_err(|_| OenError::SignatureFailed)?;

        use ed25519_dalek::{Signature, Verifier};

        let signature_bytes: [u8; 64] = signature_bytes
            .try_into()
            .map_err(|_| OenError::SignatureFailed)?;

        let signature = Signature::from_bytes(&signature_bytes);

        let public_key_bytes = hex::decode(public_key_hex)
            .map_err(|_| OenError::SigningError("invalid public key hex".to_string()))?;

        let public_key_bytes: [u8; 32] = public_key_bytes
            .try_into()
            .map_err(|_| OenError::SigningError("expected 32-byte public key".to_string()))?;

        let public_key = ed25519_dalek::VerifyingKey::from_bytes(&public_key_bytes)
            .map_err(|_| OenError::SigningError("invalid public key".to_string()))?;

        public_key
            .verify(&payload_bytes, &signature)
            .map_err(|_| OenError::SignatureFailed)?;

        // Cache the verified signature
        {
            let mut cache = self.signature_cache.lock().await;
            cache.insert(cache_key, CacheEntry::default());
        }

        Ok(())
    }
}

impl Default for OenVerifier {
    fn default() -> Self {
        Self {
            timestamp_tolerance_secs: 300, // 5 minutes
            signature_ttl_secs: 60,        // Reduced from 300 s to 60 s (issue #12)
            signature_cache: std::sync::Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }
}

/// Decode base64 string to bytes.
fn base64_decode(input: &str) -> Result<Vec<u8>, ()> {
    base64::engine::general_purpose::STANDARD
        .decode(input)
        .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_serialization() {
        let json = r#""publish""#;
        let msg_type: MessageType = serde_json::from_str(json).unwrap();
        assert_eq!(msg_type, MessageType::Publish);

        let back_to_json = serde_json::to_string(&msg_type).unwrap();
        assert_eq!(back_to_json, "\"publish\"");
    }

    #[test]
    fn test_all_message_types_serialize() {
        for mt in [
            MessageType::Publish,
            MessageType::Fetch,
            MessageType::Report,
            MessageType::Revoke,
            MessageType::Feedback,
        ] {
            let json = serde_json::to_string(&mt).unwrap();
            let back: MessageType = serde_json::from_str(&json).unwrap();
            assert_eq!(mt, back, "round-trip failed for {mt}");
        }
    }

    #[tokio::test]
    async fn test_envelope_parsing() {
        let json = r#"{
            "sender_id": "agent-123",
            "message_type": "publish",
            "payload": {"gene": {}},
            "signature": "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz01234567",
            "timestamp": "2026-04-14T10:00:00Z"
        }"#;

        let envelope = OenEnvelope::from_json(json).unwrap();
        assert_eq!(envelope.sender_id, "agent-123");
        assert_eq!(envelope.message_type, MessageType::Publish);
        // New fields should default to empty
        assert!(envelope.assets.is_empty());
        assert!(envelope.content_hash.is_empty());
        assert!(envelope.manifest.is_none());
    }

    #[test]
    fn test_envelope_backward_compat_with_protocol_fields() {
        // An envelope with all the former EvolutionEnvelope fields should parse
        let json = r#"{
            "sender_id": "agent-456",
            "message_type": "publish",
            "payload": {"gene": {"id": "g1"}},
            "signature": "sig",
            "timestamp": "2026-09-11T00:00:00Z",
            "protocol": "OEN",
            "protocol_version": "1.0",
            "message_id": "msg-001",
            "content_hash": "abc123"
        }"#;

        let envelope = OenEnvelope::from_json(json).unwrap();
        assert_eq!(envelope.protocol, "OEN");
        assert_eq!(envelope.protocol_version, "1.0");
        assert_eq!(envelope.message_id, "msg-001");
        assert_eq!(envelope.content_hash, "abc123");
    }

    #[tokio::test]
    async fn test_clear_cache_for_sender() {
        let verifier = OenVerifier::new();

        // Insert a fake cache entry
        {
            let mut cache = verifier.signature_cache.lock().await;
            cache.insert("agent-x:key1:sig1".to_string(), CacheEntry::default());
            cache.insert("agent-x:key2:sig2".to_string(), CacheEntry::default());
            cache.insert("agent-y:key3:sig3".to_string(), CacheEntry::default());
        }

        // Clear agent-x's entries
        verifier.clear_cache_for_sender("agent-x").await;

        {
            let cache = verifier.signature_cache.lock().await;
            assert!(!cache.contains_key("agent-x:key1:sig1"));
            assert!(!cache.contains_key("agent-x:key2:sig2"));
            assert!(
                cache.contains_key("agent-y:key3:sig3"),
                "agent-y should be untouched"
            );
        }
    }

    #[test]
    fn test_minimal_envelope_constructor() {
        let env = OenEnvelope::minimal(
            "agent-1",
            MessageType::Publish,
            serde_json::json!({"data": 1}),
            "sig",
            "2026-09-11T00:00:00Z",
        );
        assert_eq!(env.sender_id, "agent-1");
        assert!(env.assets.is_empty());
        assert!(env.protocol.is_empty());
    }
}
