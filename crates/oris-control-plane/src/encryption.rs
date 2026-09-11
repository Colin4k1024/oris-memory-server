//! Data encryption & key management module.
//!
//! Provides column-level encryption for sensitive memory fields, centralized
//! key management with rotation support, and credential detection to prevent
//! secrets from entering the Memory store.
//!
//! # Architecture
//!
//! - [`EncryptionManager`] orchestrates encrypt/decrypt using a [`KeyProvider`]
//!   and a pluggable [`Cipher`].
//! - [`KeyProvider`] abstracts where keys live (in-memory, KMS, vault, …).
//! - [`Cipher`] abstracts the algorithm. A [`DummyCipher`] (XOR) is provided
//!   for testing; production should use AES-256-GCM.
//! - [`CredentialDetector`] scans text for credential patterns and works with
//!   [`PoisonGuard`](crate::poison_guard::PoisonGuard) to block credential
//!   injection into Memory.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use async_trait::async_trait;
use base64::{engine::general_purpose, Engine as _};
use oris_memory_contract::PrivacyClass;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

// ──────────────────────────── Errors ────────────────────────────

#[derive(Debug, Clone, Error)]
pub enum KeyError {
    #[error("key not found for version {0}")]
    NotFound(u32),
    #[error("key provider error: {0}")]
    Provider(String),
    #[error("decryption failed: {0}")]
    DecryptionFailed(String),
    #[error("encryption failed: {0}")]
    EncryptionFailed(String),
}

// ──────────────────────────── Encryption Key ────────────────────────────

/// A versioned AES-256 key (32 bytes of material).
#[derive(Debug, Clone)]
pub struct EncryptionKey {
    /// Monotonic version number for key rotation tracking.
    pub version: u32,
    /// Raw key material (256-bit for AES-256).
    pub material: Vec<u8>,
}

// ──────────────────────────── Key Provider ────────────────────────────

/// Abstracts the source of encryption keys (in-memory, KMS, vault, …).
#[async_trait]
pub trait KeyProvider: Send + Sync {
    /// Get the current (latest) encryption key.
    async fn get_current_key(&self) -> Result<EncryptionKey, KeyError>;
    /// Get a key by version (for decrypting data encrypted with an older key).
    async fn get_key_by_version(&self, version: u32) -> Result<EncryptionKey, KeyError>;
    /// Rotate to a new key, returning the new version number.
    async fn rotate_key(&self) -> Result<u32, KeyError>;
}

// ──────────────────────────── Cipher Trait ────────────────────────────

/// Abstraction over the encryption algorithm.
///
/// Implementations must be symmetric: `decrypt(encrypt(pt, k), k) == pt`.
pub trait Cipher: Send + Sync {
    /// Encrypt `plaintext` using `key` material.
    fn encrypt(&self, plaintext: &[u8], key: &[u8]) -> Result<Vec<u8>, KeyError>;
    /// Decrypt `ciphertext` using `key` material.
    fn decrypt(&self, ciphertext: &[u8], key: &[u8]) -> Result<Vec<u8>, KeyError>;
}

/// XOR-based cipher for testing and development. **NOT for production use.**
///
/// Each byte of plaintext is XORed with the corresponding byte of the key
/// (cycling through the key). This makes it deterministic but sufficient
/// for round-trip testing without pulling in a full crypto crate.
pub struct DummyCipher;

impl Cipher for DummyCipher {
    fn encrypt(&self, plaintext: &[u8], key: &[u8]) -> Result<Vec<u8>, KeyError> {
        if key.is_empty() {
            return Err(KeyError::EncryptionFailed("empty key material".into()));
        }
        Ok(plaintext
            .iter()
            .enumerate()
            .map(|(i, &b)| b ^ key[i % key.len()])
            .collect())
    }

    fn decrypt(&self, ciphertext: &[u8], key: &[u8]) -> Result<Vec<u8>, KeyError> {
        if key.is_empty() {
            return Err(KeyError::DecryptionFailed("empty key material".into()));
        }
        Ok(ciphertext
            .iter()
            .enumerate()
            .map(|(i, &b)| b ^ key[i % key.len()])
            .collect())
    }
}

// ──────────────────────────── Encryption Manager ────────────────────────────

/// Column-level encryption for sensitive memory fields.
pub struct EncryptionManager {
    key_provider: Arc<dyn KeyProvider>,
    cipher: Box<dyn Cipher>,
}

impl EncryptionManager {
    /// Create with the default [`DummyCipher`].
    pub fn new(key_provider: Arc<dyn KeyProvider>) -> Self {
        Self {
            key_provider,
            cipher: Box::new(DummyCipher),
        }
    }

    /// Create with a custom [`Cipher`] implementation.
    pub fn with_cipher(key_provider: Arc<dyn KeyProvider>, cipher: Box<dyn Cipher>) -> Self {
        Self {
            key_provider,
            cipher,
        }
    }

    /// Encrypt a plaintext string.
    ///
    /// Returns `(base64_ciphertext, key_version)`. The key version must be
    /// persisted alongside the ciphertext so the correct key can be used for
    /// decryption after rotation.
    pub async fn encrypt(&self, plaintext: &str) -> Result<(String, u32), KeyError> {
        let key = self.key_provider.get_current_key().await?;
        let encrypted = self.cipher.encrypt(plaintext.as_bytes(), &key.material)?;
        let encoded = general_purpose::STANDARD.encode(&encrypted);
        Ok((encoded, key.version))
    }

    /// Decrypt a ciphertext that was encrypted with a specific key version.
    pub async fn decrypt(&self, ciphertext: &str, key_version: u32) -> Result<String, KeyError> {
        let key = self.key_provider.get_key_by_version(key_version).await?;
        let decoded = general_purpose::STANDARD
            .decode(ciphertext)
            .map_err(|e| KeyError::DecryptionFailed(e.to_string()))?;
        let plaintext = self.cipher.decrypt(&decoded, &key.material)?;
        String::from_utf8(plaintext)
            .map_err(|e| KeyError::DecryptionFailed(format!("invalid utf-8: {e}")))
    }

    /// Check if a field should be encrypted based on its [`PrivacyClass`].
    ///
    /// `Restricted` and `Confidential` data must be encrypted at rest.
    pub fn should_encrypt(privacy: &PrivacyClass) -> bool {
        matches!(
            privacy,
            PrivacyClass::Restricted | PrivacyClass::Confidential
        )
    }

    /// Rotate the encryption key via the key provider.
    pub async fn rotate_key(&self) -> Result<u32, KeyError> {
        self.key_provider.rotate_key().await
    }
}

// ──────────────────────────── In-Memory Key Provider ────────────────────────────

/// Simple in-memory key provider for testing and development.
///
/// Keys are deterministically derived from the version number using SHA-256,
/// so tests are reproducible without a source of randomness.
pub struct InMemoryKeyProvider {
    keys: RwLock<HashMap<u32, EncryptionKey>>,
    current_version: RwLock<u32>,
}

impl InMemoryKeyProvider {
    /// Create a provider with a single initial key (version 1).
    pub fn new() -> Self {
        let mut keys = HashMap::new();
        let version = 1;
        keys.insert(version, Self::generate_key(version));
        Self {
            keys: RwLock::new(keys),
            current_version: RwLock::new(version),
        }
    }

    /// Derive 32 bytes of key material from a version number.
    fn generate_key(version: u32) -> EncryptionKey {
        let mut hasher = Sha256::new();
        hasher.update(b"oris-encryption-key-v");
        hasher.update(version.to_le_bytes());
        let material = hasher.finalize().to_vec();
        EncryptionKey { version, material }
    }
}

impl Default for InMemoryKeyProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl KeyProvider for InMemoryKeyProvider {
    async fn get_current_key(&self) -> Result<EncryptionKey, KeyError> {
        let v = {
            let guard = self
                .current_version
                .read()
                .map_err(|e| KeyError::Provider(e.to_string()))?;
            *guard
        };
        self.get_key_by_version(v).await
    }

    async fn get_key_by_version(&self, version: u32) -> Result<EncryptionKey, KeyError> {
        let keys = self
            .keys
            .read()
            .map_err(|e| KeyError::Provider(e.to_string()))?;
        keys.get(&version)
            .cloned()
            .ok_or(KeyError::NotFound(version))
    }

    async fn rotate_key(&self) -> Result<u32, KeyError> {
        let new_version = {
            let guard = self
                .current_version
                .read()
                .map_err(|e| KeyError::Provider(e.to_string()))?;
            *guard + 1
        };

        {
            let mut keys = self
                .keys
                .write()
                .map_err(|e| KeyError::Provider(e.to_string()))?;
            keys.insert(new_version, Self::generate_key(new_version));
        }
        {
            let mut cv = self
                .current_version
                .write()
                .map_err(|e| KeyError::Provider(e.to_string()))?;
            *cv = new_version;
        }
        Ok(new_version)
    }
}

// ──────────────────────────── Credential Detection ────────────────────────────

/// The type of credential detected in text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialType {
    Password,
    ApiKey,
    JwtToken,
    BearerToken,
    AccessToken,
    SecretKey,
    PrivateKey,
}

/// Detects passwords, tokens, API keys, and other credentials in text.
///
/// Works with [`PoisonGuard`](crate::poison_guard::PoisonGuard) to block
/// credential injection into Memory (§11.3).
pub struct CredentialDetector;

type PatternList = Vec<(CredentialType, Regex)>;

/// Lazily compiled regex patterns.
fn patterns() -> &'static PatternList {
    static PATTERNS: OnceLock<PatternList> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            // Private key headers (PEM) — must check before Bearer/JWT.
            (
                CredentialType::PrivateKey,
                Regex::new(r"-----BEGIN\s+(RSA|EC|OPENSSH|PGP)\s+PRIVATE\s+KEY-----").unwrap(),
            ),
            // JWT — three dot-separated base64url segments starting with eyJ.
            (
                CredentialType::JwtToken,
                Regex::new(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+").unwrap(),
            ),
            // Bearer token.
            (
                CredentialType::BearerToken,
                Regex::new(r"Bearer\s+[a-zA-Z0-9._-]+").unwrap(),
            ),
            // API key — `sk-` prefix (e.g. OpenAI-style).
            (
                CredentialType::ApiKey,
                Regex::new(r"sk-[a-zA-Z0-9]{20,}").unwrap(),
            ),
            // API key — Stripe live/test secret key prefixes.
            (
                CredentialType::ApiKey,
                Regex::new(r"sk_(live|test)_[a-zA-Z0-9]{16,}").unwrap(),
            ),
            // API key — GitHub token prefixes (ghp_, gho_, ghs_, ghr_, ghu_).
            (
                CredentialType::ApiKey,
                Regex::new(r"gh[pousr]_[a-zA-Z0-9]{20,}").unwrap(),
            ),
            // API key — Slack token prefixes.
            (
                CredentialType::ApiKey,
                Regex::new(r"xox[bp]-[a-zA-Z0-9-]{10,}").unwrap(),
            ),
            // API key — Google API key prefix.
            (
                CredentialType::ApiKey,
                Regex::new(r"AIza[a-zA-Z0-9_-]{35}").unwrap(),
            ),
            // Password — connection string with embedded credentials.
            (
                CredentialType::Password,
                Regex::new(r"(?i)\w+://\S+:\S+@\S+").unwrap(),
            ),
            // API key — key=value form.
            (
                CredentialType::ApiKey,
                Regex::new(r"(?i)api[_-]?key\s*[=:]\s*\S+").unwrap(),
            ),
            // Password.
            (
                CredentialType::Password,
                Regex::new(r"(?i)password\s*[=:]\s*\S+").unwrap(),
            ),
            // Access token.
            (
                CredentialType::AccessToken,
                Regex::new(r"(?i)access[_-]?token\s*[=:]\s*\S+").unwrap(),
            ),
            // Secret key.
            (
                CredentialType::SecretKey,
                Regex::new(r"(?i)secret[_-]?key\s*[=:]\s*\S+").unwrap(),
            ),
        ]
    })
}

impl CredentialDetector {
    /// Scan text for credential patterns.
    ///
    /// Returns a list of *unique* [`CredentialType`]s detected. Order follows
    /// the pattern registration order above.
    pub fn scan(text: &str) -> Vec<CredentialType> {
        let mut found: Vec<CredentialType> = Vec::new();
        for (cred_type, re) in patterns() {
            if re.is_match(text) && !found.contains(cred_type) {
                found.push(cred_type.clone());
            }
        }
        found
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── InMemoryKeyProvider tests ──

    #[tokio::test]
    async fn test_inmemory_key_provider_create() {
        let provider = InMemoryKeyProvider::new();
        let key = provider.get_current_key().await.unwrap();
        assert_eq!(key.version, 1);
        assert_eq!(key.material.len(), 32, "SHA-256 produces 32 bytes");
    }

    #[tokio::test]
    async fn test_inmemory_key_provider_get_current() {
        let provider = InMemoryKeyProvider::new();
        let key = provider.get_current_key().await.unwrap();
        assert_eq!(key.version, 1);
        assert!(!key.material.is_empty());
    }

    #[tokio::test]
    async fn test_inmemory_key_provider_rotate() {
        let provider = InMemoryKeyProvider::new();
        let new_version = provider.rotate_key().await.unwrap();
        assert_eq!(new_version, 2);
        let key = provider.get_current_key().await.unwrap();
        assert_eq!(key.version, 2);
    }

    #[tokio::test]
    async fn test_inmemory_key_provider_get_by_version() {
        let provider = InMemoryKeyProvider::new();
        provider.rotate_key().await.unwrap(); // now v2
        let old = provider.get_key_by_version(1).await.unwrap();
        assert_eq!(old.version, 1);
        let cur = provider.get_key_by_version(2).await.unwrap();
        assert_eq!(cur.version, 2);
    }

    #[tokio::test]
    async fn test_inmemory_key_provider_get_nonexistent_version() {
        let provider = InMemoryKeyProvider::new();
        let result = provider.get_key_by_version(999).await;
        assert!(matches!(result, Err(KeyError::NotFound(999))));
    }

    // ── CredentialDetector tests ──

    #[test]
    fn test_credential_detector_password() {
        let result = CredentialDetector::scan("password=supersecret123");
        assert_eq!(result, vec![CredentialType::Password]);

        // case-insensitive
        let result = CredentialDetector::scan("PASSWORD: hunter2");
        assert!(result.contains(&CredentialType::Password));
    }

    #[test]
    fn test_credential_detector_api_key() {
        // key=value form
        let result = CredentialDetector::scan("api_key=abc123def456");
        assert!(result.contains(&CredentialType::ApiKey));

        // sk- prefix form
        let result = CredentialDetector::scan("sk-abcdefghijklmnopqrstuvwxyz1234567890");
        assert!(result.contains(&CredentialType::ApiKey));
    }

    #[test]
    fn test_credential_detector_jwt() {
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpXJqZzM";
        let result = CredentialDetector::scan(jwt);
        assert!(result.contains(&CredentialType::JwtToken));
    }

    #[test]
    fn test_credential_detector_bearer() {
        let result = CredentialDetector::scan("Authorization: Bearer abc123.def456");
        assert!(result.contains(&CredentialType::BearerToken));
    }

    #[test]
    fn test_credential_detector_access_token() {
        let result = CredentialDetector::scan("access_token=ya29.a0ARrdaM");
        assert!(result.contains(&CredentialType::AccessToken));
    }

    #[test]
    fn test_credential_detector_secret_key() {
        let result = CredentialDetector::scan("secret_key=my-secret-value");
        assert!(result.contains(&CredentialType::SecretKey));
    }

    #[test]
    fn test_credential_detector_private_key() {
        for variant in &["RSA", "EC", "OPENSSH", "PGP"] {
            let header = format!("-----BEGIN {variant} PRIVATE KEY-----");
            let result = CredentialDetector::scan(&header);
            assert!(
                result.contains(&CredentialType::PrivateKey),
                "failed for {variant}"
            );
        }
    }

    #[test]
    fn test_credential_detector_clean_text() {
        let result = CredentialDetector::scan("This is a normal sentence about the weather.");
        assert!(result.is_empty(), "expected no credentials, got {result:?}");
    }

    #[test]
    fn test_credential_detector_multiple() {
        let text = "password=abc123 api_key=xyz789 Bearer token123 \
                    -----BEGIN RSA PRIVATE KEY-----";
        let result = CredentialDetector::scan(text);
        assert!(result.contains(&CredentialType::Password));
        assert!(result.contains(&CredentialType::ApiKey));
        assert!(result.contains(&CredentialType::BearerToken));
        assert!(result.contains(&CredentialType::PrivateKey));
        assert!(
            result.len() >= 4,
            "expected at least 4 types, got {result:?}"
        );
    }

    // ── should_encrypt tests ──

    #[test]
    fn test_should_encrypt_restricted() {
        assert!(EncryptionManager::should_encrypt(&PrivacyClass::Restricted));
    }

    #[test]
    fn test_should_encrypt_confidential() {
        assert!(EncryptionManager::should_encrypt(
            &PrivacyClass::Confidential
        ));
    }

    #[test]
    fn test_should_encrypt_public() {
        assert!(!EncryptionManager::should_encrypt(&PrivacyClass::Public));
    }

    #[test]
    fn test_should_encrypt_internal() {
        assert!(!EncryptionManager::should_encrypt(&PrivacyClass::Internal));
    }

    // ── Encryption round-trip tests ──

    #[tokio::test]
    async fn test_encryption_round_trip() {
        let provider = Arc::new(InMemoryKeyProvider::new());
        let manager = EncryptionManager::new(provider);
        let plaintext = "sensitive data that needs protection";
        let (ciphertext, version) = manager.encrypt(plaintext).await.unwrap();
        assert_ne!(
            ciphertext, plaintext,
            "ciphertext should differ from plaintext"
        );
        let decrypted = manager.decrypt(&ciphertext, version).await.unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[tokio::test]
    async fn test_key_rotation_old_data_decryptable() {
        let provider = Arc::new(InMemoryKeyProvider::new());
        let manager = EncryptionManager::new(provider.clone());

        // Encrypt with v1.
        let plaintext = "data encrypted before rotation";
        let (ciphertext, v1) = manager.encrypt(plaintext).await.unwrap();
        assert_eq!(v1, 1);

        // Rotate to v2.
        let v2 = manager.rotate_key().await.unwrap();
        assert_eq!(v2, 2);

        // New encryption uses v2.
        let (_new_ct, new_ver) = manager.encrypt("new data").await.unwrap();
        assert_eq!(new_ver, 2);

        // Old data still decryptable with v1 key.
        let decrypted = manager.decrypt(&ciphertext, v1).await.unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[tokio::test]
    async fn test_key_error_not_found() {
        let provider = Arc::new(InMemoryKeyProvider::new());
        let manager = EncryptionManager::new(provider);
        let result = manager.decrypt("dGVzdA==", 999).await;
        assert!(matches!(result, Err(KeyError::NotFound(999))));
    }

    #[tokio::test]
    async fn test_key_error_decryption_failed() {
        let provider = Arc::new(InMemoryKeyProvider::new());
        let manager = EncryptionManager::new(provider);
        // Invalid base64 → DecryptionFailed.
        let result = manager.decrypt("!!!not-base64!!!", 1).await;
        assert!(matches!(result, Err(KeyError::DecryptionFailed(_))));
    }

    #[tokio::test]
    async fn test_key_error_provider() {
        struct FailingProvider;
        #[async_trait]
        impl KeyProvider for FailingProvider {
            async fn get_current_key(&self) -> Result<EncryptionKey, KeyError> {
                Err(KeyError::Provider("kms unavailable".into()))
            }
            async fn get_key_by_version(&self, _v: u32) -> Result<EncryptionKey, KeyError> {
                Err(KeyError::Provider("kms unavailable".into()))
            }
            async fn rotate_key(&self) -> Result<u32, KeyError> {
                Err(KeyError::Provider("kms unavailable".into()))
            }
        }
        let provider = Arc::new(FailingProvider);
        let manager = EncryptionManager::new(provider);
        let result = manager.encrypt("test").await;
        assert!(matches!(result, Err(KeyError::Provider(_))));
    }

    #[tokio::test]
    async fn test_empty_text_encryption() {
        let provider = Arc::new(InMemoryKeyProvider::new());
        let manager = EncryptionManager::new(provider);
        let (ciphertext, version) = manager.encrypt("").await.unwrap();
        let decrypted = manager.decrypt(&ciphertext, version).await.unwrap();
        assert_eq!(decrypted, "");
    }

    #[tokio::test]
    async fn test_unicode_text_encryption() {
        let provider = Arc::new(InMemoryKeyProvider::new());
        let manager = EncryptionManager::new(provider);
        let plaintext = "机密数据 — encrypt this 🔐";
        let (ciphertext, version) = manager.encrypt(plaintext).await.unwrap();
        let decrypted = manager.decrypt(&ciphertext, version).await.unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[tokio::test]
    async fn test_ciphertext_changes_after_rotation() {
        let provider = Arc::new(InMemoryKeyProvider::new());
        let manager = EncryptionManager::new(provider);
        let plaintext = "same plaintext";
        let (ct1, v1) = manager.encrypt(plaintext).await.unwrap();
        manager.rotate_key().await.unwrap();
        let (ct2, v2) = manager.encrypt(plaintext).await.unwrap();
        assert_ne!(v1, v2);
        assert_ne!(ct1, ct2, "ciphertext must differ after key rotation");
        // Both still decrypt correctly.
        assert_eq!(manager.decrypt(&ct1, v1).await.unwrap(), plaintext);
        assert_eq!(manager.decrypt(&ct2, v2).await.unwrap(), plaintext);
    }
}
