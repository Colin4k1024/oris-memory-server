//! Compliance: legal hold & data anonymization.
//!
//! - [`LegalHoldManager`] places and releases litigation-preservation holds
//!   on memory items. While a hold is active the item cannot be forgotten,
//!   deleted, or archived — even by GDPR forget requests.
//! - [`PiiMasker`] detects and masks personally identifiable information
//!   (email, phone, ID card, bank card, passport, address) in free text.
//! - [`Anonymizer`] applies PII masking and identifying-field removal to
//!   [`MemoryItem`]s for batch export.
//!
//! Architecture reference: §11.5 (legal hold, anonymization & masking) and
//! §11.3 (credentials must not enter Memory).

use chrono::{DateTime, Utc};
use oris_memory_store::memory_types::MemoryItem;
use oris_memory_store::postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::Row;
use uuid::Uuid;

// ──────────────────────────── LegalHoldManager ─────────────────────────

/// Manages legal holds on memory items. When a hold is active, the item
/// cannot be forgotten, deleted, or archived — even by GDPR forget requests.
pub struct LegalHoldManager {
    pool: Pool,
}

#[derive(Debug, thiserror::Error)]
pub enum LegalHoldError {
    #[error("memory {0} is under legal hold — cannot be deleted or forgotten")]
    HoldActive(Uuid),
    #[error("legal hold not found: {0}")]
    NotFound(Uuid),
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
}

/// A memory item currently held under legal preservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeldItem {
    pub memory_id: Uuid,
    pub reason: String,
    pub case_id: Option<String>,
    pub placed_at: DateTime<Utc>,
}

/// DDL for the `legal_hold` table. The base `memory_item` schema does not
/// carry a `legal_hold` column, so holds are tracked in a dedicated table.
const LEGAL_HOLD_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS legal_hold (
    memory_id  UUID PRIMARY KEY,
    tenant_id  TEXT NOT NULL,
    reason     TEXT NOT NULL,
    case_id    TEXT,
    placed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_legal_hold_tenant
    ON legal_hold (tenant_id, placed_at DESC);
"#;

impl LegalHoldManager {
    /// Create a new manager backed by the given connection pool.
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Ensure the `legal_hold` table exists. Idempotent.
    async fn ensure_schema(&self) -> Result<(), LegalHoldError> {
        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS legal_hold (
                memory_id  UUID PRIMARY KEY,
                tenant_id  TEXT NOT NULL,
                reason     TEXT NOT NULL,
                case_id    TEXT,
                placed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            r#"CREATE INDEX IF NOT EXISTS idx_legal_hold_tenant
               ON legal_hold (tenant_id, placed_at DESC)"#,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Place a legal hold on a memory item.
    ///
    /// Sets `legal_hold = true` semantics by inserting a row in `legal_hold`,
    /// which prevents all deletion/forget operations. The tenant is resolved
    /// from the `memory_item` row so callers need only know the memory id.
    /// Re-placing a hold on an already-held item is a no-op.
    pub async fn place_hold(
        &self,
        memory_id: Uuid,
        reason: &str,
        case_id: Option<&str>,
    ) -> Result<(), LegalHoldError> {
        self.ensure_schema().await?;
        sqlx::query(
            r#"INSERT INTO legal_hold (memory_id, tenant_id, reason, case_id, placed_at)
               SELECT $1, tenant_id, $2, $3, NOW()
               FROM memory_item
               WHERE memory_id = $1
               ON CONFLICT (memory_id) DO NOTHING"#,
        )
        .bind(memory_id)
        .bind(reason)
        .bind(case_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Release a legal hold.
    ///
    /// Returns [`LegalHoldError::NotFound`] when no hold exists for the given
    /// memory id.
    pub async fn release_hold(&self, memory_id: Uuid) -> Result<(), LegalHoldError> {
        self.ensure_schema().await?;
        let result = sqlx::query("DELETE FROM legal_hold WHERE memory_id = $1")
            .bind(memory_id)
            .execute(&self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(LegalHoldError::NotFound(memory_id));
        }
        Ok(())
    }

    /// Check if a memory item is under legal hold.
    pub async fn is_under_hold(&self, memory_id: Uuid) -> Result<bool, LegalHoldError> {
        self.ensure_schema().await?;
        let row = sqlx::query("SELECT 1 AS hit FROM legal_hold WHERE memory_id = $1")
            .bind(memory_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.is_some())
    }

    /// Get all memory items under legal hold for a tenant, newest first.
    pub async fn list_holds(&self, tenant_id: &str) -> Result<Vec<HeldItem>, LegalHoldError> {
        self.ensure_schema().await?;
        let rows = sqlx::query(
            r#"SELECT memory_id, reason, case_id, placed_at
               FROM legal_hold
               WHERE tenant_id = $1
               ORDER BY placed_at DESC"#,
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;

        let mut items = Vec::with_capacity(rows.len());
        for row in rows {
            items.push(HeldItem {
                memory_id: row.try_get("memory_id")?,
                reason: row.try_get("reason")?,
                case_id: row.try_get("case_id")?,
                placed_at: row.try_get("placed_at")?,
            });
        }
        Ok(items)
    }

    /// Verify that a memory can be deleted/forgotten.
    ///
    /// Returns [`LegalHoldError::HoldActive`] when the item is under legal
    /// hold. [`ForgetManager`] and [`RetentionManager`] must consult this
    /// before any delete/archive/forget operation.
    ///
    /// [`ForgetManager`]: crate::governance::forget::ForgetManager
    /// [`RetentionManager`]: crate::governance::retention::RetentionManager
    pub async fn verify_deletable(&self, memory_id: Uuid) -> Result<(), LegalHoldError> {
        if self.is_under_hold(memory_id).await? {
            return Err(LegalHoldError::HoldActive(memory_id));
        }
        Ok(())
    }
}

// ────────────────────────────────── PiiMasker ───────────────────────────

/// Detects and masks PII (Personally Identifiable Information) in text.
pub struct PiiMasker;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskReport {
    pub masked_count: usize,
    pub mask_types: Vec<MaskType>,
    pub original_length: usize,
    pub masked_length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskType {
    Email,
    PhoneNumber,
    IdCard,
    BankCard,
    Passport,
    Address,
}

/// A single PII hit produced while scanning text.
struct Hit {
    /// Exclusive end offset (char indices) of the matched span.
    end: usize,
    kind: MaskType,
    mask: String,
}

impl PiiMasker {
    /// Scan text and mask all detected PII.
    /// Returns `(masked_text, mask_report)`.
    pub fn mask(text: &str) -> (String, MaskReport) {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut out = String::with_capacity(n);
        let mut types = Vec::new();
        let mut i = 0;

        while i < n {
            if let Some(hit) = match_at(&chars, i) {
                out.push_str(&hit.mask);
                types.push(hit.kind);
                i = hit.end;
                continue;
            }
            out.push(chars[i]);
            i += 1;
        }

        let report = MaskReport {
            masked_count: types.len(),
            mask_types: types,
            original_length: n,
            masked_length: out.chars().count(),
        };
        (out, report)
    }
}

/// Try every PII matcher at position `i`. The first match wins.
///
/// Order matters: email is tried first so that addresses whose local part
/// starts with digits are not mistaken for phone numbers; the digit-run
/// matcher (ID card / bank card / phone) follows; passport and address are
/// tried last.
fn match_at(chars: &[char], i: usize) -> Option<Hit> {
    try_email(chars, i)
        .or_else(|| try_digit_run(chars, i))
        .or_else(|| try_passport(chars, i))
        .or_else(|| try_address(chars, i))
}

fn is_local_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-')
}

fn is_domain_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-')
}

/// Email: `local@domain.tld`.
fn try_email(chars: &[char], i: usize) -> Option<Hit> {
    let mut j = i;
    while j < chars.len() && is_local_char(chars[j]) {
        j += 1;
    }
    // Need a non-empty local part followed by '@'.
    if j == i || j >= chars.len() || chars[j] != '@' {
        return None;
    }
    let at = j;
    let mut k = at + 1;
    while k < chars.len() && is_domain_char(chars[k]) {
        k += 1;
    }
    if k == at + 1 {
        return None;
    }
    let domain: String = chars[at + 1..k].iter().collect();
    let dot = domain.rfind('.')?;
    let tld = &domain[dot + 1..];
    if dot == 0 || tld.len() < 2 || !tld.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    Some(Hit {
        end: k,
        kind: MaskType::Email,
        mask: "***@***.***".to_string(),
    })
}

/// ID card (18-digit Chinese), bank card, or Chinese phone number — all are
/// digit runs, so they share one collector and are disambiguated by length
/// and surrounding context.
fn try_digit_run(chars: &[char], i: usize) -> Option<Hit> {
    if i >= chars.len() || !chars[i].is_ascii_digit() {
        return None;
    }
    let mut j = i;
    while j < chars.len() && chars[j].is_ascii_digit() {
        j += 1;
    }
    let digits_len = j - i;

    // A trailing 'X'/'x' only extends a 17-digit run into an 18-char ID card.
    let mut end = j;
    let has_trailing_x =
        digits_len == 17 && j < chars.len() && (chars[j] == 'X' || chars[j] == 'x');
    if has_trailing_x {
        end = j + 1;
    }
    let total = end - i;

    // 1. ID card — 18 chars, first 17 digits, last is digit or X/x.
    if total == 18 {
        return Some(Hit {
            end,
            kind: MaskType::IdCard,
            mask: "*".repeat(18),
        });
    }

    // 2. Bank card — 16–19 digits preceded by "card"/"bank" context.
    if matches!(total, 16..=19) && bank_context(chars, i) {
        return Some(Hit {
            end,
            kind: MaskType::BankCard,
            mask: "****".to_string(),
        });
    }

    // 3. Chinese mobile — exactly 11 digits, starts with 1[3-9].
    if total == 11 && chars[i] == '1' && matches!(chars[i + 1], '3'..='9') {
        return Some(Hit {
            end,
            kind: MaskType::PhoneNumber,
            mask: "1**********".to_string(),
        });
    }

    None
}

/// Look back up to 24 chars for "card" or "bank" (case-insensitive).
fn bank_context(chars: &[char], start: usize) -> bool {
    let from = start.saturating_sub(24);
    let window: String = chars[from..start].iter().collect();
    let lower = window.to_lowercase();
    lower.contains("card") || lower.contains("bank")
}

/// Passport: one uppercase letter + exactly 8 digits, when the surrounding
/// context suggests a passport.
fn try_passport(chars: &[char], i: usize) -> Option<Hit> {
    if i >= chars.len() || !chars[i].is_ascii_uppercase() {
        return None;
    }
    if i + 9 > chars.len() {
        return None;
    }
    if !chars[i + 1..i + 9].iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    // Ensure exactly 8 digits (not part of a longer run).
    if i + 9 < chars.len() && chars[i + 9].is_ascii_digit() {
        return None;
    }
    if !passport_context(chars, i, i + 9) {
        return None;
    }
    Some(Hit {
        end: i + 9,
        kind: MaskType::Passport,
        mask: "*********".to_string(),
    })
}

/// Look within ~30 chars before (and a little after) for "passport"/"护照".
fn passport_context(chars: &[char], start: usize, end: usize) -> bool {
    let from = start.saturating_sub(30);
    let to = (end + 16).min(chars.len());
    let window: String = chars[from..to].iter().collect();
    window.to_lowercase().contains("passport") || window.contains("护照")
}

/// Address: a line beginning with "地址:" or "address:" (case-insensitive).
/// The label is preserved; the value that follows is redacted.
fn try_address(chars: &[char], i: usize) -> Option<Hit> {
    let at_line_start = i == 0 || chars[i - 1] == '\n';
    if !at_line_start {
        return None;
    }

    let chinese =
        i + 3 <= chars.len() && chars[i] == '地' && chars[i + 1] == '址' && chars[i + 2] == ':';
    let english = i + 8 <= chars.len() && {
        let label: String = chars[i..i + 8].iter().collect();
        label.eq_ignore_ascii_case("address:")
    };
    if !chinese && !english {
        return None;
    }

    let label_end = if chinese { i + 3 } else { i + 8 };
    let mut k = label_end;
    while k < chars.len() && chars[k] != '\n' {
        k += 1;
    }

    let label: String = chars[i..label_end].iter().collect();
    Some(Hit {
        end: k,
        kind: MaskType::Address,
        mask: format!("{label}[REDACTED]"),
    })
}

// ────────────────────────────────── Anonymizer ─────────────────────────

/// Anonymizes memory items for batch export.
pub struct Anonymizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizedItem {
    pub memory_id: Uuid,
    pub memory_type: String,
    pub scope: String,
    pub content: String,
    pub metadata: Value,
    pub anonymized: bool,
    pub anonymization_report: MaskReport,
}

/// Lower-cased key fragments that mark a metadata field as PII and cause it
/// to be dropped during anonymization.
const PII_KEY_FRAGMENTS: [&str; 18] = [
    "email",
    "phone",
    "mobile",
    "id_card",
    "idcard",
    "identity",
    "ssn",
    "address",
    "passport",
    "bank_card",
    "bankcard",
    "credit_card",
    "card_number",
    "account_number",
    "password",
    "credential",
    "token",
    "secret",
];

fn is_pii_key(key: &str) -> bool {
    let lower = key.to_lowercase();
    for &frag in PII_KEY_FRAGMENTS.iter() {
        if lower.contains(frag) {
            return true;
        }
    }
    false
}

/// Recursively strip object keys whose names look like PII. Arrays and leaf
/// values are preserved as-is.
fn strip_pii_keys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                if is_pii_key(k) {
                    continue;
                }
                out.insert(k.clone(), strip_pii_keys(v));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(strip_pii_keys).collect()),
        other => other.clone(),
    }
}

impl Anonymizer {
    /// Anonymize a memory item by removing/blurring identifying fields.
    ///
    /// `content` is PII-masked; `metadata` (derived from
    /// `structured_payload`) has PII-named keys removed. Identifying fields
    /// such as `subject_id`, `created_by_user`, and `source_reference` are
    /// dropped entirely since they are absent from [`AnonymizedItem`].
    pub fn anonymize(item: &MemoryItem) -> AnonymizedItem {
        let content_src = item.content.as_deref().unwrap_or("");
        let (masked_content, report) = PiiMasker::mask(content_src);
        let metadata = item
            .structured_payload
            .as_ref()
            .map(strip_pii_keys)
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

        AnonymizedItem {
            memory_id: item.memory_id,
            memory_type: item.memory_type.as_str().to_string(),
            scope: item.scope.as_str().to_string(),
            content: masked_content,
            metadata,
            anonymized: true,
            anonymization_report: report,
        }
    }

    /// Batch anonymize multiple items.
    pub fn anonymize_batch(items: &[MemoryItem]) -> Vec<AnonymizedItem> {
        items.iter().map(Self::anonymize).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oris_memory_store::{
        AuthorityLevel, MemoryStatus, MemoryType, PrivacyClass, Scope, SourceType,
    };
    use serde_json::json;

    /// Build a `MemoryItem` with test values.
    fn test_memory_item() -> MemoryItem {
        MemoryItem {
            memory_id: Uuid::new_v4(),
            tenant_id: "tenant-1".to_string(),
            memory_type: MemoryType::Semantic,
            scope: Scope::Personal,
            subject_type: Some("user".to_string()),
            subject_id: Some("user-123".to_string()),
            entity_refs: vec![json!({"type": "person", "id": "user-123"})],
            content: Some("test content".to_string()),
            structured_payload: Some(json!({"key": "value"})),
            embedding: None,
            source_type: SourceType::UserExplicit,
            source_reference: Some("ref-1".to_string()),
            evidence_refs: vec![json!({"ref": "evidence-1"})],
            confidence: 0.85,
            authority_level: AuthorityLevel::L1Authoritative,
            importance: 0.9,
            observed_at: Some(Utc::now()),
            valid_from: Some(Utc::now()),
            valid_to: None,
            privacy_class: PrivacyClass::Internal,
            acl: json!({"read": ["user-123"]}),
            retention_policy: Some("default".to_string()),
            status: MemoryStatus::Active,
            version: 3,
            derived_from: vec![json!("parent-uuid")],
            created_by_user: Some("user-123".to_string()),
            created_by_agent: None,
            last_verified_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    // 1. email detection and masking
    #[test]
    fn masks_email() {
        let (out, report) = PiiMasker::mask("Contact me at john.doe@example.com please");
        assert!(out.contains("***@***.***"));
        assert!(!out.contains("john.doe@example.com"));
        assert_eq!(report.masked_count, 1);
        assert_eq!(report.mask_types, vec![MaskType::Email]);
    }

    // 2. phone number (Chinese format) masking
    #[test]
    fn masks_chinese_phone() {
        let (out, report) = PiiMasker::mask("Call 13912345678 now");
        assert!(out.contains("1**********"));
        assert!(!out.contains("13912345678"));
        assert_eq!(report.masked_count, 1);
        assert_eq!(report.mask_types, vec![MaskType::PhoneNumber]);
    }

    // 3. ID card (18-digit) masking
    #[test]
    fn masks_id_card() {
        let (out, report) = PiiMasker::mask("ID 11010119900307663X");
        assert!(out.contains("******************"));
        assert!(!out.contains("11010119900307663X"));
        assert_eq!(report.masked_count, 1);
        assert_eq!(report.mask_types, vec![MaskType::IdCard]);
    }

    // 4. multiple PII types in one text
    #[test]
    fn masks_multiple_pii_types() {
        let text = "Email a@b.com, phone 13800138000, id 110101199003076634";
        let (out, report) = PiiMasker::mask(text);
        assert!(out.contains("***@***.***"));
        assert!(out.contains("1**********"));
        assert!(out.contains("******************"));
        assert_eq!(report.masked_count, 3);
        assert_eq!(
            report.mask_types,
            vec![MaskType::Email, MaskType::PhoneNumber, MaskType::IdCard]
        );
    }

    // 5. clean text — no masks
    #[test]
    fn clean_text_unmasked() {
        let text = "Hello world, no secrets here";
        let (out, report) = PiiMasker::mask(text);
        assert_eq!(out, text);
        assert_eq!(report.masked_count, 0);
        assert!(report.mask_types.is_empty());
    }

    // 6. mask report correct count and types
    #[test]
    fn mask_report_counts_and_lengths() {
        let text = "reach jane@doe.io and 13800138000";
        let (out, report) = PiiMasker::mask(text);
        assert_eq!(report.masked_count, 2);
        assert_eq!(report.mask_types.len(), 2);
        assert_eq!(report.original_length, text.chars().count());
        // masked text replaces the PII spans with fixed-width masks
        assert_eq!(report.masked_length, out.chars().count());
    }

    // 7. single item anonymization
    #[test]
    fn anonymize_single_item() {
        let mut item = test_memory_item();
        item.content = Some("reach me at jane@doe.io".to_string());
        let anon = Anonymizer::anonymize(&item);
        assert_eq!(anon.memory_id, item.memory_id);
        assert_eq!(anon.memory_type, "semantic");
        assert_eq!(anon.scope, "personal");
        assert!(anon.anonymized);
        assert!(anon.content.contains("***@***.***"));
        assert!(!anon.content.contains("jane@doe.io"));
        assert!(anon.anonymization_report.masked_count >= 1);
    }

    // 8. batch anonymization
    #[test]
    fn anonymize_batch_items() {
        let mut a = test_memory_item();
        a.content = Some("mail a@b.com".to_string());
        let mut b = test_memory_item();
        b.content = Some("phone 13912345678".to_string());
        let items = vec![a.clone(), b.clone()];
        let result = Anonymizer::anonymize_batch(&items);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].memory_id, a.memory_id);
        assert_eq!(result[1].memory_id, b.memory_id);
        assert!(result[0].content.contains("***@***.***"));
        assert!(result[1].content.contains("1**********"));
    }

    // 9. metadata PII fields removed
    #[test]
    fn anonymize_strips_pii_metadata_keys() {
        let mut item = test_memory_item();
        item.structured_payload = Some(json!({
            "email": "secret@corp.com",
            "note": "keep me",
            "phone": "13912345678",
            "nested": { "address": "somewhere", "count": 3 }
        }));
        let anon = Anonymizer::anonymize(&item);
        let meta = anon.metadata.as_object().expect("metadata is an object");
        assert!(!meta.contains_key("email"));
        assert!(!meta.contains_key("phone"));
        assert!(meta.contains_key("note"));
        let nested = meta
            .get("nested")
            .and_then(|v| v.as_object())
            .expect("nested object");
        assert!(!nested.contains_key("address"));
        assert!(nested.contains_key("count"));
    }

    // 10. LegalHoldError: HoldActive variant
    #[test]
    fn legal_hold_error_hold_active_display() {
        let id = Uuid::new_v4();
        let err = LegalHoldError::HoldActive(id);
        let msg = format!("{err}");
        assert!(msg.contains("under legal hold"));
        assert!(msg.contains(&id.to_string()));
    }

    // 11. LegalHoldError: NotFound variant
    #[test]
    fn legal_hold_error_not_found_display() {
        let id = Uuid::new_v4();
        let err = LegalHoldError::NotFound(id);
        let msg = format!("{err}");
        assert!(msg.contains("legal hold not found"));
        assert!(msg.contains(&id.to_string()));
    }

    // 12. HeldItem serialization
    #[test]
    fn held_item_serialization_roundtrip() {
        let item = HeldItem {
            memory_id: Uuid::new_v4(),
            reason: "pending litigation".to_string(),
            case_id: Some("CASE-2026-001".to_string()),
            placed_at: Utc::now(),
        };
        let serialized = serde_json::to_string(&item).expect("serialize");
        let restored: HeldItem = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(restored.memory_id, item.memory_id);
        assert_eq!(restored.reason, item.reason);
        assert_eq!(restored.case_id, item.case_id);
    }

    // 13. MaskType variants
    #[test]
    fn mask_type_serde_variants() {
        assert_eq!(
            serde_json::to_string(&MaskType::Email).unwrap(),
            "\"email\""
        );
        assert_eq!(
            serde_json::to_string(&MaskType::PhoneNumber).unwrap(),
            "\"phone_number\""
        );
        assert_eq!(
            serde_json::to_string(&MaskType::IdCard).unwrap(),
            "\"id_card\""
        );
        assert_eq!(
            serde_json::to_string(&MaskType::BankCard).unwrap(),
            "\"bank_card\""
        );
        assert_eq!(
            serde_json::to_string(&MaskType::Passport).unwrap(),
            "\"passport\""
        );
        assert_eq!(
            serde_json::to_string(&MaskType::Address).unwrap(),
            "\"address\""
        );
    }

    // 14. edge case: empty text
    #[test]
    fn mask_empty_text() {
        let (out, report) = PiiMasker::mask("");
        assert_eq!(out, "");
        assert_eq!(report.masked_count, 0);
        assert_eq!(report.original_length, 0);
        assert_eq!(report.masked_length, 0);
    }

    // 15. edge case: text with only PII
    #[test]
    fn mask_text_that_is_only_pii() {
        let (out, report) = PiiMasker::mask("a@b.com");
        assert_eq!(out, "***@***.***");
        assert_eq!(report.masked_count, 1);
        assert_eq!(report.mask_types, vec![MaskType::Email]);
        assert_eq!(report.original_length, 7);
    }
}
