//! Context Router — intent-based routing with latency budgets.
//!
//! Classifies an incoming request by [`RequestIntent`] and produces a
//! [`RoutingPlan`] that tells downstream layers which context sources to
//! activate (hot context, canonical-user memory, shared task, structured /
//! vector / graph retrieval, SoT verification) and what latency and token
//! budgets to respect.
//!
//! The routing rules follow the architecture spec:
//! - §8.2 — context escalation ladder: each intent is a superset of the
//!   previous one's context sources.
//! - §8.4 — per-intent latency and token budgets.
//!
//! Intent is either supplied explicitly by the caller (via
//! [`RouteRequest::intent`]) or auto-detected from the natural-language query
//! by [`ContextRouter::detect_intent`]. A present [`RouteRequest::task_id`]
//! forces [`RequestIntent::CrossAgentTask`] regardless of query text, since a
//! cross-agent task is a structural signal rather than a linguistic one.

use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};
use uuid::Uuid;

use crate::identity::ResolvedIdentity;

// ──────────────────────────── Intent ────────────────────────────

/// The classified intent of a request.
///
/// Each variant activates progressively more context sources, forming the
/// escalation ladder defined in architecture §8.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestIntent {
    /// 闲聊 — casual conversation. Hot context only.
    Chat,
    /// 个人事务 — personal task. Adds canonical-user memory.
    PersonalTask,
    /// 业务查询 — business query. Adds structured retrieval.
    BusinessQuery,
    /// 跨 Agent — cross-agent task. Adds shared task context.
    CrossAgentTask,
    /// 决策支持 — decision support. Adds vector search + SoT verification.
    DecisionSupport,
    /// 深度研究 — deep research. Adds graph search (full escalation).
    DeepResearch,
}

impl RequestIntent {
    /// Stable lowercase label for tracing and metrics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::PersonalTask => "personal_task",
            Self::BusinessQuery => "business_query",
            Self::CrossAgentTask => "cross_agent_task",
            Self::DecisionSupport => "decision_support",
            Self::DeepResearch => "deep_research",
        }
    }
}

// ──────────────────────────── Routing Plan ────────────────────────────

/// The decision produced by [`ContextRouter::route`].
///
/// Each boolean flag activates one context source; a `false` flag means
/// "do not fetch from this source". The two budget fields cap how much
/// wall-clock time and how many tokens the assembled context may consume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingPlan {
    pub intent: RequestIntent,
    pub hot_context: bool,
    pub canonical_user: bool,
    pub shared_task: bool,
    pub structured_search: bool,
    pub vector_search: bool,
    pub graph_search: bool,
    pub sot_verification: bool,
    pub latency_budget_ms: u64,
    pub token_budget: usize,
}

// ──────────────────────────── Route Request ────────────────────────────

/// Input to [`ContextRouter::route`].
///
/// `identity` is the resolved caller identity. When `intent` is `None` the
/// router auto-detects it from `query` (unless `task_id` is present, which
/// forces [`RequestIntent::CrossAgentTask`]). `max_latency_ms` and
/// `max_tokens` let the caller *tighten* (never loosen) the intent's default
/// budgets.
#[derive(Debug, Clone)]
pub struct RouteRequest<'a> {
    pub identity: &'a ResolvedIdentity,
    pub query: String,
    pub intent: Option<RequestIntent>,
    /// Present when this is a cross-agent task; forces that intent.
    pub task_id: Option<Uuid>,
    /// Caller cap on latency; the lesser of this and the intent budget wins.
    pub max_latency_ms: Option<u64>,
    /// Caller cap on tokens; the lesser of this and the intent budget wins.
    pub max_tokens: Option<usize>,
}

// ──────────────────────────── Router ────────────────────────────

/// Intent-based context router.
///
/// Holds the configurable `default_token_budget` (4096 by default) from which
/// every intent's token budget is derived as a fixed multiple, reproducing the
/// §8.4 table (2K / 4K / 4K / 6K / 8K / 8K). Latency budgets are fixed per
/// intent and are unaffected by `default_token_budget`.
pub struct ContextRouter {
    default_token_budget: usize,
}

impl Default for ContextRouter {
    fn default() -> Self {
        Self {
            default_token_budget: 4096,
        }
    }
}

impl ContextRouter {
    /// Create a router with the standard 4096-token default budget.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a router with a custom base token budget.
    ///
    /// All per-intent token budgets scale relative to this base.
    pub fn with_token_budget(default_token_budget: usize) -> Self {
        Self {
            default_token_budget,
        }
    }

    /// Route a request based on intent, identity, and optional hints.
    ///
    /// Resolution order:
    /// 1. Explicit `intent` (caller override) — used directly.
    /// 2. `task_id` present → [`RequestIntent::CrossAgentTask`].
    /// 3. Auto-detect from `query` via [`Self::detect_intent`].
    ///
    /// Caller-supplied `max_latency_ms` / `max_tokens` can only *tighten*
    /// the intent's default budget (the lesser value wins), never exceed it.
    #[instrument(skip_all)]
    pub fn route(&self, req: &RouteRequest) -> RoutingPlan {
        let intent = req.intent.unwrap_or_else(|| match req.task_id {
            Some(_) => RequestIntent::CrossAgentTask,
            None => Self::detect_intent(&req.query),
        });

        let mut plan = Self::plan_for_intent(intent, self.default_token_budget);

        if let Some(cap) = req.max_latency_ms {
            plan.latency_budget_ms = plan.latency_budget_ms.min(cap);
        }
        if let Some(cap) = req.max_tokens {
            plan.token_budget = plan.token_budget.min(cap);
        }

        debug!(
            intent = %intent.as_str(),
            latency_budget_ms = plan.latency_budget_ms,
            token_budget = plan.token_budget,
            "context routing plan resolved"
        );

        plan
    }

    /// Auto-detect intent from a natural-language query.
    ///
    /// Keyword heuristics (first match wins, in priority order):
    /// 1. decision / decide / recommend / "should we" → [`DecisionSupport`]
    /// 2. research / analyze / investigate / compare → [`DeepResearch`]
    /// 3. business terms (product, order, supplier, quality) → [`BusinessQuery`]
    /// 4. personal pronouns (my, I) → [`PersonalTask`]
    /// 5. fallback → [`Chat`]
    ///
    /// `task_id`-driven [`CrossAgentTask`] is handled in [`route`](Self::route),
    /// since this function only sees the query text.
    #[instrument]
    pub fn detect_intent(query: &str) -> RequestIntent {
        let q = query.to_lowercase();

        const DECISION: [&str; 4] = ["decision", "decide", "recommend", "should we"];
        if DECISION.into_iter().any(|kw| matches_word(&q, kw)) {
            return RequestIntent::DecisionSupport;
        }

        const RESEARCH: [&str; 4] = ["research", "analyze", "investigate", "compare"];
        if RESEARCH.into_iter().any(|kw| matches_word(&q, kw)) {
            return RequestIntent::DeepResearch;
        }

        const BUSINESS: [&str; 4] = ["product", "order", "supplier", "quality"];
        if BUSINESS.into_iter().any(|kw| matches_word(&q, kw)) {
            return RequestIntent::BusinessQuery;
        }

        const PERSONAL: [&str; 2] = ["my", "i"];
        if PERSONAL.into_iter().any(|kw| matches_word(&q, kw)) {
            return RequestIntent::PersonalTask;
        }

        RequestIntent::Chat
    }

    /// Build the base [`RoutingPlan`] for an intent, before caller overrides.
    fn plan_for_intent(intent: RequestIntent, default_token_budget: usize) -> RoutingPlan {
        let latency_budget_ms = latency_for(intent);
        let token_budget = tokens_for(intent, default_token_budget);

        match intent {
            RequestIntent::Chat => RoutingPlan {
                intent,
                hot_context: true,
                canonical_user: false,
                shared_task: false,
                structured_search: false,
                vector_search: false,
                graph_search: false,
                sot_verification: false,
                latency_budget_ms,
                token_budget,
            },
            RequestIntent::PersonalTask => RoutingPlan {
                intent,
                hot_context: true,
                canonical_user: true,
                shared_task: false,
                structured_search: false,
                vector_search: false,
                graph_search: false,
                sot_verification: false,
                latency_budget_ms,
                token_budget,
            },
            RequestIntent::BusinessQuery => RoutingPlan {
                intent,
                hot_context: true,
                canonical_user: true,
                shared_task: false,
                structured_search: true,
                vector_search: false,
                graph_search: false,
                sot_verification: false,
                latency_budget_ms,
                token_budget,
            },
            RequestIntent::CrossAgentTask => RoutingPlan {
                intent,
                hot_context: true,
                canonical_user: true,
                shared_task: true,
                structured_search: true,
                vector_search: false,
                graph_search: false,
                sot_verification: false,
                latency_budget_ms,
                token_budget,
            },
            RequestIntent::DecisionSupport => RoutingPlan {
                intent,
                hot_context: true,
                canonical_user: true,
                shared_task: true,
                structured_search: true,
                vector_search: true,
                graph_search: false,
                sot_verification: true,
                latency_budget_ms,
                token_budget,
            },
            RequestIntent::DeepResearch => RoutingPlan {
                intent,
                hot_context: true,
                canonical_user: true,
                shared_task: true,
                structured_search: true,
                vector_search: true,
                graph_search: true,
                sot_verification: true,
                latency_budget_ms,
                token_budget,
            },
        }
    }
}

// ──────────────────────────── Budget lookups ────────────────────────────

/// Fixed latency budget (ms) per intent — architecture §8.4.
fn latency_for(intent: RequestIntent) -> u64 {
    match intent {
        RequestIntent::Chat => 60,
        RequestIntent::PersonalTask => 120,
        RequestIntent::BusinessQuery => 200,
        RequestIntent::CrossAgentTask => 200,
        RequestIntent::DecisionSupport => 500,
        RequestIntent::DeepResearch => 800,
    }
}

/// Token budget as a multiple of the router's `default_token_budget`.
///
/// With the default base of 4096 this reproduces the §8.4 table:
/// Chat 2K, PersonalTask 4K, BusinessQuery 4K, CrossAgentTask 6K,
/// DecisionSupport 8K, DeepResearch 8K.
fn tokens_for(intent: RequestIntent, base: usize) -> usize {
    match intent {
        RequestIntent::Chat => base / 2,
        RequestIntent::PersonalTask | RequestIntent::BusinessQuery => base,
        RequestIntent::CrossAgentTask => base * 3 / 2,
        RequestIntent::DecisionSupport | RequestIntent::DeepResearch => base * 2,
    }
}

// ──────────────────────────── Text matching ────────────────────────────

/// Case-insensitive whole-word / whole-phrase match.
///
/// `haystack` and `needle` must already be lowercased. Returns `true` if
/// `needle` occurs in `haystack` bounded by non-alphanumeric characters (or
/// string edges). This avoids false positives such as "my" inside "economy"
/// or "order" inside "border" while still matching phrases like "should we".
fn matches_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let nlen = needle.len();
    if nlen == 0 || nlen > bytes.len() {
        return false;
    }

    let mut from = 0usize;
    while let Some(rel) = haystack[from..].find(needle) {
        let start = from + rel;
        let end = start + nlen;

        let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let after_ok = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();

        if before_ok && after_ok {
            return true;
        }
        // `end` is a char boundary (it follows a matched ASCII substring), so
        // slicing `haystack[end..]` is always valid.
        from = end;
    }
    false
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::ResolvedIdentity;

    /// Build a routing plan for an explicit intent (no query, no overrides).
    fn plan_for(intent: RequestIntent) -> RoutingPlan {
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: String::new(),
            intent: Some(intent),
            task_id: None,
            max_latency_ms: None,
            max_tokens: None,
        };
        ContextRouter::new().route(&req)
    }

    /// Assert every flag field of `plan` against the expected matrix.
    fn assert_flags(
        plan: &RoutingPlan,
        intent: RequestIntent,
        hot: bool,
        user: bool,
        shared: bool,
        structured: bool,
        vector: bool,
        graph: bool,
        sot: bool,
    ) {
        assert_eq!(plan.intent, intent, "intent");
        assert_eq!(plan.hot_context, hot, "hot_context");
        assert_eq!(plan.canonical_user, user, "canonical_user");
        assert_eq!(plan.shared_task, shared, "shared_task");
        assert_eq!(plan.structured_search, structured, "structured_search");
        assert_eq!(plan.vector_search, vector, "vector_search");
        assert_eq!(plan.graph_search, graph, "graph_search");
        assert_eq!(plan.sot_verification, sot, "sot_verification");
    }

    // ── Routing rules (§8.2 escalation ladder) ──

    #[test]
    fn chat_plan_hot_context_only() {
        let plan = plan_for(RequestIntent::Chat);
        assert_flags(
            &plan,
            RequestIntent::Chat,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
        );
    }

    #[test]
    fn personal_task_plan_adds_canonical_user() {
        let plan = plan_for(RequestIntent::PersonalTask);
        assert_flags(
            &plan,
            RequestIntent::PersonalTask,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
        );
    }

    #[test]
    fn business_query_plan_adds_structured_search() {
        let plan = plan_for(RequestIntent::BusinessQuery);
        assert_flags(
            &plan,
            RequestIntent::BusinessQuery,
            true,
            true,
            false,
            true,
            false,
            false,
            false,
        );
    }

    #[test]
    fn cross_agent_task_plan_adds_shared_task() {
        let plan = plan_for(RequestIntent::CrossAgentTask);
        assert_flags(
            &plan,
            RequestIntent::CrossAgentTask,
            true,
            true,
            true,
            true,
            false,
            false,
            false,
        );
    }

    #[test]
    fn decision_support_plan_adds_vector_and_sot() {
        let plan = plan_for(RequestIntent::DecisionSupport);
        assert_flags(
            &plan,
            RequestIntent::DecisionSupport,
            true,
            true,
            true,
            true,
            true,
            false,
            true,
        );
    }

    #[test]
    fn deep_research_plan_full_escalation() {
        let plan = plan_for(RequestIntent::DeepResearch);
        assert_flags(
            &plan,
            RequestIntent::DeepResearch,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
        );
    }

    // ── Latency & token budgets (§8.4) ──

    #[test]
    fn latency_budgets_match_spec_table() {
        assert_eq!(plan_for(RequestIntent::Chat).latency_budget_ms, 60);
        assert_eq!(plan_for(RequestIntent::PersonalTask).latency_budget_ms, 120);
        assert_eq!(
            plan_for(RequestIntent::BusinessQuery).latency_budget_ms,
            200
        );
        assert_eq!(
            plan_for(RequestIntent::CrossAgentTask).latency_budget_ms,
            200
        );
        assert_eq!(
            plan_for(RequestIntent::DecisionSupport).latency_budget_ms,
            500
        );
        assert_eq!(plan_for(RequestIntent::DeepResearch).latency_budget_ms, 800);
    }

    #[test]
    fn token_budgets_match_spec_table() {
        assert_eq!(plan_for(RequestIntent::Chat).token_budget, 2048);
        assert_eq!(plan_for(RequestIntent::PersonalTask).token_budget, 4096);
        assert_eq!(plan_for(RequestIntent::BusinessQuery).token_budget, 4096);
        assert_eq!(plan_for(RequestIntent::CrossAgentTask).token_budget, 6144);
        assert_eq!(plan_for(RequestIntent::DecisionSupport).token_budget, 8192);
        assert_eq!(plan_for(RequestIntent::DeepResearch).token_budget, 8192);
    }

    #[test]
    fn token_budget_scales_with_custom_base() {
        let router = ContextRouter::with_token_budget(8192);
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: String::new(),
            intent: Some(RequestIntent::Chat),
            task_id: None,
            max_latency_ms: None,
            max_tokens: None,
        };
        // base 8192 → Chat = base/2 = 4096
        assert_eq!(router.route(&req).token_budget, 4096);

        let req = RouteRequest {
            identity: &id,
            query: String::new(),
            intent: Some(RequestIntent::DecisionSupport),
            task_id: None,
            max_latency_ms: None,
            max_tokens: None,
        };
        // base 8192 → DecisionSupport = base*2 = 16384
        assert_eq!(router.route(&req).token_budget, 16384);
    }

    // ── Intent detection heuristics ──

    #[test]
    fn detect_decision_keywords() {
        assert_eq!(
            ContextRouter::detect_intent("should we proceed?"),
            RequestIntent::DecisionSupport
        );
        assert_eq!(
            ContextRouter::detect_intent("decide on the plan"),
            RequestIntent::DecisionSupport
        );
        assert_eq!(
            ContextRouter::detect_intent("recommend a tool"),
            RequestIntent::DecisionSupport
        );
        assert_eq!(
            ContextRouter::detect_intent("this is a big decision"),
            RequestIntent::DecisionSupport
        );
    }

    #[test]
    fn detect_research_keywords() {
        assert_eq!(
            ContextRouter::detect_intent("research the topic"),
            RequestIntent::DeepResearch
        );
        assert_eq!(
            ContextRouter::detect_intent("analyze the data"),
            RequestIntent::DeepResearch
        );
        assert_eq!(
            ContextRouter::detect_intent("investigate the failure"),
            RequestIntent::DeepResearch
        );
        assert_eq!(
            ContextRouter::detect_intent("compare the options"),
            RequestIntent::DeepResearch
        );
    }

    #[test]
    fn detect_business_terms() {
        assert_eq!(
            ContextRouter::detect_intent("check product availability"),
            RequestIntent::BusinessQuery
        );
        assert_eq!(
            ContextRouter::detect_intent("track the order"),
            RequestIntent::BusinessQuery
        );
        assert_eq!(
            ContextRouter::detect_intent("contact the supplier"),
            RequestIntent::BusinessQuery
        );
        assert_eq!(
            ContextRouter::detect_intent("review quality metrics"),
            RequestIntent::BusinessQuery
        );
    }

    #[test]
    fn detect_personal_pronouns() {
        assert_eq!(
            ContextRouter::detect_intent("my tasks today"),
            RequestIntent::PersonalTask
        );
        assert_eq!(
            ContextRouter::detect_intent("I need help"),
            RequestIntent::PersonalTask
        );
    }

    #[test]
    fn detect_defaults_to_chat() {
        assert_eq!(
            ContextRouter::detect_intent("hello there"),
            RequestIntent::Chat
        );
        assert_eq!(
            ContextRouter::detect_intent("good morning"),
            RequestIntent::Chat
        );
    }

    // ── Detection priority (first match wins) ──

    #[test]
    fn decision_beats_personal() {
        // "I" present but "decide" wins.
        assert_eq!(
            ContextRouter::detect_intent("I must decide soon"),
            RequestIntent::DecisionSupport
        );
    }

    #[test]
    fn business_beats_personal() {
        // "my" present but "order" (business) wins.
        assert_eq!(
            ContextRouter::detect_intent("track my order"),
            RequestIntent::BusinessQuery
        );
    }

    #[test]
    fn research_beats_business() {
        // "product" present but "research" wins.
        assert_eq!(
            ContextRouter::detect_intent("research the product line"),
            RequestIntent::DeepResearch
        );
    }

    #[test]
    fn word_boundary_prevents_false_positives() {
        // "my" inside "economy" must NOT trigger PersonalTask.
        assert_eq!(
            ContextRouter::detect_intent("the economy is improving"),
            RequestIntent::Chat
        );
        // "order" inside "border" must NOT trigger BusinessQuery.
        assert_eq!(
            ContextRouter::detect_intent("cross the border"),
            RequestIntent::Chat
        );
    }

    // ── route() resolution order ──

    #[test]
    fn explicit_intent_overrides_detection() {
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: "research the data".into(), // would auto-detect DeepResearch
            intent: Some(RequestIntent::Chat), // explicit override
            task_id: None,
            max_latency_ms: None,
            max_tokens: None,
        };
        assert_eq!(ContextRouter::new().route(&req).intent, RequestIntent::Chat);
    }

    #[test]
    fn task_id_forces_cross_agent() {
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: "hello there".into(), // would auto-detect Chat
            intent: None,
            task_id: Some(Uuid::nil()), // forces CrossAgentTask
            max_latency_ms: None,
            max_tokens: None,
        };
        assert_eq!(
            ContextRouter::new().route(&req).intent,
            RequestIntent::CrossAgentTask
        );
    }

    #[test]
    fn auto_detects_when_intent_none_and_no_task_id() {
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: "should we proceed".into(),
            intent: None,
            task_id: None,
            max_latency_ms: None,
            max_tokens: None,
        };
        assert_eq!(
            ContextRouter::new().route(&req).intent,
            RequestIntent::DecisionSupport
        );
    }

    // ── Caller budget overrides (tighten only) ──

    #[test]
    fn caller_latency_override_tightens() {
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: String::new(),
            intent: Some(RequestIntent::Chat), // default 60ms
            task_id: None,
            max_latency_ms: Some(30),
            max_tokens: None,
        };
        assert_eq!(ContextRouter::new().route(&req).latency_budget_ms, 30);
    }

    #[test]
    fn caller_latency_cannot_loosen() {
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: String::new(),
            intent: Some(RequestIntent::Chat), // default 60ms
            task_id: None,
            max_latency_ms: Some(500), // ignored — exceeds intent budget
            max_tokens: None,
        };
        assert_eq!(ContextRouter::new().route(&req).latency_budget_ms, 60);
    }

    #[test]
    fn caller_token_override_tightens() {
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: String::new(),
            intent: Some(RequestIntent::Chat), // default 2048 tokens
            task_id: None,
            max_latency_ms: None,
            max_tokens: Some(512),
        };
        assert_eq!(ContextRouter::new().route(&req).token_budget, 512);
    }

    #[test]
    fn caller_token_cannot_loosen() {
        let id = ResolvedIdentity::system("acme", "t");
        let req = RouteRequest {
            identity: &id,
            query: String::new(),
            intent: Some(RequestIntent::Chat), // default 2048 tokens
            task_id: None,
            max_latency_ms: None,
            max_tokens: Some(99_999), // ignored — exceeds intent budget
        };
        assert_eq!(ContextRouter::new().route(&req).token_budget, 2048);
    }

    #[test]
    fn intent_as_str_labels() {
        assert_eq!(RequestIntent::Chat.as_str(), "chat");
        assert_eq!(RequestIntent::DeepResearch.as_str(), "deep_research");
    }
}
