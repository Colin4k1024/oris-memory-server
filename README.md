# Oris Memory Server

An **Enterprise Context & Memory Service** — a model-agnostic control plane that lets AI agents persist, recall, and share context and memory with governed provenance, permissions, versioning, and audit. It is the single authority for Canonical User Context, Shared Task Memory, and cross-agent experience reuse, backed by **PostgreSQL + pgvector** as the canonical store and **Redis** for hot-context materialization.

> Memory layers logically; the implementation physically merges. Read paths fan out in parallel; write paths are primarily asynchronous.

## Features

### Control Plane
- **Canonical User Memory** — unified user profiles, identity links, explicit preferences, and versioned authoritative context (§5).
- **Shared Task Memory** — structured task contracts for cross-agent handoff: goals, findings, decisions, evidence, and ACL-gated handover (§6).
- **Context Router** — intent-based routing (Chat → PersonalTask → BusinessQuery → CrossAgentTask → DecisionSupport → DeepResearch) with per-intent latency and token budgets (§8).
- **Context Assembler** — multi-source context assembly with compression, token budgets (2K–8K), conflict flags, and degradation annotations (§7).
- **Memory Write Pipeline** — candidate → poison scan → importance scoring → entity linking → dedup → conflict → policy → store → outbox (§9.1).
- **Poison Guard** — prompt-injection detection, sensitive-data scanning, and untrusted-source isolation for memory poisoning protection (§11.3).

### Data Plane
- **PostgreSQL + pgvector** — authoritative store with HNSW vector index, hybrid search (structured + keyword + vector), RLS tenant isolation, and transactional outbox (§3.4).
- **Redis** — non-authoritative hot-context materialization, result caching, permission snapshots, and distributed locks (§3.5).
- **Pluggable Memory Engines** — Mem0 (personal long-term inferred memory), Cognee (enterprise semantic/graph memory), and Graphiti (temporal graph memory), each isolated with its own circuit breaker and timeout (§3.6–3.8).

### Governance & Security
- **RBAC + ABAC** — role, organization, factory, project, task, and data-classification based access control with PostgreSQL Row Level Security (§11.2).
- **Audit & Versioning** — full access audit trail, version snapshots, rollback, and lineage tracking (§11.5).
- **Retention & Forget** — differentiated retention policies, legal hold, soft/hard/cascade delete, and GDPR user-data bulk forget (§11.4).
- **Compliance** — PII masking, anonymization, encryption-at-rest with key management (§11.5).
- **Degradation** — graceful degradation with `must_deny()` fail-closed for permissions; recall can degrade, auth never fail-opens (§8.5).

### Agent Adapters
- **OpenClaw Adapter** — personal agent runtime adapter for canonical user context reads, candidate memory submission, and private memory retention (§3.1).
- **DeerFlow Adapter** — enterprise agent harness adapter with pluggable memory backend contract, context injection, result submission, and timeout/degradation (§3.2).

### API Surfaces
- **REST API (V1)** — 12 endpoints covering candidate submission, hybrid search, context assembly, user/task context, scope promotion, verification, forget, and lineage (§10.2).
- **MCP Tools (V1)** — 13 LLM agent tools mapping to the same service traits behind the REST API (§5.3).
- **Legacy Experience API** — retained `/v1/experience-assets` endpoints and `oris_experience_*` MCP tools for backward compatibility.

## Architecture

A self-contained Rust workspace of **four crates** with zero external monorepo dependencies:

| Crate | Role | Internal deps |
|-------|------|---------------|
| `oris-memory-server` | HTTP + MCP server entry point binary | `oris-control-plane` |
| `oris-control-plane` | Control plane logic: identity, router, assembler, write pipeline, governance, engines, adapters, API | `oris-memory-store`, `oris-memory-contract` |
| `oris-memory-store` | PostgreSQL + pgvector repos, Redis hot-context, SQLite legacy store, engine traits | `oris-memory-contract` |
| `oris-memory-contract` | Canonical shared types: `MemoryItem`, `CanonicalUserProfile`, `SharedTaskContext`, `GeneV1`/`CapsuleV1`/`ExperienceBundleV1`, enums | none |

```
oris-memory-server
  └── oris-control-plane
        ├── oris-memory-store
        │     └── oris-memory-contract
        └── oris-memory-contract
```

The dependency graph is acyclic and closed: the two leaf crates have no path dependencies, and `oris-control-plane` references them only via relative workspace paths.

### Control Plane Modules

| Module | Responsibility |
|--------|---------------|
| `identity.rs` | Identity Resolver — SSO/IAM delegated identity with Redis permission snapshot caching |
| `context_router.rs` | Intent classification + routing plan with latency/token budgets |
| `context_assembler.rs` | Multi-source context assembly + compression + conflict detection |
| `write_pipeline.rs` | Candidate → scan → score → dedup → conflict → policy → store → outbox |
| `poison_guard.rs` | Prompt injection + sensitive data detection |
| `governance/` | ACL, audit, version, forget, retention, policy |
| `engines/` | Mem0, Cognee, Graphiti engine adapters with circuit breakers |
| `adapters/` | OpenClaw and DeerFlow runtime adapters |
| `reflection.rs` | Episode → Experience → DecisionPattern → BestPractice → SOP/Skill |
| `scope_escalation.rs` | personal → team → factory → enterprise promotion with approval workflow |
| `outbox_worker.rs` | Transactional outbox processing + cache invalidation |
| `sot_verification.rs` | Source-of-Truth verification and validity checking |
| `degradation.rs` | Graceful degradation manager with fail-closed permissions |
| `compliance.rs` | Legal hold, PII masking, anonymization |
| `slo_monitoring.rs` | SLO metrics collection and violation detection |
| `cost_tracking.rs` | Engine cost comparison and budget alerts |
| `eval.rs` | Evaluation datasets: equipment failure, quality defect, user preference scenarios |
| `poc_framework.rs` | PoC framework with focus areas, success thresholds, stop conditions, data boundaries |

## Quick Start

### Prerequisites

- Rust toolchain (stable, edition 2021)
- PostgreSQL 15+ with `pgvector` extension
- Redis 6+

### Build

```bash
# Build the workspace
cargo build --release
```

### Run

```bash
# Run the HTTP server (V1 REST + legacy experience API)
cargo run --example server

# Or run the MCP stdio server (for an MCP host / agent runtime)
cargo run --bin oris-memory-server
```

### Configuration (env)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (e.g. `postgres://user:pass@localhost/oris`) |
| `REDIS_URL` | — | Redis connection string (e.g. `redis://localhost:6379`) |
| `ORIS_EXPERIENCE_DB` | `.oris/experience_repo.db` | Legacy SQLite experience store path |
| `ORIS_EXPERIENCE_KEY_DB` | `.oris/experience_keys.db` | API-key store SQLite path |
| `ORIS_AGENT_ID` | `local-agent` | Agent identity for MCP auth |
| `ORIS_MCP_SCOPES` | `experience:read,experience:write` | Comma-separated scopes (`*` = all) |

### V1 REST API endpoints

```
# Memory operations
POST   /v1/memories/candidates          # Submit candidate memory
GET    /v1/memories/{id}                 # Retrieve single memory
POST   /v1/memories/search               # Hybrid search (structured + keyword + vector)
POST   /v1/memories/forget               # Forget / delete (soft, hard, cascade, user_data)
POST   /v1/memories/{id}/promote         # Scope promotion
POST   /v1/memories/{id}/verify          # Source verification
GET    /v1/memories/{id}/lineage         # Version chain

# Context assembly
POST   /v1/context/assemble             # Assemble context for LLM prompt

# User & task context
GET    /v1/users/{id}/canonical-context  # Canonical user profile
GET    /v1/tasks/{id}/context            # Shared task context
PATCH  /v1/tasks/{id}/context            # Update task context

# User memory governance
GET    /v1/users/{id}/memories           # List user memories
PATCH  /v1/memories/{id}                 # Correct memory
GET    /v1/users/{id}/memories/export    # Export user data
POST   /v1/users/{id}/consent/revoke     # Revoke consent

# Health
GET    /v1/health
```

### Legacy experience API (retained)

```
GET    /experience                  POST /experience
GET    /v1/experience-assets        POST /v1/experience-assets
GET    /v1/experience-assets/{id}
GET    /v1/experience-assets/{id}/skill
POST   /v1/experience-assets/{id}/use
POST   /v1/experience-assets/{id}/outcomes
POST   /v1/experience-assets/{id}/promote
POST   /v1/experience-assets/{id}/revoke
POST   /mcp                          (MCP JSON-RPC over HTTP)
GET    /keys                         POST /keys
DELETE /keys/{key_id}                POST /keys/{key_id}/rotate
GET    /public-keys                  POST /public-keys
DELETE /public-keys/{sender_id}
```

### MCP Tools (V1)

13 LLM agent tools with JSON Schema input definitions:

| Tool | Service | REST endpoint |
|------|---------|---------------|
| `oris_memory_remember` | CandidateService | `POST /v1/memories/candidates` |
| `oris_memory_recall` | MemoryService | `GET /v1/memories/{id}` |
| `oris_memory_search` | SearchService | `POST /v1/memories/search` |
| `oris_memory_get_context` | AssembleService | `POST /v1/context/assemble` |
| `oris_memory_update` | VersionService | `GET /v1/memories/{id}/lineage` |
| `oris_memory_forget` | ForgetService | `POST /v1/memories/forget` |
| `oris_memory_share` | AssembleService | (ContextPackage projection) |
| `oris_memory_promote` | MemoryService | `POST /v1/memories/{id}/promote` |
| `oris_memory_reflect` | CandidateService | (experience reflection) |
| `oris_memory_verify` | MemoryService | `POST /v1/memories/{id}/verify` |
| `oris_user_get_context` | CanonicalUserService | `GET /v1/users/{id}/canonical-context` |
| `oris_task_get_context` | SharedTaskService | `GET /v1/tasks/{id}/context` |
| `oris_task_update_context` | SharedTaskService | `PATCH /v1/tasks/{id}/context` |

## Design Reference

The architecture follows the **Enterprise Context & Memory Service** design specification:

- **§2** — One control plane + two agent runtimes + layered pluggable data plane
- **§5** — Canonical User Memory: explicit authoritative context + inferred personal memory projection
- **§6** — Shared Task Memory: structured handoff contracts with version/etag optimistic locking
- **§8** — Context Router: intent-based routing with P95 latency targets (hot context ≤60ms, standard recall ≤200ms)
- **§9** — Write pipeline + hybrid retrieval + reflection/consolidation learning loop
- **§11** — Governance: source authority levels, RBAC+ABAC, memory poisoning protection, data lifecycle

See `docs/CONTROL_PLANE_ARCHITECTURE.md` for the detailed crate layout, schema DDL, and module design.

## Test Fixtures

A few `#[cfg(test)]` blocks embed golden fixtures via `include_str!` from paths outside the four crates (`spec/experience/golden/experience-bundle-v1.json` and `plugins/oris-experience/capabilities.json`). These are test-only references — the product library and binaries have no such external references at runtime.

## Known Issues

See the [GitHub issues tracker](https://github.com/Colin4k1024/oris-memory-server/issues) for tracked gaps:

- [#48](https://github.com/Colin4k1024/oris-memory-server/issues/48) — Dual engine type definitions (`store::traits` vs `control_plane::engine`)
- [#49](https://github.com/Colin4k1024/oris-memory-server/issues/49) — `EngineRegistry.search_all` is serial, not parallel
- [#50](https://github.com/Colin4k1024/oris-memory-server/issues/50) — Keyword search uses `ILIKE` instead of PostgreSQL full-text search
- [#51](https://github.com/Colin4k1024/oris-memory-server/issues/51) — `CanonicalUserProfile` lacks explicit/inferred preference separation
- [#52](https://github.com/Colin4k1024/oris-memory-server/issues/52) — Build & test verification status unconfirmed (no Rust toolchain in dev env)

## License

MIT OR Apache-2.0
