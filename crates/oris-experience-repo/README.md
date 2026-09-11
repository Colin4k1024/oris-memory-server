# Oris Memory Server

An MCP (Model Context Protocol) based cross-session memory server for AI coding agents. It lets autonomous agents persist, search, and reuse experiences — **genes** (reusable units of capability) and **capsules** (validated outcomes) — across sessions and across agents, so that hard-won knowledge survives process restarts and can be shared safely with provenance, signing, and governed promotion.

## Features

- **Gene & experience storage** — durable SQLite-backed store for genes, capsules, and usage receipts with schema migrations.
- **MCP JSON-RPC surface** — a full `tools/list` + `tools/call` server exposing read, write, govern, and admin tool families, usable over stdio or Streamable HTTP.
- **HTTP REST API** — Axum endpoints for searching, proposing, using, and promoting experience assets, plus key and public-key management.
- **CLI client** — a built-in `oris-experience-mcp` binary that speaks the MCP JSON-RPC protocol over stdio.
- **SQLite persistence** — `rusqlite` with the `bundled` feature; no external database server required.
- **API-key auth & scopes** — per-agent keys with `experience:read|write|govern|admin` scopes, rotation, and revocation.
- **Ed25519 PKI** — agents register/verify public keys so shared experiences carry cryptographic provenance.
- **Rate limiting** — token-bucket throttling per route via `governor`.

## Quick Start

```bash
# Build the workspace
cargo build --release

# Run the HTTP server (example binary)
cargo run --example server

# Or run the MCP stdio server (for an MCP host / agent runtime)
cargo run --bin oris-experience-mcp
```

### Configuration (env)

| Variable | Default | Description |
|----------|---------|-------------|
| `ORIS_EXPERIENCE_DB` | `.oris/experience_repo.db` | Experience store SQLite path |
| `ORIS_EXPERIENCE_KEY_DB` | `.oris/experience_keys.db` | API-key store SQLite path |
| `ORIS_AGENT_ID` | `local-agent` | Agent identity for MCP auth |
| `ORIS_MCP_SCOPES` | `experience:read,experience:write` | Comma-separated scopes (`*` = all) |

### HTTP API endpoints

```
GET    /health
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

## Architecture

This product is a self-contained Rust workspace of three crates with **zero dependencies on the rest of the Oris monorepo**:

| Crate | Role | Internal deps |
|-------|------|---------------|
| `oris-experience-repo` | HTTP + MCP server, control plane, key/PKI services, client | `oris-genestore`, `oris-experience-contract` |
| `oris-experience-contract` | Canonical cross-agent experience types (`GeneV1`, `CapsuleV1`, `ExperienceBundleV1`, `UsageReceiptV1`) | none |
| `oris-genestore` | SQLite gene/capsule store, migrations, replay hooks | none |

```
oris-experience-repo
├── oris-genestore            (0 oris deps)
└── oris-experience-contract  (0 oris deps)
```

The dependency graph is acyclic and closed: the two leaf crates have no path dependencies, and `oris-experience-repo` references them only via relative workspace paths. This makes the set trivially extractable as a standalone product.

### Test fixtures

A few `#[cfg(test)]` blocks embed golden fixtures via `include_str!` from paths outside the three crates (`spec/experience/golden/experience-bundle-v1.json` and `plugins/oris-experience/capabilities.json`). These are test-only references — the product library and binaries have no such external references. The extraction script detects and bundles these fixtures so `cargo check`/`cargo test` work standalone; they do not affect the library or server at runtime.

## Extraction (standalone repo)

Product A can be extracted into a standalone repository using the extraction script located in the monorepo at `scripts/extract-product-a.sh`. The script copies the three crates, generates a fresh workspace `Cargo.toml` and `.gitignore`, copies this README, and runs `cargo check` to verify the result compiles on its own.

From the Oris monorepo root:

```bash
# Extract to a sibling directory (default: ../oris-memory-server)
./scripts/extract-product-a.sh

# Or extract to an explicit path
./scripts/extract-product-a.sh /path/to/oris-memory-server
```

The script refuses to overwrite an existing target directory and exits non-zero if `cargo check` fails, so a clean exit guarantees a buildable standalone repo.
