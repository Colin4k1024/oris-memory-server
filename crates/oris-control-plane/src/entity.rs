//! Manufacturing Entity & Entity Relation Manager.
//!
//! Manages the manufacturing entity graph — products, BOMs, materials,
//! suppliers, processes, equipment, lines, batches, and quality/maintenance
//! events — plus the typed relations that connect them (contains, produced_by,
//! supplied_by, maintained_by, located_at, part_of, caused_by, resolved_by,
//! depends_on).
//!
//! The graph is stored across two PostgreSQL tables (`entity` and
//! `entity_relation`) defined in [`oris_memory_store::postgres::schema`].
//! This module provides full CRUD plus two graph-traversal helpers:
//! bounded BFS [`EntityManager::traverse`] and shortest-path
//! [`EntityManager::shortest_path`].
//!
//! # Architecture
//!
//! The graph algorithms ([`bfs_traverse`] and [`bfs_shortest_path`]) are pure
//! functions operating on an in-memory adjacency list. The async
//! [`EntityManager`] methods query the database to build that adjacency list,
//! then delegate to the pure functions. This keeps the BFS logic
//! unit-testable without a live PostgreSQL connection.

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::{DateTime, Utc};
use oris_memory_store::postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::Row;
use thiserror::Error;
use tracing::instrument;
use uuid::Uuid;

// ──────────────────────────── Errors ────────────────────────────

/// Errors that can occur during entity or relation operations.
#[derive(Debug, Error)]
pub enum EntityError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("entity {0} not found")]
    NotFound(Uuid),

    #[error("unknown entity type: {0}")]
    InvalidEntityType(String),

    #[error("unknown relation type: {0}")]
    InvalidRelationType(String),
}

// ──────────────────────────── Domain Types ────────────────────────────

/// The kind of manufacturing entity.
///
/// Serialized as `snake_case` to match the `entity_type` column values
/// documented in the schema (`product`, `quality_event`, …).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Product,
    Bom,
    Material,
    Supplier,
    Process,
    Equipment,
    Line,
    Batch,
    QualityEvent,
    MaintenanceEvent,
}

impl EntityType {
    /// Return the canonical `snake_case` database string.
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Product => "product",
            EntityType::Bom => "bom",
            EntityType::Material => "material",
            EntityType::Supplier => "supplier",
            EntityType::Process => "process",
            EntityType::Equipment => "equipment",
            EntityType::Line => "line",
            EntityType::Batch => "batch",
            EntityType::QualityEvent => "quality_event",
            EntityType::MaintenanceEvent => "maintenance_event",
        }
    }

    /// Parse a database string into an [`EntityType`].
    ///
    /// Returns `None` for unrecognized values.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "product" => Some(EntityType::Product),
            "bom" => Some(EntityType::Bom),
            "material" => Some(EntityType::Material),
            "supplier" => Some(EntityType::Supplier),
            "process" => Some(EntityType::Process),
            "equipment" => Some(EntityType::Equipment),
            "line" => Some(EntityType::Line),
            "batch" => Some(EntityType::Batch),
            "quality_event" => Some(EntityType::QualityEvent),
            "maintenance_event" => Some(EntityType::MaintenanceEvent),
            _ => None,
        }
    }
}

/// The kind of typed relation between two entities.
///
/// Serialized as `snake_case` to match the `relation_type` column.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    /// BOM contains material.
    Contains,
    /// Product produced by process.
    ProducedBy,
    /// Material supplied by supplier.
    SuppliedBy,
    /// Equipment maintained by.
    MaintainedBy,
    /// Equipment located at line.
    LocatedAt,
    /// Batch part of.
    PartOf,
    /// Quality event caused by.
    CausedBy,
    /// Event resolved by process.
    ResolvedBy,
    /// Process depends on equipment.
    DependsOn,
}

impl RelationType {
    /// Return the canonical `snake_case` database string.
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationType::Contains => "contains",
            RelationType::ProducedBy => "produced_by",
            RelationType::SuppliedBy => "supplied_by",
            RelationType::MaintainedBy => "maintained_by",
            RelationType::LocatedAt => "located_at",
            RelationType::PartOf => "part_of",
            RelationType::CausedBy => "caused_by",
            RelationType::ResolvedBy => "resolved_by",
            RelationType::DependsOn => "depends_on",
        }
    }

    /// Parse a database string into a [`RelationType`].
    ///
    /// Returns `None` for unrecognized values.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "contains" => Some(RelationType::Contains),
            "produced_by" => Some(RelationType::ProducedBy),
            "supplied_by" => Some(RelationType::SuppliedBy),
            "maintained_by" => Some(RelationType::MaintainedBy),
            "located_at" => Some(RelationType::LocatedAt),
            "part_of" => Some(RelationType::PartOf),
            "caused_by" => Some(RelationType::CausedBy),
            "resolved_by" => Some(RelationType::ResolvedBy),
            "depends_on" => Some(RelationType::DependsOn),
            _ => None,
        }
    }
}

/// A manufacturing entity row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub entity_id: Uuid,
    pub tenant_id: String,
    pub entity_type: EntityType,
    pub name: String,
    pub attributes: Value,
    pub source: String,
    pub created_at: DateTime<Utc>,
}

/// A typed relation between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    pub relation_id: Uuid,
    pub from_entity: Uuid,
    pub to_entity: Uuid,
    pub relation_type: RelationType,
    pub attributes: Value,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub source: String,
    pub created_at: DateTime<Utc>,
}

/// Request payload for creating a new entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEntityRequest {
    pub tenant_id: String,
    pub entity_type: EntityType,
    pub name: String,
    #[serde(default = "default_json_object")]
    pub attributes: Value,
    pub source: String,
}

/// Request payload for creating a new relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRelationRequest {
    pub from_entity: Uuid,
    pub to_entity: Uuid,
    pub relation_type: RelationType,
    #[serde(default = "default_json_object")]
    pub attributes: Value,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub source: String,
}

fn default_json_object() -> Value {
    Value::Object(serde_json::Map::new())
}

// ──────────────────────────── EntityManager ────────────────────────────

/// Manager for the manufacturing entity graph.
///
/// Provides CRUD for entities and relations, plus bounded BFS traversal
/// and shortest-path queries over the (undirected) relation graph.
pub struct EntityManager {
    pool: Pool,
}

impl EntityManager {
    /// Create a new manager backed by the given connection pool.
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    // ── Entity CRUD ──

    /// Insert a new entity and return its generated `entity_id`.
    #[instrument(skip(self, req))]
    pub async fn create_entity(&self, req: &CreateEntityRequest) -> Result<Uuid, EntityError> {
        let entity_id = Uuid::new_v4();

        sqlx::query(
            r#"INSERT INTO entity (
                entity_id, tenant_id, entity_type, name, attributes, source, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, NOW())"#,
        )
        .bind(entity_id)
        .bind(&req.tenant_id)
        .bind(req.entity_type.as_str())
        .bind(&req.name)
        .bind(&req.attributes)
        .bind(&req.source)
        .execute(&self.pool)
        .await?;

        Ok(entity_id)
    }

    /// Fetch a single entity by ID.
    #[instrument(skip(self))]
    pub async fn get_entity(&self, entity_id: Uuid) -> Result<Option<Entity>, EntityError> {
        let row = sqlx::query(r#"SELECT * FROM entity WHERE entity_id = $1"#)
            .bind(entity_id)
            .fetch_optional(&self.pool)
            .await?;

        row.map(|r| map_row_to_entity(&r)).transpose()
    }

    /// List entities of a given type for a tenant, newest first.
    #[instrument(skip(self))]
    pub async fn list_by_type(
        &self,
        tenant_id: &str,
        entity_type: EntityType,
        limit: i64,
    ) -> Result<Vec<Entity>, EntityError> {
        let rows = sqlx::query(
            r#"SELECT * FROM entity
               WHERE tenant_id = $1 AND entity_type = $2
               ORDER BY created_at DESC
               LIMIT $3"#,
        )
        .bind(tenant_id)
        .bind(entity_type.as_str())
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(map_row_to_entity).collect()
    }

    /// Case-insensitive name search within a tenant.
    #[instrument(skip(self))]
    pub async fn search_entities(
        &self,
        tenant_id: &str,
        name_pattern: &str,
    ) -> Result<Vec<Entity>, EntityError> {
        let pattern = format!("%{name_pattern}%");

        let rows = sqlx::query(
            r#"SELECT * FROM entity
               WHERE tenant_id = $1 AND name ILIKE $2
               ORDER BY created_at DESC"#,
        )
        .bind(tenant_id)
        .bind(pattern)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(map_row_to_entity).collect()
    }

    /// Replace the `attributes` JSONB of an entity.
    #[instrument(skip(self, attributes))]
    pub async fn update_attributes(
        &self,
        entity_id: Uuid,
        attributes: Value,
    ) -> Result<(), EntityError> {
        let result = sqlx::query(r#"UPDATE entity SET attributes = $2 WHERE entity_id = $1"#)
            .bind(entity_id)
            .bind(&attributes)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            return Err(EntityError::NotFound(entity_id));
        }
        Ok(())
    }

    // ── Relation CRUD ──

    /// Insert a new relation and return its generated `relation_id`.
    #[instrument(skip(self, req))]
    pub async fn create_relation(&self, req: &CreateRelationRequest) -> Result<Uuid, EntityError> {
        let relation_id = Uuid::new_v4();

        sqlx::query(
            r#"INSERT INTO entity_relation (
                relation_id, from_entity, to_entity, relation_type,
                attributes, valid_from, valid_to, source, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())"#,
        )
        .bind(relation_id)
        .bind(req.from_entity)
        .bind(req.to_entity)
        .bind(req.relation_type.as_str())
        .bind(&req.attributes)
        .bind(req.valid_from)
        .bind(req.valid_to)
        .bind(&req.source)
        .execute(&self.pool)
        .await?;

        Ok(relation_id)
    }

    /// Return all relations touching `entity_id` (as either from or to).
    #[instrument(skip(self))]
    pub async fn get_relations(&self, entity_id: Uuid) -> Result<Vec<EntityRelation>, EntityError> {
        let rows = sqlx::query(
            r#"SELECT * FROM entity_relation
               WHERE from_entity = $1 OR to_entity = $1
               ORDER BY created_at DESC"#,
        )
        .bind(entity_id)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(map_row_to_relation).collect()
    }

    /// Return all relations of a given type for a tenant.
    ///
    /// Because `entity_relation` has no `tenant_id` column, the tenant
    /// filter is applied by joining on the `from_entity`'s tenant.
    #[instrument(skip(self))]
    pub async fn get_relations_by_type(
        &self,
        tenant_id: &str,
        relation_type: RelationType,
    ) -> Result<Vec<EntityRelation>, EntityError> {
        let rows = sqlx::query(
            r#"SELECT er.* FROM entity_relation er
               JOIN entity e ON e.entity_id = er.from_entity
               WHERE e.tenant_id = $1 AND er.relation_type = $2
               ORDER BY er.created_at DESC"#,
        )
        .bind(tenant_id)
        .bind(relation_type.as_str())
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(map_row_to_relation).collect()
    }

    // ── Graph queries ──

    /// Find all entities reachable from `start` within `max_depth` hops.
    ///
    /// The graph is treated as **undirected**: a relation between A and B
    /// can be traversed from either side. Returns `(entity_id, depth)`
    /// pairs where depth 0 is the `start` node itself.
    #[instrument(skip(self))]
    pub async fn traverse(
        &self,
        start: Uuid,
        max_depth: u32,
    ) -> Result<Vec<(Uuid, u32)>, EntityError> {
        let adjacency = self.build_adjacency(start, max_depth).await?;
        Ok(bfs_traverse(&adjacency, start, max_depth))
    }

    /// Find the shortest path between two entities using BFS.
    ///
    /// Returns `Some(path)` (including both endpoints) when a path exists,
    /// or `None` when the two entities are in disconnected components.
    #[instrument(skip(self))]
    pub async fn shortest_path(
        &self,
        from: Uuid,
        to: Uuid,
    ) -> Result<Option<Vec<Uuid>>, EntityError> {
        // Build the full connected component so the pure BFS has every
        // adjacency entry it might look up.
        let adjacency = self.build_adjacency(from, u32::MAX).await?;
        Ok(bfs_shortest_path(&adjacency, from, to))
    }

    /// Fetch the set of entities directly connected to `entity_id`
    /// (either direction), as a `Vec` for adjacency-list assembly.
    async fn fetch_neighbors(&self, entity_id: Uuid) -> Result<Vec<Uuid>, EntityError> {
        let rows = sqlx::query(
            r#"SELECT to_entity AS neighbor FROM entity_relation WHERE from_entity = $1
               UNION
               SELECT from_entity AS neighbor FROM entity_relation WHERE to_entity = $1"#,
        )
        .bind(entity_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .filter_map(|r| r.try_get::<Uuid, _>("neighbor").ok())
            .collect())
    }

    /// Build an adjacency list via bounded BFS from `start`.
    ///
    /// Fetches neighbours for every node at depth `0..max_depth` — exactly
    /// the entries that [`bfs_traverse`] will look up. When `max_depth` is
    /// `u32::MAX` the entire connected component is loaded.
    async fn build_adjacency(
        &self,
        start: Uuid,
        max_depth: u32,
    ) -> Result<HashMap<Uuid, Vec<Uuid>>, EntityError> {
        let mut adjacency: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        let mut visited: HashSet<Uuid> = HashSet::new();
        visited.insert(start);
        let mut frontier: Vec<Uuid> = vec![start];

        for _ in 0..max_depth {
            if frontier.is_empty() {
                break;
            }
            let mut next_frontier: Vec<Uuid> = Vec::new();
            for entity_id in &frontier {
                let neighbors = self.fetch_neighbors(*entity_id).await?;
                for n in &neighbors {
                    if visited.insert(*n) {
                        next_frontier.push(*n);
                    }
                }
                adjacency.insert(*entity_id, neighbors);
            }
            frontier = next_frontier;
        }
        Ok(adjacency)
    }
}

// ──────────────────────────── Row mappers ────────────────────────────

fn map_row_to_entity(row: &sqlx::postgres::PgRow) -> Result<Entity, EntityError> {
    let type_str: String = row.try_get("entity_type")?;
    let entity_type =
        EntityType::from_str(&type_str).ok_or_else(|| EntityError::InvalidEntityType(type_str))?;

    Ok(Entity {
        entity_id: row.try_get("entity_id")?,
        tenant_id: row.try_get("tenant_id")?,
        entity_type,
        name: row.try_get("name")?,
        attributes: row
            .try_get::<Value, _>("attributes")
            .unwrap_or_else(|_| default_json_object()),
        source: row.try_get("source")?,
        created_at: row.try_get("created_at")?,
    })
}

fn map_row_to_relation(row: &sqlx::postgres::PgRow) -> Result<EntityRelation, EntityError> {
    let type_str: String = row.try_get("relation_type")?;
    let relation_type = RelationType::from_str(&type_str)
        .ok_or_else(|| EntityError::InvalidRelationType(type_str))?;

    Ok(EntityRelation {
        relation_id: row.try_get("relation_id")?,
        from_entity: row.try_get("from_entity")?,
        to_entity: row.try_get("to_entity")?,
        relation_type,
        attributes: row
            .try_get::<Value, _>("attributes")
            .unwrap_or_else(|_| default_json_object()),
        valid_from: row.try_get("valid_from")?,
        valid_to: row.try_get("valid_to")?,
        source: row.try_get("source")?,
        created_at: row.try_get("created_at")?,
    })
}

/// Reconstruct the path from the start node to `to` using parent pointers.
fn reconstruct_path(parent: &HashMap<Uuid, Uuid>, to: Uuid) -> Vec<Uuid> {
    let mut path = vec![to];
    let mut node = to;
    while let Some(&p) = parent.get(&node) {
        path.push(p);
        node = p;
    }
    path.reverse();
    path
}

// ──────────────────────────── Pure graph helpers ────────────────────────────
//
// Extracted so the BFS logic is unit-testable without a database. The async
// `EntityManager` methods build an adjacency list from PostgreSQL, then
// delegate to these functions.

/// Bounded BFS traversal over an in-memory adjacency list.
///
/// Returns `(entity_id, depth)` pairs, depth 0 being the `start` node.
/// Missing adjacency entries are treated as "no neighbours".
fn bfs_traverse(
    adjacency: &HashMap<Uuid, Vec<Uuid>>,
    start: Uuid,
    max_depth: u32,
) -> Vec<(Uuid, u32)> {
    let mut visited: HashSet<Uuid> = HashSet::new();
    visited.insert(start);
    let mut result: Vec<(Uuid, u32)> = vec![(start, 0)];
    let mut frontier: Vec<Uuid> = vec![start];

    for depth in 0..max_depth {
        if frontier.is_empty() {
            break;
        }
        let mut next_frontier: Vec<Uuid> = Vec::new();
        for entity_id in &frontier {
            if let Some(neighbors) = adjacency.get(entity_id) {
                for neighbor in neighbors {
                    if visited.insert(*neighbor) {
                        next_frontier.push(*neighbor);
                        result.push((*neighbor, depth + 1));
                    }
                }
            }
        }
        frontier = next_frontier;
    }
    result
}

/// Shortest path (BFS) over an in-memory adjacency list.
///
/// Returns `Some(path)` (including both endpoints) when reachable, `None`
/// otherwise.
fn bfs_shortest_path(
    adjacency: &HashMap<Uuid, Vec<Uuid>>,
    from: Uuid,
    to: Uuid,
) -> Option<Vec<Uuid>> {
    if from == to {
        return Some(vec![from]);
    }

    let mut visited: HashSet<Uuid> = HashSet::new();
    let mut parent: HashMap<Uuid, Uuid> = HashMap::new();
    let mut queue: VecDeque<Uuid> = VecDeque::new();
    queue.push_back(from);
    visited.insert(from);

    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&current) {
            for neighbor in neighbors {
                if visited.insert(*neighbor) {
                    parent.insert(*neighbor, current);
                    if *neighbor == to {
                        return Some(reconstruct_path(&parent, to));
                    }
                    queue.push_back(*neighbor);
                }
            }
        }
    }
    None
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EntityType conversion tests ──

    #[test]
    fn entity_type_as_str_round_trips() {
        let all = [
            EntityType::Product,
            EntityType::Bom,
            EntityType::Material,
            EntityType::Supplier,
            EntityType::Process,
            EntityType::Equipment,
            EntityType::Line,
            EntityType::Batch,
            EntityType::QualityEvent,
            EntityType::MaintenanceEvent,
        ];
        for et in &all {
            assert_eq!(EntityType::from_str(et.as_str()), Some(*et));
        }
    }

    #[test]
    fn entity_type_from_str_rejects_unknown() {
        assert_eq!(EntityType::from_str("robot"), None);
        assert_eq!(EntityType::from_str(""), None);
        assert_eq!(EntityType::from_str("Product"), None); // case-sensitive
    }

    #[test]
    fn entity_type_serde_uses_snake_case() {
        let json = serde_json::to_string(&EntityType::QualityEvent).unwrap();
        assert_eq!(json, "\"quality_event\"");

        let et: EntityType = serde_json::from_str("\"maintenance_event\"").unwrap();
        assert_eq!(et, EntityType::MaintenanceEvent);
    }

    #[test]
    fn entity_type_as_str_matches_schema_values() {
        // The SQL comment lists the exact allowed values.
        let schema_values = [
            "product",
            "bom",
            "material",
            "supplier",
            "process",
            "equipment",
            "line",
            "batch",
            "quality_event",
            "maintenance_event",
        ];
        for sv in &schema_values {
            assert!(EntityType::from_str(sv).is_some(), "{sv} should parse");
        }
    }

    // ── RelationType conversion tests ──

    #[test]
    fn relation_type_as_str_round_trips() {
        let all = [
            RelationType::Contains,
            RelationType::ProducedBy,
            RelationType::SuppliedBy,
            RelationType::MaintainedBy,
            RelationType::LocatedAt,
            RelationType::PartOf,
            RelationType::CausedBy,
            RelationType::ResolvedBy,
            RelationType::DependsOn,
        ];
        for rt in &all {
            assert_eq!(RelationType::from_str(rt.as_str()), Some(*rt));
        }
    }

    #[test]
    fn relation_type_from_str_rejects_unknown() {
        assert_eq!(RelationType::from_str("connected_to"), None);
        assert_eq!(RelationType::from_str(""), None);
    }

    #[test]
    fn relation_type_serde_uses_snake_case() {
        let json = serde_json::to_string(&RelationType::ProducedBy).unwrap();
        assert_eq!(json, "\"produced_by\"");

        let rt: RelationType = serde_json::from_str("\"depends_on\"").unwrap();
        assert_eq!(rt, RelationType::DependsOn);
    }

    // ── Request default / serde tests ──

    #[test]
    fn create_entity_request_attributes_default_to_empty_object() {
        let json = r#"{"tenant_id":"acme","entity_type":"product","name":"Widget","source":"erp"}"#;
        let req: CreateEntityRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.attributes, default_json_object());
        assert_eq!(req.entity_type, EntityType::Product);
    }

    #[test]
    fn create_relation_request_deserializes_with_optional_fields() {
        let json = r#"{
            "from_entity":"00000000-0000-0000-0000-000000000001",
            "to_entity":"00000000-0000-0000-0000-000000000002",
            "relation_type":"contains",
            "source":"mes"
        }"#;
        let req: CreateRelationRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.relation_type, RelationType::Contains);
        assert!(req.valid_from.is_none());
        assert!(req.valid_to.is_none());
    }

    // ── BFS traverse tests (mock adjacency) ──

    fn uuid(n: u8) -> Uuid {
        // Deterministic UUIDs for test readability.
        Uuid::from_bytes([n; 16])
    }

    #[test]
    fn bfs_traverse_isolated_node() {
        let start = uuid(1);
        let adj = HashMap::new();
        let result = bfs_traverse(&adj, start, 5);
        assert_eq!(result, vec![(start, 0)]);
    }

    #[test]
    fn bfs_traverse_linear_chain() {
        // 1 → 2 → 3
        let a = uuid(1);
        let b = uuid(2);
        let c = uuid(3);
        let mut adj = HashMap::new();
        adj.insert(a, vec![b]);
        adj.insert(b, vec![a, c]);
        adj.insert(c, vec![b]);

        let result = bfs_traverse(&adj, a, 10);
        let map: HashMap<Uuid, u32> = result.into_iter().collect();
        assert_eq!(map.get(&a), Some(&0));
        assert_eq!(map.get(&b), Some(&1));
        assert_eq!(map.get(&c), Some(&2));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn bfs_traverse_respects_max_depth() {
        // 1 → 2 → 3 → 4
        let a = uuid(1);
        let b = uuid(2);
        let c = uuid(3);
        let d = uuid(4);
        let mut adj = HashMap::new();
        adj.insert(a, vec![b]);
        adj.insert(b, vec![a, c]);
        adj.insert(c, vec![b, d]);
        adj.insert(d, vec![c]);

        let result = bfs_traverse(&adj, a, 1);
        let map: HashMap<Uuid, u32> = result.into_iter().collect();
        assert!(map.contains_key(&a));
        assert!(map.contains_key(&b));
        assert!(!map.contains_key(&c));
        assert!(!map.contains_key(&d));
    }

    #[test]
    fn bfs_traverse_handles_cycles_without_infinite_loop() {
        // 1 ↔ 2 with a duplicate edge
        let a = uuid(1);
        let b = uuid(2);
        let mut adj = HashMap::new();
        adj.insert(a, vec![b, b]);
        adj.insert(b, vec![a]);

        let result = bfs_traverse(&adj, a, 100);
        let map: HashMap<Uuid, u32> = result.into_iter().collect();
        assert_eq!(map.len(), 2); // a + b only, no duplicates
    }

    #[test]
    fn bfs_traverse_diamond_graph() {
        //     b
        //    / \
        //   a   d
        //    \ /
        //     c
        let a = uuid(1);
        let b = uuid(2);
        let c = uuid(3);
        let d = uuid(4);
        let mut adj = HashMap::new();
        adj.insert(a, vec![b, c]);
        adj.insert(b, vec![a, d]);
        adj.insert(c, vec![a, d]);
        adj.insert(d, vec![b, c]);

        let result = bfs_traverse(&adj, a, 10);
        let map: HashMap<Uuid, u32> = result.into_iter().collect();
        assert_eq!(map.get(&a), Some(&0));
        assert_eq!(map.get(&b), Some(&1));
        assert_eq!(map.get(&c), Some(&1));
        assert_eq!(map.get(&d), Some(&2));
    }

    // ── BFS shortest path tests (mock adjacency) ──

    #[test]
    fn bfs_shortest_path_same_node() {
        let a = uuid(1);
        let adj = HashMap::new();
        let path = bfs_shortest_path(&adj, a, a);
        assert_eq!(path, Some(vec![a]));
    }

    #[test]
    fn bfs_shortest_path_direct_edge() {
        let a = uuid(1);
        let b = uuid(2);
        let mut adj = HashMap::new();
        adj.insert(a, vec![b]);
        adj.insert(b, vec![a]);

        let path = bfs_shortest_path(&adj, a, b);
        assert_eq!(path, Some(vec![a, b]));
    }

    #[test]
    fn bfs_shortest_path_multi_hop_picks_shortest() {
        // a - b - c - d  (3 hops)
        //  \_________/  (a also connects directly to d) => shortest is 1 hop
        let a = uuid(1);
        let b = uuid(2);
        let c = uuid(3);
        let d = uuid(4);
        let mut adj = HashMap::new();
        adj.insert(a, vec![b, d]);
        adj.insert(b, vec![a, c]);
        adj.insert(c, vec![b, d]);
        adj.insert(d, vec![a, c]);

        let path = bfs_shortest_path(&adj, a, d).unwrap();
        assert_eq!(path, vec![a, d]); // direct edge, not a→b→c→d
    }

    #[test]
    fn bfs_shortest_path_disconnected_returns_none() {
        let a = uuid(1);
        let b = uuid(2);
        let mut adj = HashMap::new();
        adj.insert(a, vec![]);
        adj.insert(b, vec![]);

        let path = bfs_shortest_path(&adj, a, b);
        assert_eq!(path, None);
    }

    #[test]
    fn bfs_shortest_path_long_chain() {
        // 1 → 2 → 3 → 4 → 5
        let nodes: Vec<Uuid> = (1..=5).map(uuid).collect();
        let mut adj = HashMap::new();
        for i in 0..nodes.len() {
            let mut neighbors = Vec::new();
            if i > 0 {
                neighbors.push(nodes[i - 1]);
            }
            if i + 1 < nodes.len() {
                neighbors.push(nodes[i + 1]);
            }
            adj.insert(nodes[i], neighbors);
        }

        let path = bfs_shortest_path(&adj, nodes[0], nodes[4]).unwrap();
        assert_eq!(path.len(), 5);
        assert_eq!(path.first(), Some(&nodes[0]));
        assert_eq!(path.last(), Some(&nodes[4]));
    }

    #[test]
    fn reconstruct_path_builds_correct_order() {
        // parent: b→a, c→b  => path a,b,c
        let a = uuid(1);
        let b = uuid(2);
        let c = uuid(3);
        let mut parent = HashMap::new();
        parent.insert(b, a);
        parent.insert(c, b);

        let path = reconstruct_path(&parent, c);
        assert_eq!(path, vec![a, b, c]);
    }
}
