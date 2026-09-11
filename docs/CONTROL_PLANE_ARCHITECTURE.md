# Oris Memory Server — 控制面架构设计

> 目标：将当前 Experience Repository 演进为文档定义的 **Enterprise Context & Memory Service** 控制面。

---

## 1. 定位与边界

| 属性 | 值 |
|------|----|
| 产品定位 | Enterprise Context & Memory Service（企业自有、模型无关的统一控制面） |
| 不是什么 | 不是 OpenClaw/DeerFlow Runtime；不是 Mem0/Cognee 替代品；不直接连 Source of Truth |
| 核心职责 | Canonical User Memory、Shared Task Memory、Context Router、权限治理、记忆晋级、审计遗忘 |
| 数据底座 | PostgreSQL + pgvector（权威存储）+ Redis（热上下文）|
| 可插拔引擎 | Mem0（个人长期推断）、Cognee（企业语义关系）、Graphiti（时间演化）|

模型调用顺序：`Agent → Context Assembly → Model Router → Model`

---

## 2. Crate 重组

当前 3 crate → 目标 4 crate：

```
oris-memory-server/
├── crates/
│   ├── oris-memory-contract/       # 全量类型契约（重命名自 oris-experience-contract）
│   │   ├── memory_item.rs           # memory_item, MemoryType, Scope, AuthorityLevel
│   │   ├── canonical_user.rs        # CanonicalUserProfile, IdentityLink, Preference
│   │   ├── shared_task.rs           # SharedTaskContext, TaskHandoff, TaskStep
│   │   ├── decision.rs              # DecisionRecord, Option, Recommendation
│   │   ├── entity.rs                # Entity, EntityRelation
│   │   ├── experience.rs            # GeneV1, CapsuleV1, UsageReceiptV1（保留现有）
│   │   ├── context.rs               # ContextPackage, ContextReference, TokenBudget
│   │   ├── governance.rs            # Policy, Acl, AuditEntry, RetentionPolicy, ForgetRequest
│   │   └── events.rs                # OutboxEvent, MemoryEvent 枚举
│   ├── oris-memory-store/           # 数据面（重命名自 oris-genestore）
│   │   ├── postgres/
│   │   │   ├── schema.rs            # DDL + 迁移
│   │   │   ├── memory_repo.rs       # memory_item CRUD
│   │   │   ├── user_repo.rs         # canonical_user_profile CRUD
│   │   │   ├── task_repo.rs         # shared_task_context CRUD
│   │   │   ├── search.rs            # 混合检索（结构化 + 关键词 + pgvector）
│   │   │   ├── outbox.rs            # outbox_event 读写
│   │   │   └── rls.rs               # Row Level Security 策略
│   │   ├── redis/
│   │   │   ├── hot_context.rs       # 热上下文物化
│   │   │   ├── cache.rs             # 结果缓存
│   │   │   └── session.rs           # 会话状态、分布式锁
│   │   └── traits.rs                # 可插拔引擎适配 trait
│   ├── oris-control-plane/          # 控制面逻辑（重命名/扩展自 oris-experience-repo）
│   │   ├── identity.rs              # Identity Resolver（SSO/IAM 委托身份）
│   │   ├── context_router.rs        # 意图路由 + 延迟预算
│   │   ├── context_assembler.rs     # 上下文装配 + 压缩 + Token 预算
│   │   ├── write_pipeline.rs        # 候选→扫描→评分→去重→冲突→策略→存储
│   │   ├── retrieval.rs             # 混合召回 + Rerank
│   │   ├── reflection.rs            # Episode→Experience→Pattern→SOP/Skill
│   │   ├── promotion.rs            # scope 晋级（personal→team→factory→enterprise）
│   │   ├── governance/
│   │   │   ├── policy.rs            # 保留策略、有效期、隐私分类
│   │   │   ├── acl.rs              # RBAC + ABAC 权限判定
│   │   │   ├── audit.rs            # 访问审计
│   │   │   └── retention.rs         # 生命周期管理（decay→archive→delete）
│   │   ├── outbox_worker.rs         # 异步事件处理 + 缓存失效
│   │   └── poison_guard.rs         # Memory Poisoning 检测
│   └── oris-memory-server/         # HTTP + MCP 对外接口
│       ├── http/                    # REST API（含现有 experience 端点）
│       ├── mcp/                     # MCP JSON-RPC（含现有 experience 工具）
│       └── main.rs                  # 入口
```

依赖图（无环）：

```
oris-memory-server
  ├── oris-control-plane
  │     ├── oris-memory-store
  │     └── oris-memory-contract
  ├── oris-memory-store
  └── oris-memory-contract

oris-memory-store
  └── oris-memory-contract
```

---

## 3. 数据模型（PostgreSQL）

### 3.1 核心表

```sql
-- 统一记忆表：memory_type 是字段，不是数据库
CREATE TABLE memory_item (
    memory_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id        TEXT NOT NULL,
    memory_type      TEXT NOT NULL,   -- semantic|episodic|decision|experience|user_preference
    scope            TEXT NOT NULL,   -- personal|agent|task|team|process|factory|enterprise
    subject_type     TEXT,
    subject_id       TEXT,
    entity_refs      JSONB DEFAULT '[]',
    content          TEXT,
    structured_payload JSONB,
    embedding        vector(1536),    -- pgvector HNSW
    source_type      TEXT NOT NULL,   -- iam|hr|user_explicit|agent_inferred|tool_result|business_event
    source_reference TEXT,
    evidence_refs    JSONB DEFAULT '[]',
    confidence       REAL DEFAULT 0.5,
    authority_level  TEXT NOT NULL,   -- L0_source_of_truth|L1_authoritative|L2_verified|L3_inferred
    importance       REAL DEFAULT 0.5,
    observed_at      TIMESTAMPTZ,
    valid_from       TIMESTAMPTZ,
    valid_to         TIMESTAMPTZ,
    privacy_class    TEXT NOT NULL,   -- public|internal|confidential|restricted
    acl              JSONB DEFAULT '{}',
    retention_policy TEXT,
    status           TEXT DEFAULT 'active', -- candidate|active|archived|revoked|quarantined
    version          INTEGER DEFAULT 1,
    derived_from     JSONB DEFAULT '[]',
    created_by_user  TEXT,
    created_by_agent TEXT,
    last_verified_at TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW 向量索引 + 分区
CREATE INDEX idx_memory_embedding ON memory_item
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_memory_filter ON memory_item (tenant_id, scope, memory_type, status, valid_to);
CREATE INDEX idx_memory_subject ON memory_item (subject_type, subject_id);
```

### 3.2 Canonical User Profile

```sql
CREATE TABLE canonical_user_profile (
    user_id          TEXT PRIMARY KEY,
    organization_id  TEXT NOT NULL,
    factory_id       TEXT,
    identity_links   JSONB DEFAULT '[]',   -- [{system, external_id, verified}]
    role             TEXT,
    position         TEXT,
    language         TEXT,
    timezone         TEXT,
    preferences      JSONB DEFAULT '{}',    -- 显式偏好（用户确认 > Agent 设置 > 领域默认 > 全局默认）
    common_entities  JSONB DEFAULT '[]',
    active_projects  JSONB DEFAULT '[]',
    consent_scope    JSONB DEFAULT '{}',
    privacy_class    TEXT DEFAULT 'internal',
    source           TEXT NOT NULL,         -- iam|hr|user_explicit
    authority_level  TEXT DEFAULT 'L1_authoritative',
    version          INTEGER DEFAULT 1,
    valid_from       TIMESTAMPTZ,
    valid_to         TIMESTAMPTZ,
    last_verified_at TIMESTAMPTZ,
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);
```

### 3.3 Shared Task Context

```sql
CREATE TABLE shared_task_context (
    task_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_task_id    UUID,
    initiator_user_id TEXT NOT NULL,
    organization_scope TEXT NOT NULL,
    goal               TEXT NOT NULL,
    constraints        JSONB DEFAULT '[]',
    success_criteria   JSONB DEFAULT '[]',
    entities           JSONB DEFAULT '[]',
    business_refs      JSONB DEFAULT '[]',
    current_findings   JSONB DEFAULT '[]',
    evidence_refs      JSONB DEFAULT '[]',
    decisions          JSONB DEFAULT '[]',
    assumptions        JSONB DEFAULT '[]',
    completed_steps    JSONB DEFAULT '[]',
    pending_steps      JSONB DEFAULT '[]',
    current_owner_agent TEXT,
    participant_agents  JSONB DEFAULT '[]',
    artifact_refs       JSONB DEFAULT '[]',
    source_system_refs  JSONB DEFAULT '[]',
    status              TEXT DEFAULT 'active',
    version             INTEGER DEFAULT 1,
    expires_at          TIMESTAMPTZ,
    acl                 JSONB DEFAULT '{}',
    privacy_class       TEXT DEFAULT 'internal',
    audit_ref           TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
```

### 3.4 其他表

```sql
-- 决策记录
CREATE TABLE decision_record (
    decision_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id          UUID REFERENCES shared_task_context(task_id),
    situation       TEXT NOT NULL,
    options          JSONB DEFAULT '[]',
    recommendation  TEXT,
    human_decision  TEXT,
    reason           TEXT,
    action           TEXT,
    outcome          TEXT,
    evidence_refs    JSONB DEFAULT '[]',
    decided_at       TIMESTAMPTZ,
    outcome_at       TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- 实体与关系（一期 PostgreSQL 表达，非 RDF/Graph）
CREATE TABLE entity (
    entity_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   TEXT NOT NULL,
    entity_type TEXT NOT NULL,   -- product|bom|material|supplier|process|equipment|line|batch|quality_event|maintenance_event
    name        TEXT NOT NULL,
    attributes  JSONB DEFAULT '{}',
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE entity_relation (
    relation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_entity UUID REFERENCES entity(entity_id),
    to_entity   UUID REFERENCES entity(entity_id),
    relation_type TEXT NOT NULL,
    attributes  JSONB DEFAULT '{}',
    valid_from  TIMESTAMPTZ,
    valid_to    TIMESTAMPTZ,
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 访问审计
CREATE TABLE memory_access_audit (
    audit_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id   UUID,
    accessor_user   TEXT,
    accessor_agent  TEXT,
    action      TEXT NOT NULL,   -- read|write|promote|revoke|forget|verify
    purpose     TEXT,
    task_id     UUID,
    trace_id    TEXT,
    policy_decision_id TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 版本追溯
CREATE TABLE memory_version (
    version_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id   UUID NOT NULL,
    version     INTEGER NOT NULL,
    payload     JSONB NOT NULL,
    changed_by  TEXT NOT NULL,
    change_reason TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(memory_id, version)
);

-- Outbox（事务性事件）
CREATE TABLE outbox_event (
    event_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type  TEXT NOT NULL,   -- USER_CONTEXT_UPDATED|PREFERENCE_UPDATED|TASK_CONTEXT_UPDATED|MEMORY_PROMOTED|MEMORY_REVOKED|MEMORY_EXPIRED|PERMISSION_CHANGED|ENGINE_PROJECTION_FAILED
    aggregate_id TEXT NOT NULL,
    payload     JSONB NOT NULL,
    status      TEXT DEFAULT 'pending',  -- pending|processing|done|failed
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);
```

### 3.5 Row Level Security

```sql
ALTER TABLE memory_item ENABLE ROW LEVEL SECURITY;
ALTER TABLE canonical_user_profile ENABLE ROW LEVEL SECURITY;
ALTER TABLE shared_task_context ENABLE ROW LEVEL SECURITY;

-- 策略：tenant 隔离 + scope 过滤
CREATE POLICY tenant_isolation ON memory_item
    USING (tenant_id = current_setting('app.tenant_id'));
CREATE POLICY tenant_isolation ON canonical_user_profile
    USING (organization_id = current_setting('app.tenant_id'));
```

---

## 4. 控制面模块设计

### 4.1 Identity Resolver

```rust
pub struct IdentityResolver {
    iam_client: Arc<dyn IamClient>,      // SSO/IAM/HR 主数据
    redis: Arc<RedisPool>,                // 权限快照缓存
}

pub struct ResolvedIdentity {
    pub user_id: String,
    pub organization_id: String,
    pub factory_id: Option<String>,
    pub roles: Vec<String>,
    pub delegated_agent_id: Option<String>,
    pub permissions: PermissionSet,
    pub trace_id: String,
}

impl IdentityResolver {
    pub async fn resolve(&self, token: &str, agent_id: &str, purpose: &str)
        -> Result<ResolvedIdentity>;
}
```

### 4.2 Context Router

```rust
pub enum RequestIntent {
    Chat,              // 闲聊 — 仅热上下文
    PersonalTask,      // 个人事务 — 热上下文 + 用户记忆
    BusinessQuery,     // 业务查询 — + 结构化检索
    CrossAgentTask,    // 跨 Agent — + Shared Task Context
    DecisionSupport,   // 决策支持 — + 经验记忆 + SoT 回查
    DeepResearch,      // 深度研究 — + 语义/图检索
}

pub struct RoutingPlan {
    pub intent: RequestIntent,
    pub hot_context: bool,
    pub canonical_user: bool,
    pub shared_task: bool,
    pub structured_search: bool,
    pub vector_search: bool,
    pub graph_search: bool,       // 可选引擎
    pub sot_verification: bool,
    pub latency_budget_ms: u64,
    pub token_budget: usize,
}
```

### 4.3 Context Assembler

```rust
pub struct ContextAssembler {
    token_budget: usize,          // 2K-8K
    compression_strategy: CompressionStrategy,
}

pub struct AssembledContext {
    pub identity: IdentityContext,
    pub user_context: UserContext,      // explicit + inferred projection
    pub task_context: Option<TaskContext>,
    pub agent_private: Option<String>,
    pub enterprise_memory: Vec<MemorySlice>,
    pub current_business_state: Option<BusinessState>,
    pub total_tokens: usize,
    pub conflict_flags: Vec<ConflictFlag>,
}

impl ContextAssembler {
    pub async fn assemble(
        &self,
        identity: &ResolvedIdentity,
        plan: &RoutingPlan,
        store: &MemoryStore,
    ) -> Result<AssembledContext>;
}
```

### 4.4 Memory Write Pipeline

```rust
pub struct WritePipeline {
    poison_guard: PoisonGuard,
    importance_scorer: ImportanceScorer,
    dedup: DedupEngine,
    conflict_resolver: ConflictResolver,
    policy_engine: PolicyEngine,
}

impl WritePipeline {
    pub async fn remember(
        &self,
        candidate: MemoryCandidate,
        identity: &ResolvedIdentity,
    ) -> Result<WriteOutcome> {
        // 1. 候选提取
        // 2. 敏感数据 & 注入扫描（PoisonGuard）
        // 3. 重要性 / 新颖性 / 置信度评分
        // 4. 实体链接
        // 5. 去重 / 冲突 / 有效期检查
        // 6. 保留 & 共享策略判定
        // 7. 候选区存储 or 人工审核（高风险）
        // 8. 正式存储（同事务写 memory_item + outbox_event）
        // 9. 异步：embedding / 缓存失效 / 可选图同步
    }
}
```

### 4.5 PoisonGuard

```rust
pub struct PoisonGuard {
    injection_detector: InjectionDetector,  // prompt injection 检测
    sensitive_scanner: SensitiveScanner,     // 密码/Token/Credential 检测
}

impl PoisonGuard {
    pub fn scan(&self, content: &str, source: &SourceType)
        -> Result<SafetyVerdict>;
}
```

### 4.6 可插拔引擎适配

```rust
#[async_trait]
pub trait MemoryEngine: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> EngineCapabilities;
    async fn search(&self, query: &EngineQuery) -> Result<Vec<EngineResult>>;
    async fn write(&self, item: &MemoryItem) -> Result<()>;
    async fn delete(&self, id: &str) -> Result<()>;
    async fn health(&self) -> Result<EngineHealth>;
}

pub struct EngineRegistry {
    engines: HashMap<String, Arc<dyn MemoryEngine>>,
    circuit_breakers: HashMap<String, CircuitBreaker>,
}
```

---

## 5. API 面扩展

### 5.1 新增 REST 端点（文档 10.2）

```
POST   /v1/memories/candidates          # 提交候选记忆
GET    /v1/memories/{id}                 # 获取单条
POST   /v1/memories/search               # 混合检索
POST   /v1/context/assemble             # 装配上下文
GET    /v1/users/{id}/canonical-context  # 用户统一上下文
GET    /v1/tasks/{id}/context            # 任务上下文
PATCH  /v1/tasks/{id}/context            # 更新任务上下文
POST   /v1/memories/{id}/promote         # scope 晋级
POST   /v1/memories/{id}/verify           # 回查来源
POST   /v1/memories/forget               # 遗忘
GET    /v1/memories/{id}/lineage          # 版本链路
```

### 5.2 保留现有端点

```
GET/POST /v1/experience-assets           # 经验资产（memory_type=experience）
GET    /v1/experience-assets/{id}/skill  # Skill 投影
POST   /v1/experience-assets/{id}/use     # 使用记录
POST   /v1/experience-assets/{id}/outcomes
POST   /v1/experience-assets/{id}/promote
POST   /v1/experience-assets/{id}/revoke
```

### 5.3 MCP 工具扩展

新增：
```
oris_memory_remember      # 记忆写入
oris_memory_recall       # 按权限召回
oris_memory_search       # 混合检索
oris_memory_get_context  # 装配上下文
oris_memory_update       # 版本化更新
oris_memory_forget       # 遗忘
oris_memory_share        # 跨 Agent 共享
oris_memory_promote      # scope 晋级
oris_memory_reflect      # 经验提炼
oris_memory_verify       # 回查来源
oris_user_get_context    # 用户上下文
oris_task_get_context    # 任务上下文
oris_task_update_context # 任务更新
```

保留：`oris_experience_*` 工具族

---

## 6. 延迟预算

| 阶段 | 目标 P95 |
|------|----------|
| 身份与权限解析 | 10-20ms |
| Context Router | 5-10ms |
| Redis 热上下文 | 5-10ms |
| PostgreSQL 混合检索 | 50-120ms |
| 可选引擎并行召回 | 50-200ms |
| Rerank / Conflict | 20-50ms |
| Context Assemble | 10-30ms |
| 热上下文总延迟 | ≤60ms |
| 标准召回总延迟 | ≤200ms |
| 深度召回 | 200-800ms（仅 DecisionSupport/DeepResearch） |

---

## 7. 分阶段实施计划

### 阶段 0：架构定版（第 0-4 周）

- [ ] 确定 crate 重组方案，完成重命名与依赖图调整
- [ ] 定义 `oris-memory-contract` 全量类型（memory_item、canonical_user、shared_task、decision、entity、context、governance、events）
- [ ] 设计 PostgreSQL schema + RLS 策略
- [ ] 定义 API V1 契约（REST + MCP 工具）
- [ ] 确定 OpenClaw / DeerFlow 适配方案
- [ ] 制定 Mem0 / Cognee PoC 假设、对照组、退出条件

### 阶段 1：MVP 基础能力（第 5-12 周）

- [ ] PostgreSQL + pgvector + Redis 基础设施搭建
- [ ] `oris-memory-store`：PostgreSQL memory_repo + user_repo + task_repo + search（混合检索）
- [ ] `oris-memory-store`：Redis hot_context + cache
- [ ] `oris-control-plane`：Identity Resolver + Context Router + Context Assembler
- [ ] `oris-control-plane`：Memory Write Pipeline + PoisonGuard
- [ ] `oris-control-plane`：governance/policy + acl + audit
- [ ] `oris-control-plane`：Outbox Worker + 缓存失效
- [ ] `oris-memory-server`：新 REST 端点 + MCP 工具
- [ ] 保留并适配现有 experience 端点（memory_type=experience）
- [ ] 用户查看 / 纠正 / 遗忘基础能力

**退出标准**：一个个人助手 + 一个跨 Agent 任务端到端跑通。

### 阶段 2：经验记忆与业务闭环（第 13-24 周）

- [ ] Episodic / Decision Memory
- [ ] 设备故障→处置→结果闭环
- [ ] 质量缺陷→根因→8D→效果闭环
- [ ] 冲突检测、来源可信度、结果评分
- [ ] Reflection / Consolidation 后台作业
- [ ] 跨 Agent Context Package 与任务交接
- [ ] Mem0 个人长期记忆 PoC（vs PostgreSQL/pgvector 基线）
- [ ] Cognee 制造语义/关系记忆 PoC（vs PostgreSQL entity_relation + pgvector 基线）
- [ ] 压测、红队、灾备演练

**价值闸门**：任务交接、经验复用、问题定位、权限零泄露达标后进入规模化。

### 阶段 3：规模化与引擎选型（第 7-9 个月）

- [ ] 多租户/多工厂分区、读副本、容量治理
- [ ] 供应链 Decision Memory
- [ ] 企业级 promotion 与最佳实践发布
- [ ] 达标的 Mem0/Cognee 以 Adapter 方式生产化
- [ ] Graphiti 对照 PoC（仅在双时间需求未满足时）

### 阶段 4：组织学习与 Skill 演进（第 10-12 个月）

- [ ] 集团/工厂/团队多级 namespace
- [ ] Experience → Rule → SOP → Skill 发布链路
- [ ] Memory + Digital Twin 验证闭环
- [ ] 统一运营看板、成本与质量优化

---

## 8. 现有代码迁移策略

| 现有模块 | 迁移方案 |
|---------|---------|
| `oris-experience-contract` | 重命名为 `oris-memory-contract`，保留 GeneV1/CapsuleV1/UsageReceiptV1，新增全量类型 |
| `oris-genestore` SqliteGeneStore | 废弃；新 `oris-memory-store` 用 PostgreSQL。旧 Gene/Capsule 通过 migration.rs 适配为 `memory_type=experience` |
| `oris-experience-repo` ExperienceControlPlane | 保留生命周期逻辑（candidate→stable→deprecated→quarantined→revoked），迁移为 `memory_item.status` 字段 |
| `oris-experience-repo` search（bm25_lite + hashed_cosine）| 替换为 PostgreSQL 全文检索 + pgvector HNSW |
| `oris-experience-repo` skill_projection | 保留，作为 Reflection & Consolidation 的输出 |
| `oris-experience-repo` network_types / oen | 统一为单一 Envelope 结构 |
| `oris-experience-repo` key_service | 保留，修复 scope 读取 bug，增加 RBAC 层 |
| `oris-experience-repo` server/handlers | 扩展为全量 REST API |
| `oris-experience-repo` mcp | 扩展 MCP 工具集 |

### 必须修复的 bug（P1，迁移前完成）

1. `ApiKeyInfo::from(&ApiKey)` 硬编码 scopes → 从 `api_key_scopes` 表加载
2. OEN 双 Envelope 定义 → 统一为单一结构
3. OenVerifier 签名缓存不失效 → 增加 TTL + 撤销检查
4. 搜索无租户强制隔离 → 默认拒绝跨租户
