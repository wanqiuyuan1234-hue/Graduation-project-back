# 对话链路实现说明（检索 → 组装 → 模型回答 → 前端展示）

对应设计文档：[chat-rag-design.md](./chat-rag-design.md) 第 5 节决策。

本次落地的是系统最后一块功能：把已有的「代码入库 + 检索」接到对话上，形成可用的端到端问答。
所有已确认的决策均按原样实现：向量 + 标识符关键词 RRF 融合（不重排）、通用 OpenAI 兼容 LLM 配置层、SSE 流式、会话内锁定仓库、服务端持久化会话、三档多轮改写（默认 rule）。

---

## 1. 后端

### 1.1 新增文件

| 文件                            | 作用                                                                                                                                         |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/services/rag/__init__.py`  | RAG 服务包声明（检索 / 上下文 / LLM / 改写）。                                                                                               |
| `app/services/rag/retriever.py` | 混合检索核心：向量通道 + 关键词通道 → RRF 融合 → 阈值过滤 → 父子去重 → module_id 上溯 → 超长切片展开/截断 → 字符预算。                       |
| `app/services/rag/context.py`   | 系统提示词、来源块格式化（含 `[n]` 编号、仓库/文件/行号/分数）、token 估算、`build_messages`（系统提示 + 最近 6 轮历史 + 来源 + 当前问题）。 |
| `app/services/rag/llm.py`       | OpenAI 兼容协议的极简客户端：`stream_chat`（SSE 增量）与 `complete_chat`（非流式），只依赖 `httpx`，不引入 `openai` SDK。                    |
| `app/services/rag/rewrite.py`   | 多轮改写：`none` / `rule`（指代词 + 上轮标识符补全）/ `llm`（模型改写，失败降级为 rule）。                                                   |
| `app/schemas/chat.py`           | 对话相关出入参：`SourceOut`、`RetrieveRequest/Response`、`RetrieveDebug`、`ChatRequest/Response`、会话与消息模型。                           |
| `app/routers/chat.py`           | 路由：`/api/retrieve`、`/api/chat/config`、`/api/chat`（SSE / JSON）、`/api/chat/sessions` CRUD。                                            |
| `scripts/migrate_chat.py`       | 建表脚本：`chat_session`、`chat_message`，可重复执行。                                                                                       |

### 1.2 修改文件

| 文件                   | 改动                                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/config.py`        | 新增检索参数（candidate_k/top_k/权重/阈值/字符上限）、改写模式、聊天并发、LLM 三项必填 + 超时/温度/最大 token，以及 `llm_configured` 属性。 |
| `app/models.py`        | 新增 `ChatSession`、`ChatMessage` 两个 ORM 模型（含索引与级联）。                                                                           |
| `app/main.py`          | 注册 `chat.router`；CORS 增加 `PATCH` / `DELETE`；启动时后台线程预热 embedding 模型与 Chroma 集合，避免首问等待 17s。                       |
| `app/routers/query.py` | 重写为薄封装：`POST /api/query` 复用 `_chat_complete`（`rewrite_mode="none"`、`stream=false`），保持旧契约可用。                            |
| `.env.example`         | 补全 LLM、检索、改写、并发配置及厂商地址示例（不含真实密钥）。                                                                              |

### 1.3 关键设计点

- **进程内单例**：`retriever.get_collection()` 用锁保护 Chroma 集合；embedding 模型同样复用单例，满足 2G2核 约束。
- **流式持久化**：`_persist_assistant` 独立创建 `SessionLocal`，不依赖请求级会话；在 `finally` 中补齐「客户端中断」时已生成的部分内容。
- **并发限制**：`asyncio.Semaphore(chat_max_concurrency)` 只包裹 LLM 流式调用，检索不受限。
- **仓库锁定**：单仓库在会话创建时写入 `chat_session.repo_id`，之后请求携带的 `repo_ids` 被忽略；`repo_id` 为空则始终为「所有仓库」。
- **无来源兜底**：检索为空时不调用模型，直接返回固定说明并标记 `finish_reason="no_sources"`。
- **引用校验**：回答中出现超出 sources 数量的 `[n]` 时，`done` 事件返回 `citation_warning=true`，前端给出提示。

---

## 2. 前端

工程路径：`c:\no_space_demand\Graduation-project-front`（与后端仓库分离）。

### 2.1 新增文件

| 文件                  | 作用                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `src/lib/api/chat.ts` | 后端契约类型、会话 CRUD、会话列表缓存（`useSyncExternalStore`）、`streamChat`（fetch + ReadableStream 解析 SSE）。 |

### 2.2 修改文件

| 文件                                            | 改动                                                                                                                                                    |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lib/api/client.ts`                         | 新增 `apiPatch`、`apiDelete`；`API_BASE` 支持 `NEXT_PUBLIC_API_BASE`。                                                                                  |
| `src/lib/rag/types.ts`                          | `SourceChunk` 增加 `chunkId/repoName/fileId/truncated` 与更细的 `chunkType`；`ChatSession` 增加 `repoName/messageCount/updatedAt`；新增 `RewriteMode`。 |
| `src/lib/rag/store.ts`                          | 移除 mock 会话与 slim/fatten；新增 `rewriteMode`、`setRewriteMode`；`activeSessionId` 改为 `string                                                      |
| `src/components/rag/chat/chat-page.tsx`         | 全量接真实接口：发送走 SSE、会话切换加载历史、仓库选择与锁定、引用角标/跳转、检索调试面板、停止生成。                                                   |
| `src/components/rag/chat/session-sidebar.tsx`   | 改为纯 props 驱动，只做标题搜索，删除/重命名回调交给父组件。                                                                                            |
| `src/components/rag/shared/settings-dialog.tsx` | 新增「多轮改写」三选一（默认规则）；会话统计改为服务端数据；清空会话改为调用后端。                                                                      |
| `src/components/rag/shared/command-palette.tsx` | 会话来源改为服务端列表，显示 `messageCount`。                                                                                                           |
| `src/components/rag/shared/badges.tsx`          | `TypeBadge` 接受扩展后的 `chunkType`，非模块类型走兜底展示。                                                                                            |
| `src/components/rag/chat/retrieval-debug.tsx`   | `ChunkTypeBadge` 类型同步放宽。                                                                                                                         |

### 2.3 删除文件

- `src/app/api/chat/route.ts`、`src/app/api/retrieve/route.ts`：mock 路由已无必要，前端直连后端。
- `.next/dev/types/validator.ts`：Next 生成的过期 route 类型，引用已删除路由导致类型检查失败。

### 2.4 交互细节

- **仓库选择**：新会话可在「所有仓库 / 具体仓库」间选择，下方提示「首次提问后锁定」；已锁定会话顶部显示 `仓库名 + 锁图标 + 已锁定`，不可更改。
- **会话列表**：来自 `GET /api/chat/sessions`，按更新时间倒序；新建、切换、重命名、删除均实时同步。
- **历史加载**：切换会话时 `GET /api/chat/sessions/{id}`，消息里的 `citations` 直接还原引用角标与来源面板。
- **流式渲染**：先插入用户消息与空的助手占位，`delta` 逐段追加；`done` 后写入 token 与引用提示；「停止」通过 `AbortController` 中断，后端会保存已生成内容。
- **引用跳转**：有 `module_id` 时跳到模块详情并高亮行号；`fixed` 等无模块切片提示「该切片不支持跳转」。

---

## 3. 接口契约

后端默认 `http://localhost:8000`，前缀 `/api`。

### 3.1 `POST /api/retrieve`（只检索，不调模型）

请求：

```json
{ "query": "JsonAdapter fromJson 泛型", "repo_ids": [3], "top_k": 6, "candidate_k": 30 }
```

响应：

```json
{
  "query": "...",
  "effective_query": "...",
  "took_ms": 142,
  "strategy": "hybrid",
  "sources": [
    {
      "index": 1, "chunk_id": 24118, "module_id": 24118, "chunk_type": "method",
      "name": "fromJson", "repo_id": 3, "repo_name": "gson",
      "file_id": 512, "file_path": "gson/src/main/java/.../JsonAdapter.java",
      "language": "java", "start_line": 120, "end_line": 156,
      "score": 0.81, "content": "..."
    }
  ],
  "debug": { "vector_hits": 30, "keyword_hits": 12, "filtered": 2, "truncated": 1, "took_ms": 142, "strategy": "hybrid" }
}
```

### 3.2 `GET /api/chat/config`

```json
{ "llm_configured": true, "llm_model": "deepseek-chat", "default_rewrite_mode": "rule", "top_k": 6, "max_concurrency": 2 }
```

### 3.3 `POST /api/chat`

请求（`stream` 默认 `true`）：

```json
{
  "message": "JsonAdapter 怎么处理泛型",
  "session_id": null,
  "repo_ids": [3],
  "history": [{ "role": "user", "content": "..." }, { "role": "assistant", "content": "..." }],
  "top_k": 6,
  "stream": true,
  "rewrite_mode": "rule"
}
```

- `session_id` 为空 → 新建会话；单仓库会把范围写入会话并锁定。
- `rewrite_mode` 为空 → 用后端 `CHAT_REWRITE_MODE`。
- LLM 未配置 → `503`。
- `stream=false` → 返回 `{ content, sources, session_id, effective_query }`。

**SSE 事件**（`text/event-stream`，`event:` + JSON `data:`）：

| 事件      | 载荷                                                                                                |
| --------- | --------------------------------------------------------------------------------------------------- |
| `session` | `session_id, title, repo_id, repo_ids, created, rewrite_mode`                                       |
| `sources` | `sources[]`（同 3.1）、`retrieval`（debug 对象）                                                    |
| `delta`   | `text`                                                                                              |
| `done`    | `finish_reason, latency_ms, usage{prompt_tokens,completion_tokens}, rewrite_mode, citation_warning` |
| `error`   | `code, message`（模型调用失败时；随后不再发 `done`）                                                |

无检索结果时仍发 `session` / `sources`，随后 `delta` 给出固定说明，`done.finish_reason = "no_sources"`。

### 3.4 会话管理

| 方法   | 路径                      | 说明                                                    |
| ------ | ------------------------- | ------------------------------------------------------- |
| GET    | `/api/chat/sessions`      | 最近 100 个会话，含 `repo_name`、`message_count`        |
| POST   | `/api/chat/sessions`      | body `{ title?, repo_id? }`，`repo_id` 为空表示所有仓库 |
| GET    | `/api/chat/sessions/{id}` | `{ session, messages[] }`，消息带 `citations` 快照      |
| PATCH  | `/api/chat/sessions/{id}` | body `{ title }`                                        |
| DELETE | `/api/chat/sessions/{id}` | 消息随外键级联删除                                      |

### 3.5 `POST /api/query`（旧契约兼容）

`{ question, repo_ids?, top_k? }` → `{ content, sources, session_id }`，内部固定 `rewrite_mode="none"`、非流式。

---

## 4. 数据表与迁移

### 4.1 表结构

`chat_session`

| 字段                    | 类型         | 说明                                                |
| ----------------------- | ------------ | --------------------------------------------------- |
| id                      | BIGSERIAL    | 主键                                                |
| title                   | varchar(255) | 默认「新会话」，首问取前 30 字                      |
| repo_id                 | bigint null  | 外键 `repo.id`，`ON DELETE SET NULL`；空 = 所有仓库 |
| created_at / updated_at | timestamptz  | 默认 `now()`，更新时刷新                            |

索引：`idx_chat_session_updated(updated_at)`。

`chat_message`

| 字段                              | 类型        | 说明                                        |
| --------------------------------- | ----------- | ------------------------------------------- |
| id                                | BIGSERIAL   | 主键                                        |
| session_id                        | bigint      | 外键 `chat_session.id`，`ON DELETE CASCADE` |
| role                              | varchar(20) | `user` / `assistant`                        |
| content                           | text        | 消息正文                                    |
| citations                         | jsonb null  | 该轮来源快照（`SourceOut[]`）               |
| retrieval                         | jsonb null  | 检索调试 + `rewrite_mode` + `finish_reason` |
| prompt_tokens / completion_tokens | int null    | 用量（厂商优先，缺失按字符估算）            |
| created_at                        | timestamptz | 默认 `now()`                                |

索引：`idx_chat_message_session(session_id)`。

### 4.2 迁移

项目未使用 alembic，沿用 `scripts/migrate_ingest.py` 的做法（`create_all` + `checkfirst`）：

```powershell
cd c:\no_space_demand\Graduation-project-back
.venv\Scripts\python.exe scripts\migrate_chat.py
```

脚本只创建两张对话表，可重复执行，不会影响已有 `repo/file/code_chunk` 数据。

---

## 5. 配置与切换模型

配置全部走 `.env`（参考 [.env.example](file:///c:/no_space_demand/Graduation-project-back/.env.example)），密钥不写代码、不提交仓库。

| 配置                                         | 默认      | 说明                                              |
| -------------------------------------------- | --------- | ------------------------------------------------- |
| `LLM_BASE_URL`                               | 空        | OpenAI 兼容地址，如 `https://api.deepseek.com/v1` |
| `LLM_API_KEY`                                | 空        | 厂商密钥                                          |
| `LLM_MODEL`                                  | 空        | 模型名，如 `deepseek-chat`                        |
| `LLM_TIMEOUT_SECONDS`                        | 60        | 单次请求超时                                      |
| `LLM_MAX_TOKENS`                             | 1024      | 回答上限                                          |
| `LLM_TEMPERATURE`                            | 0.2       | 采样温度                                          |
| `RETRIEVAL_CANDIDATE_K`                      | 30        | 每路召回数                                        |
| `RETRIEVAL_TOP_K`                            | 6         | 最终来源数                                        |
| `RETRIEVAL_VECTOR_WEIGHT` / `KEYWORD_WEIGHT` | 0.7 / 0.3 | RRF 权重                                          |
| `RETRIEVAL_MIN_SCORE`                        | 0.30      | 相似度阈值                                        |
| `RETRIEVAL_MAX_SOURCE_CHARS`                 | 8000      | 单条来源字符上限                                  |
| `RETRIEVAL_MAX_TOTAL_CHARS`                  | 24000     | 全部来源字符预算                                  |
| `CHAT_REWRITE_MODE`                          | rule      | `none` / `rule` / `llm`                           |
| `CHAT_MAX_CONCURRENCY`                       | 2         | 同时进行的 LLM 流数                               |

切换厂商只需改三项（`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`）后重启后端：

- DeepSeek：`https://api.deepseek.com/v1` + `deepseek-chat`
- 阿里百炼：`https://dashscope.aliyuncs.com/compatible-mode/v1` + `qwen-plus`
- 智谱等其它兼容厂商同理。

三项缺任意一项，`/api/chat` 返回 503，`/api/chat/config` 中 `llm_configured=false`。

---

## 6. 检索与改写实现逻辑

### 6.1 检索（`retriever.py`）

1. **查询向量**：用进程内 bge-small-zh-v1.5 编码问题，查 Chroma `code_chunks`（cosine，`similarity = 1 - distance`），每路取 `candidate_k=30`，并按 `repo_ids` 过滤。
2. **标识符抽取**：正则提取驼峰/下划线标识符、点号调用链、引号内名称，去掉停用词，最多 8 个。
3. **关键词通道**：`code_chunks` 表 ILIKE 匹配——`name` 精确 3 分、`name` 子串 2 分、`signature` 1.5 分、`content` 1 分，同分按名称长度升序。
4. **RRF 融合**：`score = 0.7/(60+rank_vec) + 0.3/(60+rank_kw)`，按融合分排序。
5. **阈值过滤**：`similarity >= 0.30` 或 `keyword_score >= 2.0`，避免纯噪声入 prompt。
6. **父子去重**：同一文件若同时命中模块切片与其成员切片，保留成员；仍保留模块信息用于上溯。
7. **module_id**：成员切片沿父级上溯（最多 3 跳）找到所属模块，供前端跳转。
8. **超长处理**：内容 > 6000 字符时——类型切片（class/interface/enum）优先展开最多 3 个成员（头 20 行）；其余保留头 120 行 + 尾 40 行；硬上限 8000 字符，超出记 `truncated`。
9. **预算控制**：按分数顺序累加，超过 `24000` 字符的后续来源直接丢弃，`debug.filtered` 计数。
10. **展示分**：`0.7 * similarity + 0.3 * min(1, keyword_score/4)`，写入 `SourceOut.score`。

### 6.2 上下文（`context.py`）

- 系统提示要求：先给结论、引用具体行号并标注 `[n]`、依据不足要直说、中文回答。
- 每条来源格式：`[n] 仓库 | 文件路径 | 类型/名称 | 行号 | 分数` + 代码块。
- 历史取最近 6 条，每条截断 800 字符；来源按分数排序后拼接；最后追加当前问题。

### 6.3 改写（`rewrite.py`）

| 模式           | 行为                                                                                                             |
| -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `none`         | 原样检索。                                                                                                       |
| `rule`（默认） | 问题含「它/这个/上面/刚才」等指代词，或长度 ≤ 12 字时，把历史中的标识符补成 `（上下文标识符：X Y Z）` 后再检索。 |
| `llm`          | 用一次非流式调用把问题改写为自包含查询；调用失败或超时降级为 `rule_fallback`。                                   |

改写结果只用于**检索**；发给模型的问题仍是用户原文，避免答非所问。

---

## 7. 验证结果

### 7.1 真实调用（无 mock）

- `compileall` 与 `import app.main` 通过；`scripts/migrate_chat.py` 实际建表成功。
- `POST /api/retrieve` 走真实 embedding + Chroma + PostgreSQL：命中 `module_id=24118`，`score≈0.81`，预热后 `took_ms` 92–191ms。
- 会话 CRUD 真实读写 PostgreSQL；删除会话时消息级联删除。
- 仓库锁定真实生效：会话 `repo_id=3` 时请求携带 `repo_ids=[999]`，返回结果仍限定在仓库 3。
- `POST /api/query` 非流式、`POST /api/chat` 非流式均返回真实检索来源。

### 7.2 mock 链路验证（本机无 LLM 密钥）

为验证 SSE 与前端交互，在 `127.0.0.1:8099` 启动了一个 OpenAI 兼容的本地 mock 服务，后端通过环境变量指向它。**回答文案来自 mock，检索、上下文组装、持久化、SSE 转发均为真实实现。**

- `curl` 验证 SSE 事件顺序：`session` → `sources` → 多个 `delta` → `done`。
- 浏览器 E2E（前端临时副本运行在 3001，后端 8001）：
  1. 发送「JsonAdapter fromJson 怎么处理泛型」→ 出现流式回答「这是基于检索来源的模拟回答 [1]」，显示「来源 6 条」，顶部显示「所有仓库 · 已锁定」，左侧出现新会话。**PASS**
  2. 刷新页面 → 会话历史从 `GET /api/chat/sessions` + `GET /api/chat/sessions/{id}` 恢复；设置页「多轮改写」三选一且「规则」默认选中；锁定状态保持。**PASS**
- 测试会话已从数据库清理，临时前端服务与 mock 服务已停止。

### 7.3 前端类型检查

`npx tsc --noEmit` 退出码 0。

---

## 8. 运行方式

```powershell
# 1) 启动数据库（如未启动）
docker start code-rag-pg

# 2) 建表
cd c:\no_space_demand\Graduation-project-back
.venv\Scripts\python.exe scripts\migrate_chat.py

# 3) 配置 .env（至少 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）

# 4) 启动后端
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```powershell
# 前端
cd c:\no_space_demand\Graduation-project-front
npm run dev
```

前端默认请求 `http://localhost:8000`；如后端端口不同，设置 `NEXT_PUBLIC_API_BASE`。

---

## 9. 遗留问题

1. **未做重排**：按决策本阶段不引入 reranker；纯关键词高分但语义弱的片段在 RRF 下排名偏后。
2. **citations 存全文**：消息表按来源保存完整切片内容，长会话体积较大；后续可改为只存 `chunk_id + 行号` 快照。
3. **编辑 / 重新生成 / 继续生成**：界面组件保留，但未接服务端，当前只支持新建消息与停止生成。
4. **流式期间删除会话**：若用户在生成中删除当前会话，落库可能触发外键错误；前端目前未在删除前 abort，建议后续补上。
5. **prompt 预算**：只按来源字符数与历史条数控制，未做严格的模型 token 上限校验；超长仓库可考虑按模型上下文动态裁剪。
6. **旧本地数据**：persist v2 迁移会丢弃旧版 localStorage 中的 mock 会话，属预期行为。
7. **多仓库首轮**：新建会话时若传入多个仓库，仅首轮生效，会话仍记为「所有仓库」；这符合「锁定单仓库或全库」的语义，但与「多仓库固定集合」不同。
