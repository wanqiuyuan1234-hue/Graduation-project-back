# 对话（问答）链路设计：检索 → 上下文 → 生成 → 引用

> 文档定位：对话功能的现状对照与设计基线，供后续 P0/P1/P2 实施使用。
> 标注约定：**[实测]** 表示本次通过只读脚本或接口实际确认；**[推定]** 表示由代码与数据结构推断，未单独验证；**[参考]** 指容器镜像 `coderag-frontend-v1.tar` 中的行为。
> 本文只做设计与勘察，未修改任何业务代码、未启动常驻服务。

---

## 0. 结论摘要

1. **参考系统没有真实检索链路**。`coderag-frontend-v1.tar` 是一个只含前端产物的 Docker 镜像归档，服务端代码里没有 Chroma / embedding / 向量库调用；`/api/retrieve` 返回 `SOURCE_CHUNKS` 抽样并伪造分数，`/api/chat` 调外部 z-ai SDK 一次性生成，前端再做打字机效果。它只能作为**交互与数据形状的参考**，不能作为检索实现参考。
2. **当前系统的检索底座已经具备**：Chroma 集合 `code_chunks` 实测 5994 条、512 维、cosine；PG 中 266 文件 / 5994 切片（AST 4033 + fixed 1961）；`parent_chunk_id` 已建立父子链。缺的是检索接口、上下文组装、LLM 调用、引用映射。
3. **最大风险不是检索算法，而是切片长度**：实测 `chunk.content` 平均 981 字符、**最大 93,505 字符**（约 2000+ 行）。不处理截断/成员展开，一条结果就能吃光上下文预算。
4. **推荐基线**：纯向量召回 30 → 标识符关键词加权融合（RRF，不引入重排模型）→ 取 6 条 → 单条超长按成员展开/截头 → OpenAI 兼容 LLM + SSE 流式 → 引用 `[n]` 映射到 `chunk_id/module_id + 行号` → 复用现有 `openModule(id,{startLine,endLine})` 跳转。理由见第 5 节。
5. 需要用户拍板的 9 个决策点集中在第 5 节；未拍板前不建议动 `chat-page.tsx` 的生成逻辑。

---

## 1. 参考系统（coderag 镜像）勘察

### 1.1 包内容

| 项 | 实际情况 |
|---|---|
| 文件 | `Graduation-project-front/coderag-frontend-v1.tar`（71MB） |
| 格式 | `docker save` 的 OCI 镜像归档（`manifest.json` + 8 个 layer blobs），**不是源码包** **[实测]** |
| 内容 | Alpine + Node + Next.js 产物（`.next/server`、`.next/static`），无 `.java` 源码、无数据库迁移 |
| 检索实现 | **不存在**。对 `app/.next/server` 全量 grep `chroma|hnsw|embedding|bge|vectorStore` 无命中 **[实测]** |

### 1.2 参考系统的“假链路”

| 环节 | 参考实现 | 位置 |
|---|---|---|
| 检索 | `POST /api/retrieve` 从固定 `SOURCE_CHUNKS` 抽样，按关键词命中伪造成 `vectorScore`/`keywordScore`/`rerankScore`，固定 200–450ms 延迟 | `src/app/api/retrieve/route.ts` |
| 召回参数 | 默认 topK=6、`rerank=true`、`rerankModel=bge-reranker-v2-m3`、`vectorWeight=0.7`（调试面板可调） | `src/lib/rag/mock-data.ts`、`chat-page.tsx` |
| 生成 | `POST /api/chat` 用 z-ai SDK 一次性返回 `{content}`，最多带 10 条历史消息 | `src/app/api/chat/route.ts` |
| 流式 | **无**。前端用 `setInterval` 把整段回答逐字“打”出来 | `chat-page.tsx`（`streamReply`） |
| 引用 | `[n]` 来自 mock 的 `buildMockRetrievalContext`，点击后用 mock 的 `moduleId` 跳转 | `chat-page.tsx`、`mock-data.ts` |
| 数据形状 | `SourceChunk { id, moduleId, filePath, lines, content, score, rerankScore?, tokenCount }`；`ChatMessage { id, role, content, citations?, retrieval?, status, ... }` | `src/lib/rag/types.ts` |

**结论**：参考系统只固定了**前端期望的数据形状和交互**（来源面板、引用角标、跳转高亮、检索调试面板）。后端契约可以沿用这个形状，但检索与生成都要重新实现。

---

## 2. 当前系统真实状态

### 2.1 向量层（Chroma，实测）

| 项 | 值 | 依据 |
|---|---|---|
| 集合名 | `code_chunks` | `app/services/indexing/embedder.py` |
| 条数 | **5994** | 只读打开 PersistentClient 读取 **[实测]** |
| 维度 | **512** | 取一条 embedding 长度 **[实测]** |
| 距离度量 | **cosine**（`metadata={"hnsw:space":"cosine"}`） | 集合 metadata **[实测]** |
| 相似度换算 | `similarity = 1 - distance`（Chroma cosine distance 定义） | **[推定，需在实现时用一条已知查询核对]** |
| 元数据字段 | `chunk_id`、`repo_id`、`file_id`、`file_path`、`chunk_type`、`strategy`、`start_line`、`end_line`、`language` 全覆盖；`name`、`signature` 仅 AST 切片（4033 条）有 | 全量元数据统计 **[实测]** |
| embedding 模型 | `BAAI/bge-small-zh-v1.5`（512 维，中文语义为主） | `app/config.py` `embedding_model` |
| 查询侧编码 | 同一 `get_embedder()` 单例、`normalize_embeddings=True` | `embedder.py` |

**注意**：元数据里**没有 `module_id`、没有 `parent_chunk_id`**。引用要跳模块详情时，需要用 `chunk_id` 回 PG 推导（见 4.5）。

### 2.2 数据层（PostgreSQL，实测）

| 项 | 值 |
|---|---|
| `repo` | 3 条；仅 `id=3`（gson）`indexed_at` 非空，`id=5/6` 未完成入库 |
| `code_file` | 266 条（均属 repo 3） |
| `chunk` | 5994 条：`ast` 4033 + `fixed` 1961 |
| AST 切片构成 | 类型级 706（class 653 / interface 27 / enum 26）+ 方法/构造器 3327 |
| 长度 | 平均 981 字符、平均 28 行、**最大 93,505 字符** |
| `chunk.token_count` | **全部为 NULL**（5994/5994） |
| `chunk.content_hash` | 未回填（代码入库时未写入）**[推定]** |
| 表大小 | `chunk` 约 12MB |

关键字段（`app/models.py`）：

- `chunk`：`id`、`file_id`、`parent_chunk_id`、`chunk_type`、`strategy`、`name`、`signature`、`content`、`start_line`、`end_line`、`token_count`、`content_hash`
- `code_file`：`repo_id`、`path`、`language`、`size_bytes` 等（`chunk` 表**没有** `repo_id`，分仓统计需 join `code_file`）
- `repo`：`id`、`name`、`url`、`indexed_at`、`embedding_model`

### 2.3 已有接口（可直接复用的部分）

| 接口 | 复用点 |
|---|---|
| `GET /api/repos` | 取已入库仓库列表（`indexed_at` 非空）作为跨仓检索范围 |
| `GET /api/repos/{repo_id}/languages` | 前端筛选/展示，不参与检索 |
| `GET /api/repos/{repo_id}/files`、`/tree` | 引用跳转的文件定位 |
| `GET /api/modules`、`GET /api/modules/{id}` | **引用跳转目标**：模块详情已支持 `detailHighlight` 行高亮 |
| `GET /api/chunks/{chunk_id}` | 取切片全文、父切片、文件路径，做上下文组装与引用 |
| `GET /api/files/{file_id}/chunks` | 文件内切片列表（成员展开的备选数据源） |
| `POST /api/query` | **目前固定 501**（`app/routers/query.py`），是对话接口的占位 |

### 2.4 缺口清单

| 缺口 | 说明 |
|---|---|
| 检索接口 | 没有 `/api/retrieve`；`/api/query` 只抛 501 |
| LLM | 没有配置项、没有客户端、没有 prompt；venv 未安装 `openai`（可用 `httpx` 直接调 OpenAI 兼容协议） |
| 对话持久化 | 没有 `chat_session` / `chat_message` 表；前端会话只存在 localStorage（zustand persist） |
| 引用映射 | 向量元数据无 `module_id`，需用 `parent_chunk_id` 回查 |
| 前端真实调用 | `chat-page.tsx` 的生成/检索全走本地 mock；`fetchReply` 只兼容旧的 `{content}` 形状 |
| TS 错误 | `chat-page.tsx:579`：`streamTimerRef` 用 `ReturnType<typeof setInterval>`（解析成 NodeJS.Timeout），却赋值 `window.setInterval()` 的 `number` |
| 切片元数据 | `token_count` 全空 → 上下文预算只能按字符估算（建议 chars/3.2 保守换算） |

---

## 3. 目标链路设计

### 3.1 总览

```
用户提问
  │
  ├─(可选) 会话上下文：最近 N 轮 / 规则改写
  ▼
POST /api/chat  (SSE)
  ├─ 1. 查询编码：bge-small（进程内单例，512d）
  ├─ 2. 向量召回：Chroma code_chunks，n_results=30，where repo_id ∈ 范围
  ├─ 3. 关键词召回：查询中的标识符（类名/方法名）做精确/子串加权
  ├─ 4. 融合排序：RRF(向量 0.7 / 关键词 0.3)，去重、去父子重叠
  ├─ 5. 取 Top-6 → 回 PG 补全文件路径/父模块/token 估算
  ├─ 6. 超长切片处理：>6000 字符按成员展开或截头+尾
  ├─ 7. 组装 prompt（系统约束 + 来源块 + 历史 + 问题）
  ├─ 8. 调 LLM（OpenAI 兼容，stream=true）
  └─ 9. SSE 推送：sources → delta* → done
        │
前端渲染：answer 正文 + [n] 角标 + 来源面板
  └─ 点击 [n]/来源 → openModule(module_id, {startLine,endLine})
```

### 3.2 检索策略

**召回阶段**

| 通道 | 实现 | 作用 |
|---|---|---|
| 向量 | `collection.query(query_embeddings=[q], n_results=30, where={"repo_id":{"$in":[...]}})` | 语义匹配（“怎么解析泛型” → 相关实现） |
| 关键词 | 从问题中提取 `[A-Za-z_][A-Za-z0-9_]{2,}` 标识符；对 `chunk.name`/`signature` 精确命中加权，对 `content` 子串命中次之 | 代码问答里“类名/方法名”是最强信号，纯向量会漏 |

**融合阶段**

- RRF（Reciprocal Rank Fusion，k=60）：`score = Σ w_i / (k + rank_i)`，向量权重 0.7、关键词 0.3。
- 去重：
  - 同一 `chunk_id` 只保留一次；
  - **父子重叠**：若选中 `class` 切片，且其成员切片（`parent_chunk_id` 指向它）也命中，优先保留成员切片（更精确、更短）；类型切片仅在成员都没命中时保留。
- 不做 cross-encoder 重排（理由见 5.1）。

**结果数量**

- 候选 30（向量 30 + 关键词 30 融合后截断）。
- 最终 `top_k=6`（默认），允许请求覆盖 `1–10`。
- 相似度阈值：`similarity >= 0.30` 作为软门槛；关键词精确命中可绕过阈值。全部低于阈值时返回“未检索到相关代码”，不调用 LLM。

**延迟预估（2G2核，5994 条）**

- 查询编码：~20–60ms（模型已常驻）。
- Chroma HNSW 检索：~10–30ms。
- 关键词扫描 5994 条：~20–50ms（内存中字符串匹配，一次请求一次）。
- 合计检索 P95 目标 **< 300ms**（不含 LLM）。**[推定，需 P0 实测]**

### 3.3 切片过长怎么办

实测最大切片 93,505 字符，直接塞进 prompt 不可行。设计为三级策略：

1. **类型切片 > 6000 字符**：按 `parent_chunk_id` 取它的成员切片（方法/构造器），按“查询标识符命中 + 行号顺序”选最多 3 个；再加类声明头 20 行（保留类名/继承/字段）。
2. **成员/固定切片 > 6000 字符**：保留头 120 行 + 尾 40 行，中间插入 `// ... 省略 N 行 ...`。
3. **单条硬上限 8000 字符，全部来源合计 24,000 字符**（约 7–8k tokens，按 chars/3.2 估算）；超出按融合分数从低到高丢弃。

> P0 先用 PG 回查成员（一次 `IN` 查询），不必改 Chroma 元数据；P1 若发现回查频繁，再把 `parent_chunk_id`/`module_id` 补进向量元数据（需要重建或 `collection.update` 回填）。

### 3.4 上下文与 prompt

**预算**（总 ~8k tokens 以内，留足回答空间）：

| 部分 | 预算 |
|---|---|
| 系统约束 | ~400 tokens |
| 历史对话 | 最近 6 条消息，每条 ≤ 800 字符 |
| 代码来源 | 6 条 × ≤ 1500 tokens |
| 用户问题 | 原样 |

**来源块格式**（编号即引用号）：

```
[1] repo: gson | file: gson/src/main/java/com/google/gson/JsonAdapter.java
    | type: class JsonAdapter | lines: 45-120 | score: 0.82
```java
<content>
```
```

**系统 prompt 要点**（P0 固定文案，不做花活）：

- 只依据给定来源回答；不足时明确说“现有代码中没有找到”，不要编造。
- 涉及具体实现必须给行号，引用格式 `[n]`，n 必须在来源范围内。
- 先给结论，再给关键代码路径；不要整段复述源码。
- 中文回答，代码/标识符保持原文。

### 3.5 LLM 与流式

- 协议：**OpenAI 兼容** `POST {LLM_BASE_URL}/chat/completions`，`stream=true`。
- 客户端：直接用已安装的 `httpx.AsyncClient`，不新增 `openai` 依赖（减少体积与版本耦合）。**[实测]** venv 未安装 openai，httpx 已具备。
- 配置（新增到 `app/config.py`，全部走 `.env`）：
  ```
  LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  LLM_API_KEY=***
  LLM_MODEL=qwen-plus
  LLM_TIMEOUT_SECONDS=60
  LLM_MAX_TOKENS=1024
  LLM_TEMPERATURE=0.2
  ```
- 未配置 key 时：`/api/chat` 返回 503 + 明确文案，前端展示配置提示（不要静默回退 mock）。
- 并发：聊天是 I/O 等待，可用 `asyncio`；但同进程还有入库 worker（CPU/内存重），建议聊天并发上限 2（信号量），避免 2G 内存抖动。

### 3.6 引用与跳转

**引用结构**（`sources[]`，也是 SSE 第一个事件）：

```json
{
  "index": 1,
  "chunk_id": 12345,
  "module_id": 12000,
  "chunk_type": "method",
  "name": "fromJson",
  "repo_id": 3,
  "repo_name": "gson",
  "file_id": 88,
  "file_path": "gson/src/main/java/com/google/gson/JsonAdapter.java",
  "language": "java",
  "start_line": 45,
  "end_line": 78,
  "score": 0.82,
  "content": "  public <T> T fromJson(JsonElement json, Type typeOfT) { ... }"
}
```

`module_id` 推导规则（后端做，避免前端猜）：

1. `chunk_type ∈ {class, interface, enum}` → `module_id = chunk_id`；
2. 否则沿 `parent_chunk_id` 上溯，直到类型切片（方法→类通常 1 跳，枚举常量/嵌套最多 3 跳）；
3. 上溯不到（例如 `fixed` 切片）→ `module_id = null`。

前端跳转：

- 有 `module_id`：`openModule(module_id, { startLine, endLine })`，模块详情已有 `detailHighlight` 自动切到代码 Tab 并高亮（`module-detail.tsx:119-124`），**无需新页面**。
- 无 `module_id`：来源面板仅展示片段，不提供跳转（P2 可补“按文件路径搜索模块”）。

### 3.7 多仓库

- 请求可带 `repo_ids: number[]`；缺省为**全部已入库仓库**（`indexed_at` 非空）。
- Chroma where 用 `{"repo_id": {"$in": ids}}`；关键词通道在 PG 查询时同样按 `code_file.repo_id` 过滤。
- 前端当前会话有 `repoId` 字段（mock 阶段写死），建议：会话绑定仓库 → 请求带该 `repo_id`；“全部仓库”作为后续选项（需要 UI 增加入口，不在 P0）。

### 3.8 对话历史

两种存储方式见 5.3。无论选哪种，`/api/chat` 请求都带最近消息（role/content），后端不依赖服务端历史也能工作。

表结构（若选服务端持久化）：

```sql
chat_session(
  id bigserial pk, title text, repo_id int null references repo(id),
  created_at timestamptz, updated_at timestamptz
)
chat_message(
  id bigserial pk, session_id bigint references chat_session(id) on delete cascade,
  role text check (role in ('user','assistant','system')),
  content text, citations jsonb null, retrieval jsonb null,
  prompt_tokens int null, completion_tokens int null,
  created_at timestamptz
)
```

多轮改写：P2 再做（多一次 LLM 调用，1–2s）。P0/P1 直接把最近 6 条原文拼进 prompt。

---

## 4. 接口契约

### 4.1 `POST /api/retrieve`（检索调试，非流式）

请求：

```json
{
  "query": "Gson 怎么处理泛型反序列化",
  "repo_ids": [3],
  "top_k": 6,
  "candidate_k": 30,
  "vector_weight": 0.7,
  "keyword_weight": 0.3
}
```

响应：

```json
{
  "query": "...",
  "took_ms": 86,
  "strategy": "hybrid",
  "sources": [ { "index": 1, "chunk_id": 12345, "module_id": 12000, "...": "同 3.6" } ],
  "debug": {
    "vector_hits": 30,
    "keyword_hits": 12,
    "filtered": 4,
    "truncated": 1
  }
}
```

用途：前端“检索调试面板”复用；也便于回归测试（同一 query 的召回是否退化）。

### 4.2 `POST /api/chat`（SSE）

请求：

```json
{
  "message": "Gson 怎么处理泛型反序列化？",
  "repo_ids": [3],
  "session_id": null,
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "top_k": 6,
  "stream": true
}
```

响应 `Content-Type: text/event-stream`：

```
event: sources
data: {"sources":[...],"retrieval":{"took_ms":92,"strategy":"hybrid"}}

event: delta
data: {"text":"Gson 通过 "}

event: delta
data: {"text":"TypeToken 保留泛型信息 [1]"}

event: done
data: {"finish_reason":"stop","latency_ms":2380,"usage":{"prompt_tokens":5231,"completion_tokens":187}}
```

错误（已开始流式则用 `event: error`，未开始则 HTTP 4xx/5xx）：

```
event: error
data: {"code":"llm_unavailable","message":"LLM 未配置，请设置 LLM_API_KEY"}
```

`stream=false` 时返回与参考系统兼容的 JSON：`{ "content": "...", "sources": [...] }`，便于前端渐进切换。

### 4.3 与 `/api/query` 的关系

现有 `POST /api/query` 固定 501。建议：

- 保留路径，作为 `/api/chat` 的非流式别名（或直接删除，避免两套契约）。
- 推荐**保留 `query.py` 并实现为 `stream=false` 的薄封装**，文档与旧前端引用不用改。

---

## 5. 决策清单（需要你拍板）

> 每条给出：可选方案 → 代价 → 推荐。未确认前不写生成代码。

### 5.1 检索策略

| 方案 | 代价 | 效果 |
|---|---|---|
| A 纯向量 | 最低：无额外索引；内存仅模型常驻 | 语义问题好；类名/方法名精确问题会漏 |
| B 向量 + 标识符关键词（RRF 融合） | 低：每次请求扫 5994 条文本，~20–50ms；无新依赖 | 覆盖代码问答最常见两类问题 |
| C 混合 + cross-encoder 重排 | 高：`bge-reranker-base` 约 278M 参数，2G 内存下与 embedding 常驻叠加风险大；CPU 延迟 300ms–1s/6 条 | 排序略好，但性价比低 |

**推荐 B**。2G2核、单 worker 的约束下，重排模型的常驻内存和 CPU 抖动不划算；先用 RRF 把召回做稳，P2 再按真实 badcase 评估是否值得上重排。参考系统虽然默认 `rerank=true`，但那是 mock，没有真实成本参考价值。

### 5.2 多仓库范围

| 方案 | 代价 | 说明 |
|---|---|---|
| A 默认全库，可选 `repo_ids` | 低 | 一次 where 过滤，无需 UI |
| B 必须指定单仓库 | 低 | 现有会话有 `repoId`，但多仓对比场景做不了 |
| C 全局开关 + 仓库多选 UI | 中 | 需要改会话创建/头部 |

**推荐 A**：缺省全库、请求可传 `repo_ids`；P2 再做仓库多选 UI。

### 5.3 会话历史存储

| 方案 | 代价 | 说明 |
|---|---|---|
| A 无状态：前端传最近 N 条 | 最低，不改表 | 刷新后靠 localStorage；换设备丢历史；后端不存问答 |
| B 服务端两张表（session/message） | 中：迁移 + 3–4 个 CRUD 接口 | 可跨设备、可审计、可统计 token |
| C 单表 JSONB 存整段会话 | 低 | 查询/分页/统计不便 |

**推荐 P0 用 A（快速打通），P1 升级 B**。毕业设计场景建议最终有服务端历史，答辩时“会话可追溯”是加分项；但不要在第一版就做。

### 5.4 LLM 供应商与协议

| 方案 | 代价 | 说明 |
|---|---|---|
| OpenAI 兼容 + 国内厂商（DashScope/DeepSeek/智谱） | 低：httpx 直连，配置 3 个环境变量 | 国内网络可用，按量计费便宜 |
| 官方 OpenAI | 中：网络不可达风险 | 不推荐 |
| 本地 Ollama / llama.cpp | 高：2G2核跑不动可用质量模型 | 不推荐 |

**推荐**：OpenAI 兼容协议 + 阿里百炼 `qwen-plus`（或 DeepSeek `deepseek-chat`）。代码只依赖“兼容协议”，换厂商只改 env。

### 5.5 是否流式

| 方案 | 代价 | 说明 |
|---|---|---|
| SSE 流式 | 中：后端 async generator + 前端 ReadableStream 解析 | 首字延迟 1–2s，体验明显更好 |
| 非流式 | 低 | 6k prompt 下总耗时 3–8s，页面空转 |

**推荐 SSE**。前端已有打字机容器，改造为“收到 delta 直接追加”，比 mock 更简单。

### 5.6 引用粒度与跳转目标

| 方案 | 代价 | 说明 |
|---|---|---|
| A 引用到 chunk + module_id + 行号 | 低：后端上溯一次 | 精确到方法，模块详情可高亮 |
| B 只引用到模块 | 低 | 丢失方法级定位 |
| C 引用到文件 | 低 | 太粗，无法高亮 |

**推荐 A**。复用现有 `openModule` + `detailHighlight`，不新增页面。`fixed` 切片无模块时降级为“仅展示片段”。

### 5.7 超长切片处理

| 方案 | 代价 |
|---|---|
| A 成员展开（按 `parent_chunk_id` 取方法） | 低：一次 PG 查询；最贴合代码语义 |
| B 头尾截断 | 最低，但可能截掉关键实现 |
| C 按行窗口再切 | 中：需新切片策略，污染现有统计 |

**推荐 A 为主、B 兜底**，阈值 6000 字符。不要引入新的切片策略（会改变 5994 的统计口径）。

### 5.8 多轮改写

| 方案 | 代价 |
|---|---|
| A 不改写，直接拼最近 6 条 | 0 |
| B 规则改写（指代词 + 上一轮标识符） | 低 |
| C LLM 改写 | 每次 +1–2s、+1 次调用 |

**推荐 P0/P1 用 A，P2 视 badcase 决定 B/C**。

### 5.9 是否保留前端 mock 兜底

| 方案 | 代价 |
|---|---|
| A 保留 mock，仅在后端不可用时提示 | 低，但容易“看起来能用其实没接” |
| B 直接删除 mock，后端未配置就显式报错 | 需同步改 3 处 |

**推荐 B**（P1 完成时）。当前 `chat-page.tsx` 的 `generateReply`/`buildMockRetrievalContext` 会让联调误判，保留价值低；如果担心演示环境，用 `NEXT_PUBLIC_CHAT_MOCK=1` 显式开关。

---

## 6. 前端改造清单

| 文件 | 现状 | 改动 |
|---|---|---|
| `src/components/rag/chat/chat-page.tsx` | 全 mock：`generateReply`、`buildMockRetrievalContext`、本地打字机；`fetchReply` 只兼容 `{content}` | 改为调用 `/api/chat` SSE；`sources` 事件填充引用；`delta` 直接追加；删除 mock 生成（或加开关）；`handleReRetrieve` 调 `/api/retrieve` |
| 同上 `handleJumpToCode` | 用 mock `chunk.moduleId` 跳转 | 改用 `source.module_id`；`null` 时禁用跳转 |
| 同上 `streamTimerRef`（288 行）/ 579 行 | `ReturnType<typeof setInterval>` 在 Node 类型下是 `Timeout`，赋 `window.setInterval()` 的 `number` → TS2322 | 改为 `useRef<number | null>(null)`；`window.clearInterval` 配套 |
| `src/lib/api/hooks.ts` | 无聊天请求 | 新增 `useChatStream`（fetch + ReadableStream 解析 SSE） |
| `src/lib/api/client.ts` | 只有 `apiGet`/`apiPost` | 新增 `apiStream`（POST + `text/event-stream`） |
| `src/lib/rag/types.ts` | `SourceChunk` 已有 `moduleId` | 增补 `chunkId`、`fileId`、`repoName`、`language`；`ChatMessage.citations` 与后端 `sources` 对齐 |
| `src/app/api/chat/route.ts`、`src/app/api/retrieve/route.ts` | mock 实现 | P1 删除或改为透传后端（保留可作为 CORS 代理） |
| `src/lib/rag/mock-data.ts` | mock 切片/检索 | P2 清理（先保留调试数据） |

后端新增文件（规划，不在本轮）：

- `app/schemas/chat.py`：`RetrieveRequest/Response`、`ChatRequest`、`SourceOut`
- `app/services/rag/retriever.py`：编码 + 向量召回 + 关键词 + RRF + 去重
- `app/services/rag/context.py`：截断/成员展开 + prompt 组装
- `app/services/rag/llm.py`：OpenAI 兼容客户端 + SSE 生成器
- `app/routers/chat.py`：`/retrieve`、`/chat`
- `app/routers/query.py`：501 → 非流式薄封装

---

## 7. 分阶段实施建议

| 阶段 | 内容 | 产出 | 工作量（人日，粗估） |
|---|---|---|---|
| **P0** | `POST /api/retrieve`（纯向量 + 标识符加权）+ `POST /api/chat`（非流式）+ 引用映射 + 前端替换 mock 生成 | 能真实问答，来源/行号正确 | 3–5 |
| **P1** | SSE 流式 + 上下文预算/截断/成员展开 + 检索调试面板 + 会话绑定仓库 + 删除 mock | 体验接近成品，首字 1–2s | 4–6 |
| **P2** | 服务端会话表与历史列表 + 多轮 LLM 改写 + 多仓多选 UI + 重排评估 + `token_count` 回填 | 完整产品化 | 5–8 |

> 估算假设：1 人、熟悉现有代码；不含联调等待与模型/网络问题排查。P0 的验收标准是“10 个已知问题能命中正确模块并给出可跳转行号”。

---

## 8. 风险与验证

| 风险 | 影响 | 应对 |
|---|---|---|
| bge-small-zh 对英文标识符语义弱 | 类名/方法名查询召回差 | 关键词通道兜底；P2 评估换 `bge-m3`（1024 维，需重建 5994 条向量，内存+延迟上升） |
| 93k 字符超长切片 | prompt 爆炸/成本失控 | 3.3 的截断策略，硬上限 |
| Chroma 重建期间查询为空 | 对话短时无结果 | 入库完成前前端提示“索引更新中”；检索失败返回空 sources |
| 2G 内存 | LLM 流 + 入库并发时 OOM | 聊天并发信号量 2；不在本地跑 LLM |
| 国内网络访问 LLM | 超时 | 选国内厂商；`LLM_TIMEOUT_SECONDS=60`；失败快速返回 |
| 引用编号漂移 | `[3]` 超出 sources | 生成后校验编号范围，越界替换为 `[?]` 并在 `done` 事件标注 |
| 前端 Next dev 代理缓冲 SSE | 流“不流” | 前端直连后端 `NEXT_PUBLIC_API_BASE`（CORS 已允许） |

**验收方法**：

1. 接口层：`/api/retrieve` 对 10 个固定 query 记录 sources 的 `chunk_id`，回归对比。
2. 引用正确性：断言回答中的 `[n]` 全部落在 `sources` 范围内，且点击能打开对应模块并高亮行号。
3. 性能：检索 P95 < 300ms；首字 < 2s；6k prompt 总耗时 < 8s（国内厂商）。
4. 稳定性：连续 20 轮问答后后端 RSS 增量 < 100MB。

---

## 9. 附录

### 9.1 实测数据（只读探针）

| 项 | 值 | 方法 |
|---|---|---|
| Chroma 集合 | `code_chunks`，5994 条，512 维，cosine | `PersistentClient` 只读打开，`get(limit=…)` |
| 元数据覆盖 | 全字段覆盖；`name/signature` 仅 AST | 全量 `get(include=["metadatas"])` 统计 |
| PG | repo 3 条（1 条已入库），文件 266，切片 5994 | SQL `count(*)` |
| 切片构成 | AST 4033（类型 706：class 653 / interface 27 / enum 26；成员 3327）+ fixed 1961 | SQL group by |
| token_count | 0 条非空 | SQL |
| 长度 | avg 981 字符 / 28 行；max 93,505 字符 | SQL |
| 表大小 | `chunk` ≈ 12MB | `pg_total_relation_size` |
| 依赖 | 无 `openai`；有 `httpx`、`chromadb`、`sentence-transformers` | `pip list` |

> 探针脚本位于系统临时目录，未写入仓库；未启动常驻服务。只读打开 Chroma 时 SQLite 生成了一个 `chroma.sqlite3-journal` 临时文件，客户端关闭时被回收，属事务日志，不影响数据（集合计数一致）。

### 9.2 代码位置索引

| 主题 | 位置 |
|---|---|
| 切片与元数据 | `app/services/indexing/chunker.py`、`persist.py`、`embedder.py` |
| 入库流水线 | `app/services/ingest.py` |
| 模块查询/详情 | `app/routers/modules.py`、`app/services/module_view.py` |
| 仓库与任务 | `app/routers/repos.py`、`app/schemas/job.py` |
| 检索占位 | `app/routers/query.py`（501） |
| 前端对话 | `src/components/rag/chat/chat-page.tsx` |
| 前端 mock 接口 | `src/app/api/chat/route.ts`、`src/app/api/retrieve/route.ts` |
| 前端类型 | `src/lib/rag/types.ts` |
| 导航与高亮 | `src/lib/rag/store.ts`（`openModule`）、`src/components/rag/modules/module-detail.tsx`（`detailHighlight`） |

### 9.3 参考系统参数（仅作交互参考）

- `topK=6`、`rerank=true`、`rerankModel=bge-reranker-v2-m3`、`vectorWeight=0.7`
- 历史最多 10 条；无流式；引用来自 mock `buildMockRetrievalContext`
- 镜像内无向量库/embedding 调用（全量 grep 无命中）

### 9.4 待确认项（阻塞实施）

见第 5 节 D1–D9。最低需要先确认：**D1 检索策略、D4 LLM 供应商、D5 是否流式、D3 历史存储**；其余可按推荐默认推进。
