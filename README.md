# 🏋️ 健身和体态矫正训练规划智能体

基于多 Agent 架构的智能体系统，为用户提供体态评估、健身咨询、个性化训练方案生成、方案调整和专业验证等一站式服务。

## ✨ 核心功能

### 智能体协作
- **体态问诊**：多轮对话式问诊，假设-验证-排除，像医生一样分析体态问题
- **健身咨询**：收集用户目标、基础、条件，制定科学的健身计划
- **导诊调度**：自动识别用户意图，调度对应专家，支持体态/健身混合意图
- **方案生成**：融合体态矫正和健身目标的个性化训练方案 + 饮食建议
- **双验证机制**：医生视角（安全）+ 教练视角（效果）双重把关，自动辩论直至共识

### RAG 全栈优化
- **混合检索**：向量检索（FAISS）+ BM25 关键词检索，RRF 融合
- **重排序**：bge-reranker-v2-m3 交叉编码器，提升 Top-K 准确率
- **查询改写**：专业术语扩展 + 指代消解，适配多轮对话场景
- **元数据过滤**：按领域（体态/健身）+ 文档类型精准过滤
- **完整评估体系**：检索层（P@4/R@4/F1/MRR）+ 生成层（Faithfulness/Answer Relevancy）

### 工程化能力
- **MCP 工具集成**：自定义 MCP Server，支持知识库检索、BMI 计算、卡路里计算
- **对话历史管理**：滑动窗口 + 摘要混合策略，控制 token 消耗
- **JSON 鲁棒性**：四层防护（基础清洗 + 智能修复 + 降级兜底）
- **流式输出**：SSE 流式响应，节点级进度展示
- **Docker 容器化**：一键部署，端口映射 + 数据卷挂载
- **规范日志系统**：logging 模块，替代 print，生产级可观测

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| Agent 框架 | LangGraph、LangChain |
| 大模型 | 阿里云百炼（qwen3.7-max） |
| 向量数据库 | FAISS + BM25 |
| 嵌入模型 | qwen3.7-text-embedding |
| 重排序模型 | BAAI/bge-reranker-v2-m3 |
| MCP 协议 | MCP (Model Context Protocol) |
| 后端框架 | FastAPI + Uvicorn |
| 数据库 | SQLite + SQLAlchemy |
| 容器化 | Docker + Docker Compose |
| 流式输出 | SSE（Server-Sent Events） |
| 前端 | 原生 HTML + CSS + JavaScript |
| 开发语言 | Python 3.10 |

## 🏗 架构设计

### 整体架构：导诊 + 双专家模式

采用类似医院"导诊护士 + 专科医生"的协作模式：

```
用户输入
    ↓
导诊 Agent（协调者，每轮分析意图，决定让哪个专家说话）
    ↓
    ┌─────────┴─────────┐
    ↓                   ↓
体态专家            健身专家
（ReAct 范式）      （ReAct 范式）
（自主判断是否检索） （自主判断是否检索）
    ↓                   ↓
    └─────────┬─────────┘
              ↓
        融合专家（深度合并双专家回复）
              ↓
          方案生成 Agent
  （训练计划+饮食计划，融合体态矫正和健身目标）
              ↓
          双验证 + 辩论机制
   （医生 + 教练交叉验证，自动辩论直至共识）
               ↓
          输出给用户
```

### RAG 检索流程

```
用户查询
    ↓
查询改写（术语扩展 + 指代消解）
    ↓
混合检索（向量召回 + BM25 召回）
    ↓
RRF 融合排序
    ↓
重排序（bge-reranker-v2-m3）
    ↓
Top-K 结果注入 Prompt
```

## 📁 项目结构

```
posture-fitness-agent/
├── src/
│   ├── state.py                  # 状态定义（AgentState、UserProfile 等）
│   ├── agents/                   # 所有 Agent
│   │   ├── triage_agent.py       # 导诊 Agent（意图识别+调度）
│   │   ├── posture_agent.py     # 体态专家（康复科医生）
│   │   ├── fitness_agent.py     # 健身专家（健身教练）
│   │   ├── merge_agent.py        # 融合专家（双专家回复合并）
│   │   ├── plan_generator.py     # 方案生成 Agent
│   │   └── validators.py         # 双验证机制 + 辩论
│   ├── graph/
│   │   └── main_graph.py         # LangGraph 主图
│   ├── rag/
│   │   ├── knowledge_base.py     # RAG 知识库（FAISS + BM25 + 混合检索）
│   │   ├── reranker.py           # 重排序模块（bge-reranker-v2-m3）
│   │   ├── query_rewriter.py     # 查询改写 + 指代消解
│   │   └── get_relevant_knowledge.py  # 检索入口
│   ├── tools/
│   │   ├── mcp_server.py         # MCP 服务端
│   │   ├── mcp_client.py         # MCP 客户端
│   │   └── expert_tools.py       # 本地工具
│   ├── database/
│   │   ├── connection.py         # 数据库连接管理
│   │   ├── models.py             # 数据模型
│   │   └── crud.py               # 增删改查操作
│   └── utils/
│       ├── llm.py                # LLM 封装 + JSON 四层防护
│       ├── logger.py             # 统一日志系统
│       ├── chat_history_utils.py  # 对话历史管理（滑动窗口+摘要）
│       └── config.py             # 统一配置管理
├── api/
│   └── main.py                   # FastAPI 接口（REST + SSE 流式）
├── web/
│   ├── index.html                # 前端页面结构
│   ├── style.css                 # 前端样式
│   └── app.js                    # 前端逻辑
├── data/
│   ├── knowledge/                # 知识库文档
│   │   ├── posture/              # 体态专题文档
│   │   └── fitness/              # 健身动作文档
│   ├── faiss_index/              # 向量库文件（自动生成）
│   └── evaluation/               # 评估测试集
├── scripts/
│   ├── evaluate_retrieval.py     # 检索层评估
│   ├── evaluate_generation.py    # 生成层评估
│   └── test_e2e.py               # 端到端测试
├── Dockerfile                    # Docker 镜像构建
├── docker-compose.yml            # Docker Compose 编排
├── pyproject.toml                # 项目配置
├── requirements.txt              # Python 依赖
└── README.md                     # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd posture-fitness-agent

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

在项目根目录创建 `.env` 文件：

```env
# 大模型 API 配置
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
```

### 4. 构建知识库

```bash
python -m src.rag.knowledge_base
```

### 5. 启动服务

**方式一：直接运行**
```bash
python -m api.main
```

**方式二：Docker 部署**
```bash
docker compose up --build -d
```

服务启动后：
- API 文档：http://localhost:8000/docs
- 前端页面：http://localhost:8000/

## 📡 API 文档

### 基础接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 + 前端页面 |
| POST | `/chat` | 发送消息，返回完整回复 |
| POST | `/chat/stream` | 流式聊天（SSE 格式） |
| GET | `/sessions` | 获取所有会话列表 |
| DELETE | `/sessions/{session_id}` | 删除会话 |

## 📊 评估结果

### 检索层评估（K=4）

| 指标 | 数值 | 说明 |
|------|------|------|
| Precision@4 | 0.94 | 前4条中相关结果占比 |
| Recall@4 | 0.94 | 应命中文档被覆盖的比例 |
| Hit Rate | 1.0 | 至少命中1条的查询占比 |
| MRR | 1.0 | 第一个相关结果排名倒数平均 |

### 生成层评估（LLM-as-Judge）

| 指标 | 数值 | 说明 |
|------|------|------|
| Faithfulness | 0.92 | 回答忠实度，有无幻觉 |
| Answer Relevancy | 0.98 | 回答是否切题 |
| Context Precision | 0.88 | 上下文利用率 |

## 🔑 核心技术亮点

### 1. LangGraph 多 Agent 编排

- 使用 `StateGraph` 构建有状态的工作流
- 条件边（`conditional_edges`）实现动态路由
- 并行专家节点（both 意图时并行执行）
- 统一状态管理，TypedDict 类型定义清晰

### 2. RAG 全栈优化

- **混合检索**：向量检索 + BM25 关键词检索，RRF 融合排序
- **重排序**：bge-reranker-v2-m3 交叉编码器，精准排序
- **查询改写**：专业术语扩展 + 指代消解，适配多轮对话
- **元数据过滤**：按领域和文档类型精准过滤
- **完整评估体系**：检索层 + 生成层，量化优化效果

### 3. MCP 工具集成

- 自定义 MCP Server，通过 stdio 协议连接
- 支持工具自动注册，统一管理
- 并行安全：加锁 + 双重检查，防止并行节点冲突

### 4. 对话历史管理

- 滑动窗口 + 摘要混合策略
- 保留最近 6 轮原文，更早对话压缩成摘要
- 增量摘要更新，控制 token 消耗

### 5. JSON 鲁棒性（四层防护）

- 第一层：基础清洗（正则提取 + 去 markdown 标记）
- 第二层：智能修复（调用 LLM 修复格式错误）
- 第三层：降级兜底（纯文本当 response）

### 6. Docker 容器化

- Dockerfile 多阶段构建
- docker-compose 一键部署
- 数据卷挂载，知识库持久化
- 端口映射，外部访问

## 📝 开发踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| MCP 连接报 UnicodeDecodeError | Windows stdout 默认 GBK | mcp_server.py 开头强制 UTF-8 |
| MCP 并行连接冲突 | 两个 agent 同时调用 load_mcp_tools | 加 asyncio.Lock + 双重检查 |
| JSON 解析失败 | LLM 输出不规范 | 四层防护：清洗 + 智能修复 + 降级 |
| 对话轮次计数不对 | 多节点重复更新 + 无 reducer | 只在 triage 更新，其他节点不碰 |
| BM25 召回干扰 | 关键词检索引入无关结果 | RRF 融合 + rerank 重排序 |
| 对话历史太长 token 超限 | 全部历史都传给 LLM | 滑动窗口 + 摘要混合策略 |
| 工作目录不对找不到 src | 删除 sys.path.append 后相对路径失效 | 用 `python -m` 方式运行 |

## 📄 License

MIT License

---

*项目开发周期：约 4 周*
*技术栈：LangGraph + RAG + MCP + FastAPI + Docker*
