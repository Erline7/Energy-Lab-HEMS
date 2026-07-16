# ⚡ Energy Lab — AI 驱动家庭能源管理系统

智能家庭能源管理系统 (HEMS)，融合实时能源监控、多智能体 AI 助手、以及基于深度学习的负荷预测与电器负荷分解功能，提供统一平台。

---

## ✨ 功能特点

| 模块 | 说明 |
|------|------|
| **📊 实时能源大屏** | 通过 ECharts 可视化展示全屋用电、各电器分解功率（NILM）、24 小时负荷预测、水/气用量以及异常告警 |
| **🤖 多智能体 AI 管家（“阿福”）** | 基于 LangGraph 的对话式 AI 助手，内置 6 个专业智能体：<br/>• **数据分析师** — 历史与实时能源数据分析<br/>• **家庭规划师** — 负荷预测、账单预估、削峰填谷建议<br/>• **设备专家** — 实时 NILM 电器诊断<br/>• **知识库专员** — 基于 RAG 的说明书问答（PDF OCR + 翻译 + 向量检索）<br/>• **常规闲聊** — 中英双语日常对话<br/>• **汇总合成** — 将各专家报告整合为格式化回复 |
| **🧠 深度学习模型** | CNN+LSTM 用于负荷预测，TCN-BiGRU 用于 NILM 负荷分解 |
| **📚 RAG 知识库** | 接入电器 PDF 说明书，OCR 识别、自动翻译，并索引至 ChromaDB 进行语义检索 |
| **⏳ 时间沙盒** | 任意选择日期查看历史数据或触发日前 AI 预测 |
| **📥 数据导入管道** | 支持 CSV 数据集上传（电/水/气），智能模糊匹配列名 |
| **🌐 双语界面** | 动态切换中英文 |

---

## 🧱 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python, FastAPI, SQLAlchemy, SQLite |
| 前端 | Vue 3, TypeScript, Vite, Pinia, ECharts |
| 机器学习/深度学习 | PyTorch, scikit-learn, XGBoost |
| LLM/智能体 | LangChain, LangGraph, MCP (模型上下文协议) |
| RAG | ChromaDB, sentence-transformers, BGE 重排器, BM25 |
| PDF/OCR | PyMuPDF, OpenCV, magic-pdf, RapidOCR |
| LLM 提供商 | 阿里 DashScope (Qwen), DeepSeek, Kimi |

---

## 📂 项目结构

```
agent/
├── Backend/
│   ├── app.py                  # FastAPI 服务 & LangGraph 多智能体编排
│   ├── mcp_server.py           # MCP 工具服务器（15 个能源/AI 工具）
│   ├── setup_database.py       # SQLAlchemy ORM 模型 & 数据库表结构
│   ├── download_models.py      # 下载本地机器学习模型（Qwen via ModelScope）
│   ├── requirements.txt        # Python 依赖清单
│   ├── config/prompts.yaml     # 各智能体角色系统提示词
│   ├── Forecaster/             # 负荷预测模块（CNN+LSTM）
│   ├── NILM/                   # 非侵入式负荷监测（TCN-BiGRU）
│   ├── RAG/                    # PDF 说明书 → ChromaDB 向量库
│   ├── datasets/               # 原始电/水/气 CSV 数据集
│   ├── checkpoints/            # 训练好的模型权重（.pth, .pkl）
│   └── local_models/           # 本地嵌入/重排模型（BGE）
├── frontend/
│   ├── src/
│   │   ├── views/              # 大屏 & 聊天页面
│   │   ├── components/charts/  # ECharts 可视化组件
│   │   ├── store/              # Pinia 全局状态
│   │   └── router/             # Vue Router 配置
│   ├── package.json
│   └── vite.config.ts
└── .gitignore
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- 至少一个 LLM 提供商的 API 密钥（DashScope / DeepSeek）

### 1️⃣ 克隆仓库

```bash
git clone <repo-url>
cd agent
```

### 2️⃣ 后端配置

```bash
cd Backend

# 安装 Python 依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API 密钥：
#   DASHSCOPE_API_KEY=你的DashScope密钥
#   DEEPSEEK_API_KEY=你的DeepSeek密钥

# 下载本地机器学习模型（嵌入/重排模型）
python download_models.py

# 初始化数据库
python setup_database.py

# 启动后端服务
uvicorn app:app --host 0.0.0.0 --port 5000
```

### 3️⃣ 前端配置

```bash
cd frontend

# 安装 Node.js 依赖
npm install

# 配置 API 端点（可选，默认 http://127.0.0.1:5000）
echo "VITE_API_URL=http://127.0.0.1:5000" > .env

# 启动开发服务器
npm run dev
```

打开浏览器访问 `http://localhost:5173` 即可使用应用。

---

## ⚙️ 配置说明

### 环境变量（`Backend/.env`）

| 变量 | 说明 | 是否必须 |
|------|------|----------|
| `DASHSCOPE_API_KEY` | 阿里 DashScope API 密钥（Qwen 模型） | 至少一个* |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 至少一个* |

\* 至少需要配置一个 LLM 提供商的密钥。

### 智能体提示词（`Backend/config/prompts.yaml`）

所有 7 个智能体角色（主管、数据分析师、家庭规划师、设备专家、知识库专员、常规闲聊、汇总合成）的系统提示词均以 YAML 格式定义。你可根据需要修改这些提示词来调整智能体的行为与个性。

---

## 📡 API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/chat` | 向 AI 助手发送消息 |
| GET | `/api/dashboard` | 获取指定日期的仪表板数据 |
| POST | `/api/predict` | 触发负荷预测 |
| POST | `/api/upload` | 上传 PDF 说明书 |
| POST | `/api/upload_dataset` | 上传 CSV 能源数据集 |
| GET | `/api/progress` | SSE 流式上传进度 |
| GET | `/api/latest_date` | 获取最新可用数据日期 |

---

## 🧠 系统架构

```
用户查询
    │
    ▼
┌─────────────┐
│  主管智能体  │ ← 意图识别与路由
└──────┬──────┘
       │
       ├──► 数据分析师   ── 能源数据分析
       ├──► 家庭规划师   ── 预测与优化
       ├──► 设备专家     ── NILM 电器诊断
       ├──► 知识库专员   ── RAG 说明书问答
       └──► 常规闲聊     ── 日常对话
              │
              ▼
       ┌──────────────┐
       │ 汇总合成智能体 │ ← 合并专家报告 → 最终回复
       └──────────────┘
```

所有专业智能体通过 **MCP（模型上下文协议）** 服务器访问领域工具，该服务器提供 15 个集成工具，涵盖能源分析、预测、NILM 推理和知识检索。

---

## 📄 许可证

本项目仅供教学与研究用途。