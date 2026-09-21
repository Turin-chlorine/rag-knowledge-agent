# RAG 知识库 AI Agent

一个基于 **DeepSeek 大模型 + 本地轻量向量库** 的网页版 RAG（检索增强生成）知识库问答系统。
上传文档后自动切片、向量化入库；提问时 Agent 先调用内置检索 Skill 获取相关文档片段，再交由 DeepSeek 严格基于原文生成带引用来源的回答。

## ✨ 功能特性

- 📄 **多格式文档支持**：PDF / TXT / Markdown 上传，自动解析入库；扫描件 PDF 可选启用 OCR 降级识别
- 🔪 **智能切片**：按段落/句子自然边界切分，切片长度 500 字符、重叠 100 字符（可配置）
- 🧮 **本地向量化**：`BAAI/bge-small-zh-v1.5` 中文 Embedding 模型，无需外部向量数据库（JSON 持久化 + NumPy 余弦检索）
- 🤖 **Agent 工作流**：提问 → 调用文档检索 Skill → 取 Top-K 相关片段 → DeepSeek 生成回答
- 🛡️ **防幻觉机制**：回答仅基于检索到的原文并标注 `[n]` 引用来源；知识库无相关信息时直接回复「文档内无相关资料」，不经过大模型，从源头杜绝编造
- 🎨 **简洁 Web UI**：左侧知识库管理（上传/删除/统计），右侧对话问答，引用来源可展开查看

## 📁 项目结构

```
rag-knowledge-agent/
├── backend/
│   ├── main.py             # FastAPI 主应用（接口 + 静态页面托管）
│   ├── agent.py            # Agent 核心：检索 Skill + DeepSeek 调用 + 防幻觉逻辑
│   ├── config.py           # 全局配置（读取 .env）
│   ├── document_parser.py  # PDF/TXT/Markdown 解析（含扫描件 OCR 降级）
│   ├── text_splitter.py    # 文本切片
│   ├── embedding.py        # 向量化（sentence-transformers）
│   └── vector_store.py     # 轻量向量库（JSON 持久化 + 余弦检索）
├── frontend/
│   ├── index.html          # 页面结构
│   ├── style.css           # 样式
│   └── app.js              # 交互逻辑
├── test_docs/              # 评测用测试文档（Dify / Coze / 幻觉 三篇）
├── data/                   # 向量库持久化数据（运行后自动生成）
├── uploads/                # 上传的原始文档（运行后自动生成）
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板（API Key 填这里）
└── README.md
```

## 🚀 本地启动步骤

### 方式一：一键启动（推荐）

只需安装 Python 3.10+，然后双击运行（或在终端执行）：

- **Windows**：双击 `start.bat`
- **macOS / Linux**：`bash start.sh`

脚本会自动创建虚拟环境、安装依赖、启动服务。首次运行会自动生成 `.env`，填入你的 DeepSeek API Key 后再次运行即可。

### 方式二：手动启动

#### 1. 环境准备

- Python 3.10 及以上
- pip（建议使用虚拟环境）

```bash
# 进入项目目录
cd rag-knowledge-agent

# （推荐）创建并激活虚拟环境
python -m venv venv
# Windows PowerShell:
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次启动时会自动下载 Embedding 模型 `BAAI/bge-small-zh-v1.5`（约 100MB），请保持网络畅通。项目已默认配置国内镜像 `hf-mirror.com`，国内网络可正常下载。

### 3. 配置 DeepSeek API Key ⭐

复制环境变量模板并填入你的 API Key：

```bash
# Windows PowerShell / CMD
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

打开项目根目录的 **`.env`** 文件，将 `DEEPSEEK_API_KEY` 替换为你的真实 Key：

```env
DEEPSEEK_API_KEY=sk-你的真实Key
```

> API Key 在 [DeepSeek 开放平台](https://platform.deepseek.com/api_keys) 创建。
> 其他参数（切片长度、检索数量、相关性阈值、生成温度等）也可在 `.env` 中调整，详见 `.env.example` 注释。

### 4. 启动服务

```bash
uvicorn backend.main:app --reload
```

看到 `Uvicorn running on http://127.0.0.1:8000` 后，浏览器打开：

```
http://127.0.0.1:8000
```

### 5. 使用流程

1. 左侧点击上传区域（或拖拽）上传 PDF / TXT / Markdown 文档，等待解析与向量化完成
2. 在右侧输入框提问，Enter 发送
3. Agent 自动检索相关片段并生成回答，点击「📎 引用来源」可查看原文出处与相似度
4. 知识库中没有相关信息时，会直接回复「文档内无相关资料」

## 🔌 API 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/upload` | 上传文档（multipart/form-data，字段名 `file`） |
| GET | `/api/documents` | 获取知识库统计与文档列表 |
| DELETE | `/api/documents/{doc_id}` | 删除文档及其全部切片 |
| POST | `/api/chat` | 知识库问答，请求体 `{"question": "你的问题"}` |

启动后也可访问自动生成的交互式接口文档：`http://127.0.0.1:8000/docs`

## ⚙️ 核心实现说明

- **切片策略**：优先按 `。！？.!?\n` 等自然边界切分，超长单元硬切；相邻切片保留 100 字符重叠保证上下文连贯。切片长度 500 字符匹配 bge-small-zh 的 512 token 窗口，避免尾部截断导致召回丢失
- **检索方式**：向量归一化后用点积计算余弦相似度，取 Top-K（默认 8）并过滤低于 `RELEVANCE_THRESHOLD`（默认 0.15）的片段
- **Embedding 模型**：默认 `BAAI/bge-small-zh-v1.5` 中文模型，对中文语义匹配效果优于英文模型；如需切回轻量英文模型，可在 `.env` 中将 `EMBEDDING_MODEL` 改为 `sentence-transformers/all-MiniLM-L6-v2`（注意：切换模型需删除 `data/vector_store.json` 重建向量库，两种模型向量维度不同）
- **防幻觉**：检索结果为空时直接返回固定拒答文案，不调用大模型；系统提示词强约束模型只依据参考资料作答
- **生成温度**：默认 0.3，允许模型对参考资料进行轻度归纳与同义改写，在任务完成率与幻觉率之间取平衡

## 🔍 OCR 扫描件支持（可选）

默认情况下，PDF 解析依赖文本层（`pdfplumber` 提取），对扫描件 PDF（每页是图片、无文本层）无法提取内容。项目内置了轻量级 OCR 降级：当某页文本层为空时，自动调用 Tesseract OCR 识别图片文本。

**这是完全可选的功能，不安装也不影响纯文本 PDF / TXT / Markdown 的正常使用。**

如需启用扫描件 OCR：

1. 下载并安装 Tesseract OCR 引擎（Windows 推荐 UB-Mannheim 版本）：
   👉 https://github.com/UB-Mannheim/tesseract/wiki
2. 安装时勾选 **Chinese (Simplified)** 语言包（识别中文所需）
3. 将安装目录（默认 `C:\Program Files\Tesseract-OCR`）加入系统 `PATH`
4. 重启终端，运行 `tesseract --version` 验证安装成功
5. 重启后端服务，下次上传扫描件 PDF 即自动 OCR

`pytesseract`（Tesseract 的 Python 封装）已在 `requirements.txt` 中声明，`pip install` 时会自动安装。未检测到 Tesseract 引擎时，系统会自动跳过 OCR 降级并打印提示，不影响其他功能。

OCR 相关参数（语言、分辨率、开关）可在 `.env` 中调整，详见 `.env.example` 的 OCR 配置段。

## ❓ 常见问题

- **上传后一直卡在「正在解析并向量化」**：首次运行需联网下载 Embedding 模型（约 100MB）。项目已默认切换到国内镜像 `hf-mirror.com`，并在服务启动时后台预加载模型。下载期间前端会禁用上传并提示「模型加载中」，进度可见于后端控制台；下载完成后自动恢复。后续启动走本地缓存，无需再下载。
- **首次启动很慢**：正在下载 Embedding 模型，属正常现象，后续启动会走缓存。可通过 `GET /api/health` 查看 `model_ready` 是否就绪。
- **回答提示未配置 API Key**：确认 `.env` 文件在项目根目录且 `DEEPSEEK_API_KEY` 已填写**真实** Key（`sk-` 开头的纯英文数字，勿保留示例中的中文占位符），修改后需重启服务。
- **检索不到相关内容**：可适当调低 `.env` 中的 `RELEVANCE_THRESHOLD`（默认 0.15，可降到 0.10）。
- **扫描件 PDF 提取不到文字**：默认 PDF 解析依赖文本层，扫描件需安装 Tesseract OCR 引擎以启用 OCR 降级，详见上方「OCR 扫描件支持」。
- **修改代码后不生效**：确认已重启 `uvicorn` 服务，并强制刷新浏览器页面（Ctrl+F5）以加载最新前端。

## 📊 评测数据

项目使用 `test_docs/` 下的三篇文档（Dify 工作流、Coze 知识库、LLM 幻觉）构建了 40 道测试题（单跳事实 / 多文档对比 / 库内无答案 / 越界提问），并对召回率、检索精确率、幻觉率、任务完成率进行了完整评测。

评测明细与指标汇总见飞书多维表格：**[RAG知识库Agent评测集（40题）](https://hcna5lsumjxf.feishu.cn/base/WptQb7Qb4a8wP7scKrlcrqkznBf)**

> 表格包含每题的 Agent 最终回答、检索到的原文出处（文档 + 片段全文 + 相似度）、原文真值及四项判定标签，可直接用于复盘检索质量与防幻觉效果。
