"""
全局配置模块
从项目根目录的 .env 文件加载 DeepSeek API Key 及各类可调参数。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（backend/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载根目录下的 .env 环境变量文件
load_dotenv(BASE_DIR / ".env")

# ---------------- HuggingFace 模型下载镜像 ----------------
# 国内直连 huggingface.co 下载 Embedding 模型极慢或失败，
# 默认切换到 hf-mirror.com 镜像（可在 .env 中通过 HF_ENDPOINT 覆盖）。
# 注意：必须在 sentence_transformers / huggingface_hub 被导入之前设置。
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------------- DeepSeek 大模型配置 ----------------
# API Key 在 .env 文件中填写（参考 .env.example）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
# DeepSeek OpenAI 兼容接口地址
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# 使用的模型名称
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
# 生成温度：控制回答的随机性与发散程度。
# 0.1=极保守(幻觉低但归纳弱)；0.3=轻度放宽(允许合理归纳，幻觉可控)；
# 0.5=平衡；0.7=较强发散。项目默认 0.3，实际效果仍需通过证据评测确认。
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))


def is_api_key_configured() -> bool:
    """
    校验 DeepSeek API Key 是否已正确配置。
    排除空值与 .env.example 中的占位符文本，避免携带非法 Key 发起请求。

    :return: True 表示 Key 已配置为合法格式
    """
    if not DEEPSEEK_API_KEY:
        return False
    # 占位符（含中文提示）视为未配置
    if "你的" in DEEPSEEK_API_KEY or "API-Key" in DEEPSEEK_API_KEY:
        return False
    # 合法 Key 应为 sk- 开头的 ASCII 字符串
    return DEEPSEEK_API_KEY.startswith("sk-") and DEEPSEEK_API_KEY.isascii()

# ---------------- 文档切片配置 ----------------
# 单个切片目标长度（字符数）。
# bge-small-zh-v1.5 的 max_seq_length=512 token，中文约 1 字≈1.3 token，
# 切片控制在 500 字符以内可保证完整向量化、避免尾部截断导致召回丢失。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
# 相邻切片重叠长度（字符数），保证跨块边界信息不丢
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ---------------- 检索配置 ----------------
# 每次检索返回的最相关片段数量。
# 由 5 提至 8：扩大候选池，提高召回上限，配合 bge 中文模型更准的语义匹配。
TOP_K = int(os.getenv("TOP_K", "8"))
# 相关性阈值：余弦相似度低于该值的片段视为不相关，直接丢弃。
# bge-small-zh 的相似度分布整体低于 all-MiniLM-L6-v2，阈值由 0.25 降至 0.15，
# 避免正确片段被误过滤（降低误拒率、提升任务完成率）。
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.15"))

# ---------------- Rerank 重排配置（两阶段检索第二阶段）----------------
# 粗排候选池大小：先按向量余弦召回 20 个候选，保证召回上限；
# 再交给 Cross-Encoder 逐对精排，取 TOP_K 个喂给大模型。
RETRIEVE_CANDIDATE_K = int(os.getenv("RETRIEVE_CANDIDATE_K", "20"))
# 是否启用 Cross-Encoder 重排；加载失败时自动降级为粗排原顺序，不影响问答
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
# 重排模型：默认为 HF 模型 ID（bge-reranker-base，中文优化的 Cross-Encoder）。
# 更强可选 BAAI/bge-reranker-large / BAAI/bge-reranker-v2-m3（均更慢、更吃内存）。
# 若配置值对应的项目内本地目录存在（如 models/bge-reranker-base，
# 可在网络差时手动放置模型文件），则优先使用本地路径，完全跳过下载。
_rerank_model_raw = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
_local_rerank_dir = BASE_DIR / _rerank_model_raw
if _local_rerank_dir.exists():
    RERANK_MODEL = str(_local_rerank_dir)
else:
    RERANK_MODEL = _rerank_model_raw
# 重排批量大小：候选(问题,片段)对的推理 batch，CPU 建议 8
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
# 重排分数下限。bge-reranker 输出为未归一化 logit（可正可负，不局限 0-1），
# 默认 -100 即不过滤；如发现重排后仍混入弱相关片段，可尝试设为 0
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "-100"))

# ---------------- 会话记忆配置（多轮对话）----------------
# 保留的历史轮数（1 轮 = 用户问 + 助手答）。
# 超出后只携带最近 N 轮参与问题改写与生成，控制 token 消耗与噪声。
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "3"))

# ---------------- Embedding 模型配置 ----------------
# 使用中文 Embedding 模型 bge-small-zh-v1.5（约 100MB）。
# 此前 all-MiniLM-L6-v2 为英文模型，对中文语义匹配弱，导致召回率仅 51.85%；
# 切换中文模型从根因修复召回问题。首次运行经 hf-mirror.com 自动下载。
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ---------------- OCR 降级配置（扫描件 PDF）----------------
# 当 pdfplumber 提取不到文本层（疑似扫描件 PDF）时，降级用 Tesseract OCR 识别图片文本。
# 前置依赖：
#   1) 系统安装 Tesseract OCR 引擎（Windows 安装包后在 .env 或 PATH 中暴露可执行文件）
#   2) 通过 pip 安装 pytesseract（已在 requirements.txt 中声明）
# 未安装时自动跳过 OCR，不影响纯文本 PDF 的正常解析。
# 是否启用 OCR 降级，默认开启；未安装 Tesseract 时自动跳过
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
# OCR 识别语言包：chi_sim+eng 覆盖中文简体与英文；仅英文场景可设为 eng
OCR_LANG = os.getenv("OCR_LANG", "chi_sim+eng")
# 转图分辨率（DPI）：越高越清晰但越慢，300 适合常规扫描件
OCR_DPI = int(os.getenv("OCR_DPI", "300"))

# ---------------- 存储路径配置 ----------------
# 向量库持久化目录
DATA_DIR = BASE_DIR / "data"
# 上传原始文档存放目录
UPLOAD_DIR = BASE_DIR / "uploads"
# 向量库 JSON 文件路径
STORE_PATH = DATA_DIR / "vector_store.json"
# 前端静态页面目录
FRONTEND_DIR = BASE_DIR / "frontend"

# 允许上传的文档扩展名
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
