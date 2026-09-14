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
# 单个切片目标长度（字符数），建议 500-1000
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
# 相邻切片重叠长度（字符数），建议 100-200
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ---------------- 检索配置 ----------------
# 每次检索返回的最相关片段数量（3-5 为宜）
TOP_K = int(os.getenv("TOP_K", "5"))
# 相关性阈值：余弦相似度低于该值的片段视为不相关，直接丢弃
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))

# ---------------- Embedding 模型配置 ----------------
# 默认使用轻量级 all-MiniLM-L6-v2（首次运行自动下载，约 90MB）
# 中文文档较多时可改为 "BAAI/bge-small-zh-v1.5"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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
