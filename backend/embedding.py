"""
向量化模块
使用 sentence-transformers 将文本片段转换为稠密向量。
默认模型 all-MiniLM-L6-v2 轻量高效，首次运行自动下载。
"""
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

# 全局单例模型，避免重复加载占用内存
_model: SentenceTransformer | None = None
# 模型初始化锁：防止预加载线程与请求线程并发时重复加载模型
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """
    懒加载并返回 Embedding 模型单例（线程安全）。
    首次调用时若本地无缓存会自动下载模型（走 HF_ENDPOINT 镜像），
    下载进度打印在后端控制台。

    :return: 已加载的 SentenceTransformer 模型
    """
    global _model
    # 双重检查锁：已加载时直接返回，避免每次调用都竞争锁
    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"[Embedding] 正在加载模型 {EMBEDDING_MODEL}（首次运行需联网下载，请耐心等待）...")
                _model = SentenceTransformer(EMBEDDING_MODEL)
                print("[Embedding] 模型加载完成")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批量将文本编码为归一化向量（归一化后点积即余弦相似度）。

    :param texts: 待编码文本列表
    :return: 向量列表，每个向量为 float 列表
    """
    if not texts:
        return []
    model = get_model()
    # normalize_embeddings=True 使向量模长为 1，便于用点积算余弦相似度
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.astype(np.float32).tolist()


def embed_query(query: str) -> list[float]:
    """
    将单条用户提问编码为归一化向量。

    :param query: 用户问题文本
    :return: 归一化查询向量
    """
    return embed_texts([query])[0]
