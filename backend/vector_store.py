"""
轻量向量库模块
基于 JSON 文件持久化 + NumPy 余弦相似度检索，无需外部数据库依赖。
适合简历项目级别的中小规模知识库（万级片段内性能足够）。
"""
import json
import threading
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from config import STORE_PATH


@dataclass
class DocumentChunk:
    """向量库中的一条文本片段记录"""
    chunk_id: str          # 片段唯一 ID
    doc_id: str            # 所属文档 ID
    doc_name: str          # 所属文档文件名
    content: str           # 片段原文内容
    chunk_index: int       # 片段在原文中的序号
    vector: list[float]    # 片段的 Embedding 向量


@dataclass
class DocumentMeta:
    """已入库文档的元信息"""
    doc_id: str            # 文档唯一 ID
    doc_name: str          # 文件名
    chunk_count: int       # 切片数量
    char_count: int        # 原文字符数
    upload_time: str       # 上传时间（ISO 格式）


class VectorStore:
    """
    轻量向量库：内存保存向量矩阵，JSON 文件持久化。
    检索使用归一化向量的点积（等价于余弦相似度）。
    """

    def __init__(self, store_path: Path = STORE_PATH):
        """
        初始化向量库并从磁盘加载已有数据。

        :param store_path: 持久化 JSON 文件路径
        """
        self.store_path = store_path
        self.documents: dict[str, DocumentMeta] = {}   # doc_id -> 文档元信息
        self.chunks: list[DocumentChunk] = []          # 全部文本片段
        self._matrix: np.ndarray | None = None         # 片段向量矩阵（N x D）
        self._lock = threading.Lock()                  # 并发读写保护锁
        self._load()

    # ---------------- 持久化 ----------------

    def _load(self) -> None:
        """从 JSON 文件加载向量库数据到内存"""
        if not self.store_path.exists():
            return
        with open(self.store_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.documents = {
            d["doc_id"]: DocumentMeta(**d) for d in raw.get("documents", [])
        }
        self.chunks = [DocumentChunk(**c) for c in raw.get("chunks", [])]
        self._rebuild_matrix()

    def _save(self) -> None:
        """将当前内存数据序列化写入 JSON 文件"""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [asdict(d) for d in self.documents.values()],
            "chunks": [asdict(c) for c in self.chunks],
        }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _rebuild_matrix(self) -> None:
        """根据片段列表重建向量矩阵，供检索时批量计算相似度"""
        if self.chunks:
            self._matrix = np.array([c.vector for c in self.chunks], dtype=np.float32)
        else:
            self._matrix = None

    # ---------------- 写入 ----------------

    def add_document(self, meta: DocumentMeta, chunks: list[DocumentChunk]) -> None:
        """
        入库一篇文档的全部切片。

        :param meta: 文档元信息
        :param chunks: 该文档的切片列表（含向量）
        """
        with self._lock:
            self.documents[meta.doc_id] = meta
            self.chunks.extend(chunks)
            self._rebuild_matrix()
            self._save()

    def remove_document(self, doc_id: str) -> int:
        """
        删除指定文档及其全部切片。

        :param doc_id: 文档 ID
        :return: 被删除的切片数量
        """
        with self._lock:
            removed = [c for c in self.chunks if c.doc_id == doc_id]
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
            self.documents.pop(doc_id, None)
            self._rebuild_matrix()
            self._save()
            return len(removed)

    # ---------------- 检索 ----------------

    def search(self, query_vector: list[float], top_k: int,
               threshold: float = 0.0) -> list[dict]:
        """
        余弦相似度检索最相关的文本片段。

        :param query_vector: 归一化后的查询向量
        :param top_k: 返回片段数量上限
        :param threshold: 相似度下限，低于该值的片段被过滤
        :return: [{"chunk": DocumentChunk, "score": float}, ...] 按分数降序
        """
        if self._matrix is None or len(self.chunks) == 0:
            return []

        # 向量已归一化，点积即余弦相似度
        query = np.asarray(query_vector, dtype=np.float32)
        scores = self._matrix @ query

        # 取分数最高的 top_k 个下标
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                continue
            results.append({"chunk": self.chunks[idx], "score": score})
        return results

    # ---------------- 统计 ----------------

    def stats(self) -> dict:
        """
        返回知识库统计信息。

        :return: 文档数、片段总数、文档元信息列表
        """
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "documents": [asdict(d) for d in self.documents.values()],
        }


def new_id() -> str:
    """生成短唯一 ID（用于文档与切片标识）"""
    return uuid.uuid4().hex[:12]


# 全局向量库单例
vector_store = VectorStore()
