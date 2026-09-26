"""
Rerank 重排模块（两阶段检索的第二阶段）
使用 sentence-transformers 的 CrossEncoder 对"问题-片段"对逐对精排。

与 Embedding 双塔检索的区别：
- Embedding 粗排：问题与片段各自独立编码后算余弦，速度快但精度有限；
- Cross-Encoder 精排：问题与片段拼接后共同过 Transformer，能捕捉深层交互，
  精度显著更高，但计算更贵，因此只对粗排召回的 Top-N（默认 20）候选使用。

首次运行自动下载模型（走 config 中配置的 hf-mirror.com 镜像）。
模型加载或推理失败时由调用方（agent）降级为粗排原顺序。
"""
import threading

from sentence_transformers import CrossEncoder

from config import RERANK_BATCH_SIZE, RERANK_MODEL

# 全局单例重排模型，避免重复加载占用内存
_reranker: CrossEncoder | None = None
# 模型初始化锁：防止预加载线程与请求线程并发时重复加载
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoder:
    """
    懒加载并返回 Cross-Encoder 重排模型单例（线程安全）。
    首次调用时若本地无缓存会自动下载（走 HF_ENDPOINT 镜像），
    下载进度打印在后端控制台。

    :return: 已加载的 CrossEncoder 模型
    """
    global _reranker
    # 双重检查锁：已加载时直接返回，避免每次调用都竞争锁
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                print(f"[Rerank] 正在加载重排模型 {RERANK_MODEL}（首次运行需联网下载，请耐心等待）...")
                _reranker = CrossEncoder(RERANK_MODEL)
                print("[Rerank] 重排模型加载完成")
    return _reranker


def rerank(query: str, contents: list[str]) -> list[dict]:
    """
    对粗排候选片段进行 Cross-Encoder 精排。

    :param query: 用户问题（多轮场景下为改写后的独立问题）
    :param contents: 粗排候选片段原文列表（按粗排分数降序）
    :return: [{"index": 原候选下标, "score": 重排logit}, ...]
             覆盖全部候选并按重排分数降序；截取 Top-N 与阈值过滤由调用方完成
    """
    if not contents:
        return []

    model = get_reranker()
    # CrossEncoder 输入为 (问题, 候选) 文本对列表
    pairs = [[query, content] for content in contents]
    scores = model.predict(pairs, batch_size=RERANK_BATCH_SIZE)

    # 按重排分数降序排序，保留全部候选（调试视图需展示每条重排分）
    ordered_indices = sorted(
        range(len(contents)), key=lambda i: float(scores[i]), reverse=True
    )

    return [
        {"index": idx, "score": round(float(scores[idx]), 4)}
        for idx in ordered_indices
    ]
