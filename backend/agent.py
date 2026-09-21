"""
Agent 核心模块
实现 "提问 -> 调用检索 Skill -> 基于检索结果生成回答" 的完整流程。
回答严格限制在检索到的原文范围内，知识库无相关内容时直接拒答。
"""
from datetime import datetime

import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    TOP_K,
    RELEVANCE_THRESHOLD,
    is_api_key_configured,
)
from embedding import embed_query
from vector_store import vector_store

# 检索无结果时的固定回复文案
NO_INFO_REPLY = "文档内无相关资料"

# 系统提示词：约束模型以检索内容为主要依据作答。
# 相比此前的"完全禁止推测"，现允许归纳总结与合理推断以提升任务完成率，
# 但仍禁止编造资料中不存在的事实，并将幻觉率控制在低水平。
SYSTEM_PROMPT = """你是一个知识库问答助手。请基于下方【参考资料】回答用户问题，遵守：
1. 回答应以参考资料为主要依据，可对资料进行归纳、总结和同义改写，使回答更完整易读。
2. 可基于参考资料进行合理推断，但不得编造资料中明确不存在的事实。
3. 引用具体内容时使用 [n] 标注来源编号（n 为资料序号）。
4. 若参考资料中确实没有与问题相关的信息，回复"文档内无相关资料"。
5. 回答简洁准确，可使用 Markdown 格式组织内容。"""


def retrieve_documents(question: str) -> list[dict]:
    """
    Agent 内置 Skill：文档检索工具。
    将问题向量化后在向量库中检索最相关的文档片段。

    :param question: 用户问题
    :return: 相关片段列表 [{"content", "doc_name", "chunk_index", "score"}]
    """
    # 知识库为空时直接返回空结果
    if vector_store.stats()["chunk_count"] == 0:
        return []

    # 问题向量化并检索，同时按相似度阈值过滤不相关片段
    query_vector = embed_query(question)
    hits = vector_store.search(
        query_vector, top_k=TOP_K, threshold=RELEVANCE_THRESHOLD
    )

    return [
        {
            "content": hit["chunk"].content,
            "doc_name": hit["chunk"].doc_name,
            "chunk_index": hit["chunk"].chunk_index,
            "score": round(hit["score"], 4),
        }
        for hit in hits
    ]


def build_context(retrieved: list[dict]) -> str:
    """
    将检索到的片段拼装为带编号的参考资料文本，供提示词使用。

    :param retrieved: retrieve_documents 的返回结果
    :return: 格式化的参考资料文本
    """
    parts = []
    for i, item in enumerate(retrieved, start=1):
        parts.append(f"[{i}] 来源: {item['doc_name']}（第 {item['chunk_index'] + 1} 段）\n{item['content']}")
    return "\n\n".join(parts)


def call_deepseek(question: str, context: str) -> str:
    """
    调用 DeepSeek Chat 接口（OpenAI 兼容格式）生成回答。

    :param question: 用户问题
    :param context: 拼装好的参考资料文本
    :return: 模型生成的回答文本
    :raises RuntimeError: API Key 未配置或请求失败时抛出
    """
    # 校验 Key 是否已正确配置（排除空值与 .env.example 占位符），
    # 避免携带非法 Key 发起请求导致编码错误
    if not is_api_key_configured():
        raise RuntimeError("未配置有效的 DEEPSEEK_API_KEY，请在项目根目录 .env 文件中填写真实 Key（sk- 开头）")

    user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"

    response = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": DEEPSEEK_TEMPERATURE,  # 由 config 控制，0.3 允许轻度归纳
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek API 调用失败({response.status_code}): {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


def agent_answer(question: str) -> dict:
    """
    Agent 主流程：接收用户提问，先调用检索 Skill，再交给大模型生成回答。

    流程：
    1. 调用文档检索工具获取相关片段；
    2. 若无任何相关片段，直接返回固定拒答文案（不调用大模型，杜绝幻觉）；
    3. 否则将片段作为上下文交给 DeepSeek 生成带引用标注的回答。

    :param question: 用户问题
    :return: {"answer", "sources", "retrieval_count"}
    """
    # 第一步：调用检索 Skill
    retrieved = retrieve_documents(question)

    # 第二步：无相关内容直接拒答，不经过大模型
    if not retrieved:
        return {
            "answer": NO_INFO_REPLY,
            "sources": [],
            "retrieval_count": 0,
        }

    # 第三步：组装上下文并调用 DeepSeek 生成回答
    context = build_context(retrieved)
    answer = call_deepseek(question, context)

    # 来源列表（供前端展示引用出处）
    sources = [
        {
            "index": i + 1,
            "doc_name": item["doc_name"],
            "chunk_index": item["chunk_index"],
            "score": item["score"],
            "content": item["content"],
        }
        for i, item in enumerate(retrieved)
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_count": len(retrieved),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
