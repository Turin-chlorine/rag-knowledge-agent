"""
Agent 核心模块
实现 "提问 -> (多轮改写) -> 粗排检索 -> Cross-Encoder 重排 -> 基于结果生成回答" 的完整流程。
回答严格限制在检索到的原文范围内，知识库无相关内容时直接拒答；
支持以 session_id 关联的多轮对话记忆。
"""
from datetime import datetime

import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    TOP_K,
    RETRIEVE_CANDIDATE_K,
    RELEVANCE_THRESHOLD,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_SCORE_THRESHOLD,
    is_api_key_configured,
)
from embedding import embed_query
from memory import session_manager
from reranker import rerank
from vector_store import vector_store

# 检索无结果时的固定回复文案
NO_INFO_REPLY = "文档内无相关资料"

# 系统提示词：约束模型以检索内容为事实依据作答，并支持结合历史理解追问。
SYSTEM_PROMPT = """你是一个知识库问答助手。请基于下方【参考资料】回答用户问题，遵守：
1. 回答应以参考资料为主要依据，可对资料进行归纳、总结和同义改写，使回答更完整易读。
2. 可基于参考资料进行合理推断和补充，以提供更完整的回答。
3. 引用具体内容时使用 [n] 标注来源编号（n 为资料序号）。
4. 若参考资料中确实没有与问题相关的信息，回复"文档内无相关资料"。
5. 回答简洁准确，可使用 Markdown 格式组织内容。
6. 可结合对话历史理解最新问题中的指代与省略。"""

# 问题改写提示词：多轮场景下把追问补全为可独立检索的问题
REWRITE_SYSTEM_PROMPT = """你是一个问题改写助手。请根据对话历史，将用户最新提问改写为脱离上下文也能独立理解的完整问题（补全代词、省略的主语与对象）。
要求：
1. 只输出改写后的问题本身，不要解释、不要加引号、不要回答该问题；
2. 若最新问题本身语义已完整，则原样输出。"""


def chat_completion(messages: list[dict], temperature: float) -> str:
    """
    通用 DeepSeek Chat 接口调用（OpenAI 兼容格式）。

    :param messages: 完整消息列表 [{"role", "content"}, ...]
    :param temperature: 本次调用的生成温度
    :return: 模型生成的文本
    :raises RuntimeError: API Key 未配置或请求失败时抛出
    """
    # 校验 Key 是否已正确配置（排除空值与占位符），避免携带非法 Key 发起请求
    if not is_api_key_configured():
        raise RuntimeError("未配置有效的 DEEPSEEK_API_KEY，请在项目根目录 .env 文件中填写真实 Key（sk- 开头）")

    response = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek API 调用失败({response.status_code}): {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


def rewrite_question(question: str, history: list[dict]) -> str:
    """
    结合对话历史，将含指代/省略的追问改写为可独立理解与检索的问题。
    改写失败（接口异常等）时降级使用原问题，保证主流程不中断。

    :param question: 用户本轮原始提问
    :param history: 该会话最近若干轮的历史消息
    :return: 改写后的独立问题；失败时返回原问题
    """
    messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    try:
        # 改写任务用 0 温度，保证确定性、不发散
        rewritten = chat_completion(messages, temperature=0.0)
        return rewritten if rewritten else question
    except Exception as exc:
        print(f"[Agent] 问题改写失败，降级使用原问题检索: {exc}")
        return question


def retrieve_documents(question: str) -> tuple[list[dict], dict]:
    """
    Agent 内置 Skill：两阶段文档检索工具。
    第一阶段（粗排）：问题向量化，余弦检索 RETRIEVE_CANDIDATE_K 个候选并按阈值过滤；
    第二阶段（精排）：Cross-Encoder 对全部候选重打分，按重排分取 TOP_K。

    :param question: 用于检索的问题（多轮场景下为改写后的问题）
    :return: (最终入选片段列表, 检索调试信息)
             片段含 retrieval_score（粗排余弦）与 rerank_score（精排logit，无重排时为 None）
    """
    # 空调试骨架（知识库为空时直接返回）
    debug = {
        "retrieval_query": question,
        "candidate_k": RETRIEVE_CANDIDATE_K,
        "coarse_count": 0,
        "rerank_enabled": RERANK_ENABLED,
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL if RERANK_ENABLED else None,
        "final_count": 0,
        "candidates": [],
    }

    # 知识库为空时无需向量化，直接返回空结果
    if vector_store.stats()["chunk_count"] == 0:
        return [], debug

    # ---------- 第一阶段：Embedding 粗排 ----------
    query_vector = embed_query(question)
    hits = vector_store.search(
        query_vector, top_k=RETRIEVE_CANDIDATE_K, threshold=RELEVANCE_THRESHOLD
    )
    # 粗排候选（保持粗排顺序），chunk_id 用于后续映射与去重
    coarse = [
        {
            "chunk_id": hit["chunk"].chunk_id,
            "content": hit["chunk"].content,
            "doc_name": hit["chunk"].doc_name,
            "chunk_index": hit["chunk"].chunk_index,
            "retrieval_score": round(hit["score"], 4),
            "rerank_score": None,
        }
        for hit in hits
    ]
    debug["coarse_count"] = len(coarse)

    # ---------- 第二阶段：Cross-Encoder 精排（失败自动降级） ----------
    # 粗排下标 -> 重排分数；final 顺序按重排分
    rerank_score_by_index: dict[int, float] = {}
    rerank_applied = False
    if RERANK_ENABLED and coarse:
        try:
            ranked = rerank(question, [item["content"] for item in coarse])
            rerank_applied = True
            for hit in ranked:
                rerank_score_by_index[hit["index"]] = hit["score"]
        except Exception as exc:
            # 模型加载/推理失败：打印原因并降级为粗排原顺序，问答不受影响
            print(f"[Agent] Rerank 重排失败，降级使用粗排顺序: {exc}")

    debug["rerank_applied"] = rerank_applied

    # 确定最终入选片段：重排成功按重排分排序，否则按粗排顺序；取 TOP_K 并过阈值
    if rerank_applied:
        ordered_indices = sorted(
            rerank_score_by_index, key=lambda i: rerank_score_by_index[i], reverse=True
        )
    else:
        ordered_indices = list(range(len(coarse)))

    final_items: list[dict] = []
    # 粗排下标 -> 最终名次（用于调试视图展示名次变化）
    final_rank_by_index: dict[int, int] = {}
    for coarse_index in ordered_indices:
        rerank_score = rerank_score_by_index.get(coarse_index)
        # 重排阈值过滤（仅重排生效时）
        if rerank_applied and rerank_score < RERANK_SCORE_THRESHOLD:
            continue
        item = dict(coarse[coarse_index])
        item["rerank_score"] = rerank_score
        final_rank_by_index[coarse_index] = len(final_items) + 1
        final_items.append(item)
        if len(final_items) >= TOP_K:
            break

    # ---------- 组装调试信息：全部粗排候选 + 双分数 + 入选/名次 ----------
    candidates_debug = []
    for coarse_index, item in enumerate(coarse):
        candidates_debug.append({
            "coarse_rank": coarse_index + 1,
            "final_rank": final_rank_by_index.get(coarse_index),
            "selected": coarse_index in final_rank_by_index,
            "chunk_id": item["chunk_id"],
            "doc_name": item["doc_name"],
            "chunk_index": item["chunk_index"],
            "content": item["content"],
            "retrieval_score": item["retrieval_score"],
            "rerank_score": rerank_score_by_index.get(coarse_index) if rerank_applied else None,
        })
    debug["candidates"] = candidates_debug
    debug["final_count"] = len(final_items)

    return final_items, debug


def build_context(retrieved: list[dict]) -> str:
    """
    将检索到的片段拼装为带编号的参考资料文本，供提示词使用。
    编号与返回给前端的 sources 序号一致。

    :param retrieved: retrieve_documents 返回的最终片段列表
    :return: 格式化的参考资料文本
    """
    parts = []
    for i, item in enumerate(retrieved, start=1):
        parts.append(f"[{i}] 来源: {item['doc_name']}（第 {item['chunk_index'] + 1} 段）\n{item['content']}")
    return "\n\n".join(parts)


def call_deepseek(question: str, context: str, history: list[dict]) -> str:
    """
    携带多轮历史调用 DeepSeek 生成回答。

    :param question: 用户本轮原始提问
    :param context: 拼装好的参考资料文本
    :param history: 该会话最近若干轮历史消息（可为空列表）
    :return: 模型生成的回答文本
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 历史放在 system 之后、当前问题之前，模型据此理解追问中的指代
    messages.extend(history)
    user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    messages.append({"role": "user", "content": user_prompt})
    # 生成温度由 DEEPSEEK_TEMPERATURE 配置控制
    return chat_completion(messages, temperature=DEEPSEEK_TEMPERATURE)


def agent_answer(question: str, session_id: str | None = None) -> dict:
    """
    Agent 主流程：接收用户提问，完成"历史改写 → 两阶段检索 → 生成回答"全链路。

    流程：
    1. 确定会话（未传 session_id 则新建）并读取最近历史；
    2. 有历史时先用大模型把追问改写为独立问题；
    3. 粗排 + 重排获取相关片段；
    4. 无任何相关片段时直接返回固定拒答文案（不调用生成大模型，杜绝幻觉）；
    5. 否则携带历史与片段上下文生成带引用标注的回答；
    6. 将本轮问答写入会话记忆。

    :param question: 用户本轮提问
    :param session_id: 会话 ID，为空时自动新建
    :return: {"answer", "sources", "retrieval_count", "session_id", "debug"}
    """
    # 第一步：确定会话并读取历史
    if session_id is None:
        session_id = session_manager.create_session()
    history = session_manager.get_history(session_id)

    # 第二步：多轮改写（首轮无历史，直接用原问题）
    retrieval_query = rewrite_question(question, history) if history else question

    # 第三步：两阶段检索
    retrieved, debug = retrieve_documents(retrieval_query)
    # 补充改写信息，供前端调试视图展示
    debug["original_question"] = question
    debug["rewritten"] = retrieval_query != question

    # 第四步：无相关内容直接拒答（不调用生成模型）
    if not retrieved:
        answer = NO_INFO_REPLY
        sources = []
    else:
        # 第五步：组装上下文并携带历史生成回答
        context = build_context(retrieved)
        answer = call_deepseek(question, context, history)
        # 来源列表：双分数 + 原文，供前端引用面板与调试视图使用
        sources = [
            {
                "index": i + 1,
                "doc_name": item["doc_name"],
                "chunk_index": item["chunk_index"],
                "retrieval_score": item["retrieval_score"],
                "rerank_score": item["rerank_score"],
                "content": item["content"],
            }
            for i, item in enumerate(retrieved)
        ]

    # 第六步：本轮问答写入会话记忆（拒答也记录，保证后续追问可被正确改写）
    session_manager.append_message(session_id, "user", question)
    session_manager.append_message(session_id, "assistant", answer)

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_count": len(retrieved),
        "session_id": session_id,
        "debug": debug,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
