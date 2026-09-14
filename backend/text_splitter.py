"""
文本切片模块
将解析后的纯文本按语义边界切分为带重叠的片段，供向量化与检索使用。
"""
import re


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    将长文本切分为若干片段。

    策略：优先按段落/句子等自然边界切分，保证片段长度接近 chunk_size，
    相邻片段之间保留 chunk_overlap 个字符的重叠，避免语义被截断。

    :param text: 待切分的完整文本
    :param chunk_size: 单个片段目标长度（字符数）
    :param chunk_overlap: 相邻片段重叠长度（字符数）
    :return: 文本片段列表
    """
    # 清洗空白字符，去掉多余空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    # 按段落、句号、问号等自然边界拆成小单元，避免切在词中间
    units = re.split(r"(?<=[。！？.!?\n])", text)
    units = [u for u in units if u.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for unit in units:
        # 单个单元超长时强制按固定长度硬切
        if len(unit) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for i in range(0, len(unit), chunk_size - chunk_overlap):
                chunks.append(unit[i:i + chunk_size].strip())
            continue

        # 当前片段还能容纳该单元则追加
        if len(current_chunk) + len(unit) <= chunk_size:
            current_chunk += unit
        else:
            # 当前片段已满，保存并开始新片段
            chunks.append(current_chunk.strip())
            # 用上一片段末尾的重叠文本作为新片段开头，保持上下文连贯
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap_text + unit

    # 保存最后一个片段
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # 过滤过短的碎片片段（少于 20 字符通常无信息量）
    return [c for c in chunks if len(c) >= 20]
