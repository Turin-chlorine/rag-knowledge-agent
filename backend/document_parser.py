"""
文档解析模块
负责将上传的 PDF / TXT / Markdown 文件解析为纯文本。
"""
from pathlib import Path

import pdfplumber


def parse_document(file_path: Path) -> str:
    """
    根据文件扩展名选择对应的解析器，返回纯文本内容。

    :param file_path: 已保存到本地的文件路径
    :return: 解析出的纯文本
    :raises ValueError: 不支持的文件类型
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(file_path)
    if suffix in {".txt", ".md", ".markdown"}:
        return _parse_text_like(file_path)

    raise ValueError(f"不支持的文件类型: {suffix}")


def _parse_pdf(file_path: Path) -> str:
    """
    使用 pdfplumber 逐页提取 PDF 文本。

    :param file_path: PDF 文件路径
    :return: 全部页面的拼接文本
    """
    pages_text: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
    return "\n".join(pages_text)


def _parse_text_like(file_path: Path) -> str:
    """
    读取 TXT / Markdown 等纯文本文件，自动尝试多种编码。

    :param file_path: 文本文件路径
    :return: 文件文本内容
    """
    for encoding in ("utf-8", "gbk", "utf-16"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # 兜底：忽略无法解码的字节
    return file_path.read_text(encoding="utf-8", errors="ignore")
