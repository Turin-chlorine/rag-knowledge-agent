"""
文档解析模块
负责将上传的 PDF / TXT / Markdown 文件解析为纯文本。
PDF 优先提取文本层；对文本层为空的扫描件页面降级使用 Tesseract OCR 识别。
"""
from pathlib import Path

import pdfplumber

from config import OCR_DPI, OCR_ENABLED, OCR_LANG


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
    解析 PDF 文件，逐页提取文本。

    优先使用 pdfplumber 提取 PDF 文本层；当某页文本层为空（疑似扫描件）时，
    降级调用 Tesseract OCR 识别该页图片文本，实现轻量级扫描件 OCR 降级。
    未安装 Tesseract 时自动跳过 OCR，仅返回已提取到的文本层内容。

    :param file_path: PDF 文件路径
    :return: 全部页面的拼接文本
    """
    pages_text: list[str] = []
    # 延迟探测 Tesseract 可用性：未安装 pytesseract 或引擎本体时，ocr_available=False
    ocr_available = _check_tesseract_available() if OCR_ENABLED else False

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # 1) 优先提取文本层（数字化 PDF 通常直接命中）
            page_text = page.extract_text() or ""
            # 2) 文本层为空 -> 疑似扫描件 -> 降级 OCR 识别该页图片
            if not page_text.strip() and ocr_available:
                try:
                    page_text = _ocr_page(page)
                except Exception as exc:
                    # 单页 OCR 失败不中断整体解析，记录空串继续后续页
                    print(f"[OCR] 第 {page.page_number} 页识别失败，跳过: {exc}")
                    page_text = ""
            pages_text.append(page_text)
    return "\n".join(pages_text)


def _check_tesseract_available() -> bool:
    """
    探测 Tesseract OCR 引擎是否可用。

    同时校验 Python 封装（pytesseract）与系统引擎本体；任一缺失则返回 False，
    避免在纯文本 PDF 解析路径上引入 ImportError 或引擎未找到错误。

    :return: True 表示可调用 Tesseract OCR
    """
    try:
        import pytesseract
        # 实际调用一次引擎版本接口，触发系统 Tesseract 可执行文件存在性校验
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        print("[OCR] Tesseract 未安装或未配置 PATH，扫描件 PDF 将跳过 OCR 降级")
        return False


def _ocr_page(page) -> str:
    """
    对单个 PDF 页面执行 Tesseract OCR 识别。

    将 pdfplumber 的 Page 对象按配置 DPI 渲染为 PIL 图像，再调用 pytesseract
    识别中文/英文文本。DPI 越高越清晰但越慢，默认 300 适合常规扫描件。

    :param page: pdfplumber 的 Page 对象
    :return: 该页识别到的文本
    """
    import pytesseract

    # to_image() 返回 PageImage，取 .original 拿到底层 PIL.Image.Image
    page_image = page.to_image(resolution=OCR_DPI).original
    # 调用 Tesseract 识别，lang 形如 "chi_sim+eng" 表示同时识别中英文
    return pytesseract.image_to_string(page_image, lang=OCR_LANG)


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
