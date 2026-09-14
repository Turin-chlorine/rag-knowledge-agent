"""
FastAPI 主应用
提供文档上传/管理、知识库问答、统计查询等 REST 接口，并托管前端静态页面。

启动方式（项目根目录）：
    uvicorn backend.main:app --reload
"""
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

# 将 backend 目录加入模块搜索路径，
# 保证从项目根目录启动（uvicorn backend.main:app）时，
# 能正确导入同目录下的 agent / config / embedding 等本地模块
BACKEND_DIR = str(Path(__file__).resolve().parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import agent_answer
from config import ALLOWED_EXTENSIONS, CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR
from document_parser import parse_document
from embedding import embed_texts, get_model
from text_splitter import split_text_into_chunks
from vector_store import DocumentChunk, DocumentMeta, new_id, vector_store

app = FastAPI(title="RAG 知识库 AI Agent", version="1.0.0")

# Embedding 模型就绪状态标记（首次运行需联网下载模型，下载期间为 False）
_model_ready = False


def _preload_embedding_model() -> None:
    """
    后台线程预加载 Embedding 模型。
    首次运行会触发模型下载（走 HF_ENDPOINT 镜像），进度打印在后端控制台，
    避免把下载耗时推迟到用户上传文档时导致页面卡住。
    """
    global _model_ready
    try:
        get_model()
        _model_ready = True
    except Exception as exc:  # 下载/加载失败时打印错误，服务仍可启动
        print(f"[Embedding] 模型加载失败: {exc}")


# 服务启动时在后台线程预加载模型（不阻塞服务启动）
threading.Thread(target=_preload_embedding_model, daemon=True).start()

# 允许前端跨域访问（本地开发兜底）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    """问答接口请求体"""
    question: str  # 用户提问内容


# ---------------- 系统状态接口 ----------------

@app.get("/api/health")
async def health():
    """
    返回服务与 Embedding 模型就绪状态。
    前端轮询该接口，在首次启动下载模型期间给出友好提示。
    """
    return {
        "status": "ok",
        "model_ready": _model_ready,
    }


# ---------------- 文档管理接口 ----------------

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile):
    """
    上传文档：保存文件 -> 解析文本 -> 切片 -> 向量化 -> 入库。

    :param file: 前端上传的文件（PDF/TXT/MD）
    :return: 入库结果摘要
    """
    # 模型未就绪（首次启动仍在下载）时直接返回明确提示，避免请求长时间挂起
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Embedding 模型尚未加载完成（首次启动需联网下载，请查看后端控制台进度），请稍后重试",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 PDF / TXT / Markdown 文件")

    # 1. 保存原始文件到 uploads 目录
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = new_id()
    save_path = UPLOAD_DIR / f"{doc_id}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 2. 解析为纯文本
        text = parse_document(save_path)
        if not text.strip():
            raise ValueError("文档内容为空或无法解析")

        # 3. 文本切片
        chunk_texts = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunk_texts:
            raise ValueError("切片结果为空，文档可能过短")

        # 4. 批量向量化
        vectors = embed_texts(chunk_texts)

        # 5. 组装切片记录并写入向量库
        chunks = [
            DocumentChunk(
                chunk_id=new_id(),
                doc_id=doc_id,
                doc_name=file.filename,
                content=chunk_text,
                chunk_index=index,
                vector=vector,
            )
            for index, (chunk_text, vector) in enumerate(zip(chunk_texts, vectors))
        ]
        meta = DocumentMeta(
            doc_id=doc_id,
            doc_name=file.filename,
            chunk_count=len(chunks),
            char_count=len(text),
            upload_time=datetime.now().isoformat(timespec="seconds"),
        )
        vector_store.add_document(meta, chunks)
    except Exception as exc:
        # 解析/向量化失败时清理已保存的文件
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"文档处理失败: {exc}") from exc

    return {
        "doc_id": doc_id,
        "doc_name": file.filename,
        "chunk_count": len(chunks),
        "char_count": len(text),
    }


@app.get("/api/documents")
async def list_documents():
    """获取知识库统计信息与文档列表"""
    return vector_store.stats()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    删除指定文档及其全部切片。

    :param doc_id: 文档 ID
    """
    removed = vector_store.remove_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"doc_id": doc_id, "removed_chunks": removed}


# ---------------- 问答接口 ----------------

@app.post("/api/chat")
async def chat(request: QuestionRequest):
    """
    知识库问答：Agent 先检索文档片段，再调用 DeepSeek 生成回答。

    :param request: 包含用户问题的请求体
    :return: 回答、引用来源、检索片段数
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        return agent_answer(question)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------- 前端静态页面托管 ----------------

# 根路径返回前端主页
@app.get("/")
async def index():
    """返回前端入口页面 index.html"""
    return FileResponse(Path(__file__).resolve().parent.parent / "frontend" / "index.html")


# 托管 frontend 目录下的静态资源（JS/CSS 等）
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend"), name="static")
