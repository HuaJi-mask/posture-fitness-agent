"""
Rerank 重排序模块
对混合检索召回的候选文档进行精排
"""
import os

# 在 import 任何 huggingface 相关库之前，先设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# 初始化
_reranker = None

def get_reranker():
    """懒加载 reranker 模型（第一次调用时才加载本地缓存）"""
    global _reranker
    if _reranker is None:
        # 直接用本地缓存路径，避免联网验证
        local_path = r"C:\Users\86183\.cache\huggingface\hub\models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
        _reranker = CrossEncoder(local_path)
    return _reranker

def rerank(query: str, docs:list[Document], top_k: int = 4) -> list[Document]:
    """
    对候选文档重排序
    
    Args:
        query: 用户查询
        docs: 混合检索召回的候选文档列表
        top_k: 返回前几条
    
    Returns:
        重排序后的文档列表
    """
    if not docs:
        return []

    reranker = get_reranker()

    # 构造(query, doc_text) 对
    pairs = [(query, doc.page_content) for doc in docs]

    # 批量打分
    scores = reranker.predict(pairs)

    # 按分数降序排序
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_k]]


