
from src.rag.knowledge_base import KnowledgeBase

from src.rag.query_rewriter import rewrite_query

from src.rag.query_rewriter import make_query_standalone

_kb = None  # 知识库实例，第一次调用时才初始化


def get_relevant_kb():
    """获取知识库单例"""
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
        _kb.load()
    return _kb


def get_relevant_knowledge(query: str, k: int = 4, category: str = None, chat_history: list = None) -> str:
    """
    从知识库中检索与查询相关的内容
    
    Args:
        query: 用户查询
        k: 返回结果数
        category: 可选，按分类过滤（posture/fitness/both）
    """
    kb = get_relevant_kb()

    standalone_query = make_query_standalone(query, chat_history)

    # 重写query
    rewritten_query = rewrite_query(standalone_query)

    # 混合检索+rerank，带分类过滤
    results = kb.hybrid_search(rewritten_query, k=k, category=category)
    search_results = [doc.page_content if hasattr(doc, "page_content") else doc for doc in results]
    knowledge_context = "\n\n".join(search_results)
    return knowledge_context


