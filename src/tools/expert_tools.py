import sys
import os


from langchain_core.tools import tool
from src.rag.get_relevant_knowledge import get_relevant_knowledge

from src.utils.logger import get_logger
logger = get_logger(__name__)


@tool
def search_knowledge(query: str, top_k: int = 3) -> str:
    """
    从体态/健身知识库检索相关专业知识。

    当需要专业知识支撑分析、不确定症状成因、想查矫正方法、
    用户提到不熟悉的专业术语时调用。

    Args:
        query: 检索关键词，要用专业术语，比如"上交叉综合征的成因和矫正方法"，
               不要用太泛的词如"脖子疼"
        top_k: 返回最相关的几条，默认3条，简单问题用2条，复杂问题用5条

    Returns:
        检索到的知识文本，未命中时返回提示信息
    """
    result = get_relevant_knowledge(query, k=top_k)

    if not result or result == "":
        return "未命中相关知识"
    return "这是检索到的资料：" + result




if __name__ == "__main__":
    result = search_knowledge.invoke({"query": "上交叉综合征的成因和矫正方法", "top_k": 2})
    logger.info(result)