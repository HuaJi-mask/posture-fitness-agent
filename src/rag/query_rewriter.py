"""
查询改写模块
把用户口语化的问题改写成更适合检索的查询
"""
from src.utils.llm import LLM

from src.utils.logger import get_logger
logger = get_logger(__name__)


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLM(temperature=0.1)
    return _llm


REWRITE_PROMPT = """你是一个检索查询改写专家。请把用户的问题改写成更适合知识库检索的查询。

要求：
1. 保留核心意图
2. 补充相关的专业术语和关键词
3. 去掉口语化的表达
4. 只输出改写后的查询，不要解释

用户问题：{query}

改写后的查询："""

def rewrite_query(query: str) -> str:
    """
    把用户原始查询改写成更适合检索的查询
    
    Args:
        query: 用户原始问题
    
    Returns:
        改写后的查询字符串
    """
    llm = get_llm()

    prompt = REWRITE_PROMPT.format(query=query)

    try:
        rewritten = llm.chat(prompt)
        # 去掉可能的换行和多余空格
        rewritten = rewritten.strip()
        # 如果改写结果太短，说明失败了，用原始查询
        if len(rewritten) < 2:
            return query
        logger.info(f"  [查询改写] {query} → {rewritten}")
        return rewritten
    except Exception as e:
        logger.info(f"  [查询改写失败] {e}，使用原始查询")
        return query



def make_query_standalone(query: str, chat_history: list = None) -> str:
    """
    把当前问题结合对话历史，改写成独立的、不需要上下文也能理解的查询
    
    Args:
        query: 当前用户问题
        chat_history: 对话历史，格式 [("user", "..."), ("assistant", "...")]
    
    Returns:
        独立查询字符串
    """
    # 如果没有对话历史，直接返回原问题
    if not chat_history:
        return query
    
    # 构造上下文
    chat_history_text = "\n".join([f"{role}: {content}" for role, content in chat_history])

    prompt_template = """
你是一个对话理解专家。请结合对话历史，把用户当前的问题改写成一个独立的、不需要上下文也能理解的查询。

要求：
1. 把"它"、"这个"、"那个"等代词替换成具体的实体
2. 把省略的信息补全
3. 只输出改写后的查询，不要解释
4. 如果当前问题已经很完整，直接返回原问题

对话历史：
{chat_history_text}

当前用户问题：{query}

独立查询：
"""
    prompt = prompt_template.format(query=query, chat_history_text=chat_history_text)

    llm = get_llm()
    standalone = llm.chat(prompt)
    standalone = standalone.strip()

    return standalone