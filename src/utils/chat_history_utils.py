"""对话历史工具模块"""

from langchain_core.messages import convert_to_messages, SystemMessage
from src.utils.llm import LLM, TEMP_PRECISE
from typing import Optional


def summarize_old_messages(old_messages: list, existing_summary: str = "") -> str:
    """
    把旧对话压缩成摘要
    如果已有摘要，就增量更新（在旧摘要基础上追加新内容）
    
    Args:
        old_messages: 需要压缩的旧对话消息列表
        existing_summary: 已有的对话摘要（可选）
    
    Returns:
        更新后的摘要字符串
    """
    
    if not old_messages:
        return existing_summary
    
    # 合并所有消息为一个字符串
    old_messages_text = "\n".join([
        f"{'用户' if m.type == 'human' else '助手'}: {m.content}"
        for m in old_messages
    ])

    prompt = f"""
    你是一个对话摘要助手。请把以下对话内容压缩成简洁的摘要，保留关键信息（症状、诊断、偏好、重要决定）。

已有摘要：
{existing_summary}

新增对话：
{old_messages_text}

请更新摘要，保留所有关键信息：

    """
    if llm is None:
        llm = LLM(temperature=TEMP_PRECISE)


    # 生成摘要
    summary = llm.chat(prompt)

    
    return summary.strip()


def build_llm_messages(state_messages: list, chat_summary: str, max_recent_turns: int = 6, llm: Optional[LLM] = None) -> tuple[list, str]:
    """
    构建传给 LLM 的消息列表：摘要(SystemMessage) + 最近 N 轮原文
    
    Args:
        state_messages: State 里的完整 messages（原始数据，不改）
        chat_summary: 已有的对话摘要
        max_recent_turns: 保留最近几轮原文
    
    Returns:
        (裁剪后的消息列表, 更新后的摘要)
    """
    state_messages = convert_to_messages(state_messages)

    # 计算需要保留多少条消息
    max_recent_messages = max_recent_turns * 2

    if len(state_messages) < max_recent_messages:
        return state_messages, chat_summary

    recent_messages = state_messages[-max_recent_messages:]
    old_messages = state_messages[:-max_recent_messages]

    existing_summary = chat_summary
    # 压缩旧对话
    new_summary = summarize_old_messages(old_messages, existing_summary)

    llm_messages = [SystemMessage(content=new_summary)] + recent_messages

    return llm_messages, new_summary


    
