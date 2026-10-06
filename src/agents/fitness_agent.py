"""
健身咨询专家
负责了解用户健身需求和基础情况
"""

import sys
import os
import asyncio

from langchain_core.messages import convert_to_messages, HumanMessage, AIMessage, SystemMessage

from src.state import AgentState
from src.utils.llm import LLM, parse_json_response

from src.rag.get_relevant_knowledge import get_relevant_knowledge
from src.tools.expert_tools import search_knowledge

from src.tools.mcp_client import load_mcp_tools

from src.utils.chat_history_utils import build_llm_messages

from src.utils.logger import get_logger
logger = get_logger(__name__)



FITNESS_EXPERT_PROMPT = """
你是一位有8年经验的健身教练，擅长为不同基础、不同身体条件的人制定训练计划。

你正在和用户进行多轮对话，了解用户的健身需求和基础情况。

【你的工作流程】
1. 仔细阅读用户说的话，提取所有有用信息
2. 基于用户提供的具体信息做需求分析（不是列总结，而是分析用户的情况和这些情况怎么影响训练）
3. 判断信息是否足够制定初步方案
   - 不够 → 有针对性地提问
   - 够了 → 给出带分析的详细目标规划，然后询问用户是否确认

【需求分析要求】（非常重要）
你不是在填表格列"目标：增肌、水平：新手"，而是在对应用户的具体信息做分析。

分析必须遵循这个结构：
  用户的具体信息 → 说明这对训练意味着什么 → 为什么

示例（用户说"我想增肌，新手，每周能练3天，家里有哑铃"）：
  ❌ 错误做法（列总结）：
  "目标总结：增肌。用户情况：新手，每周3天，家里有哑铃。"

  ✅ 正确做法（需求分析）：
  "根据你说的情况，我来梳理一下：
  你的主要目标是增肌——作为新手，你正处于'新手福利期'，前3个月进步会比较快，
  这是建立训练习惯和基础力量的好时机。
  你每周能练3天——这个频率对新手来说刚好，既能保证训练刺激，又有足够恢复时间，
  适合采用全身训练分化（每次练遍全身），而不是上下肢或推拉腿分化。
  你家里有哑铃——训练器械有限，但哑铃足够覆盖大部分复合动作，
  我会优先选择哑铃能完成的动作，避免需要杠铃或固定器械的计划。
  综合来看，你的条件很适合开始系统训练。"

需求分析的原则：
1. 必须引用用户说过的具体信息（"你提到..."、"你说的..."），不能凭空说
2. 说明每个信息对训练方案的影响（为什么这个条件要这样安排）
3. 如果用户有体态问题（从对话上下文中能看到），必须在分析中呼应——
   说明训练中会怎么考虑这些问题（比如"考虑到你有颈椎问题，我会避免颈后推举这类动作"）
4. 分析完后，自然过渡到"还需要确认什么"，然后提问

【信息足够时的回复要求】（非常重要）
当你认为信息足够时，你的 response 必须包含以下内容，用自然的语言串联，不要用方括号标签：

开头先做需求分析：
"根据你说的情况，我来梳理一下："
- 逐条对应用户的具体信息，分析目标、基础、可用条件、限制因素
- 说明每个因素怎么影响训练方案的设计
- 如果有体态问题，说明训练中会怎么考虑

然后给出训练方向建议：
"基于你的情况，训练方向上我建议..."
- 大致的训练方向（分化方式、动作选择原则、进阶思路）
- 说明为什么选这个方向（基于用户的什么情况）
- 不用给具体动作和组数，那是方案生成的事

接着说明预期效果：
"坚持下去的话，大致可以期待..."
- 大致的预期效果和时间周期
- 说明哪些效果是确定的、哪些需要坚持才能看到

最后确认：
"这个分析和你的情况对得上吗？有什么要补充或调整的？没问题的话我们开始制定训练方案。"

【参考的专业知识】
以下是从知识库检索到的、与本轮对话相关的专业资料，请结合它们来回答：

{knowledge_context}

【提问规则】
1. 一次只问1-2个问题，不要一次问太多
2. 问题要有针对性，是为了补全分析中还不确定的部分
3. 不要问用户已经说过的信息
4. 信息差不多够了就可以总结，不要为了问而问
5. 提问前先做一段需求分析，让用户知道你为什么问这些问题

【用户纠正处理规则】
用户可能会纠正之前说过的信息（比如"我刚才说想增肌，其实我想减脂"）。
处理原则：
1. 以用户最新的说法为准，之前的错误信息立即作废
2. 在 extracted_info 里，只保留最新的正确信息
3. 如果纠正影响了训练方向，要说明"根据你刚才的纠正，我调整一下分析"
4. 每轮输出的 extracted_info 必须是完整的、基于最新对话的全部信息

【输出格式】
你必须严格按照以下 JSON 格式输出，不要输出其他任何内容：

{{
  "analysis": "你的分析推理过程（内部思考，不展示给用户）",
  "extracted_info": {{
    "goals": ["增肌"],
    "fitness_level": "新手",
    "height": 175,
    "weight": 65,
    "body_fat": null,
    "training_years": 0,
    "weekly_days": 3,
    "session_duration": "45-60分钟",
    "training_location": "家里",
    "available_equipment": ["哑铃"],
    "injuries": [],
    "diet_preferences": []
  }},
  "need_more_info": true,
  "response": "给用户看的回复内容。必须使用 Markdown 格式（硬性要求，不要输出纯文本大段落）：小标题用###，重点内容用**加粗**，分点用-或数字列表。格式示范：### 训练方向\n**力量训练**是增肌核心，建议：\n- 每周3次复合动作\n- 渐进加重负荷。问诊阶段：先做需求分析，再提问。信息足够时：按需求分析、训练方向、预期效果、确认提问的顺序组织。"
效果对比
}}

【重要提醒】
- 用户已经说了一些信息了，不要假装没听到
- 先做需求分析（对应用户的具体信息），再提问
- 需求分析必须引用用户的具体信息，不能凭空说
- 如果用户有体态问题，必须在分析中呼应
- 信息足够时，response 里要包含完整的分析（需求分析、训练方向、预期效果），最后问用户是否确认
- 必须输出合法的 JSON，不要有多余的文字
- 不要用代码块包裹 JSON，直接输出合法 JSON（"response" 字段内容里可以使用 Markdown 格式）
- response 字段必须使用 Markdown 格式（### 小标题、**加粗**、- 或数字列表），不要输出纯文本大段落
- "response" 必须是 JSON 的最后一个键
"""

async def fitness_expert(state: AgentState, config) -> AgentState:
    """
    健身专家节点函数

    config 由 LangGraph 注入，必须透传给 LLM 调用，token 级流式才能生效
    """
    logger.info("【健身专家】开始分析...")

    # 初始化llm
    llm = LLM(temperature=0.3)

    # 加载MCP工具
    mcp_tools = await load_mcp_tools()

    chat_model = llm.get_chat_model().bind_tools(mcp_tools)

    # 获取历史对话（统一转成消息对象，兼容 dict 和对象两种写法）
    messages, new_summary = build_llm_messages(
        state.get("messages", []),
        state.get("chat_summary", ""),
        max_recent_turns=6,
    )
    user_input = state.get("user_input", "")

    # 把用户数据加到 messages 里
    if user_input and (not messages or messages[-1].type != "human"):
        # 用新列表，避免原地修改被 add_messages 重复追加
        messages = [*messages, HumanMessage(content=user_input)]
        state["messages"] = messages

    knowledge_context = ""
    try:
        # 决策提示词（健身教练版本）
        decision_prompt = (
            "你是一位有8年经验的健身教练。用户正在咨询健身需求。\n"
            "请判断：为了给出准确的分析，你是否需要检索专业知识库？\n"
            "- 需要检索：不确定训练方法、需要动作选择依据、用户提到不熟悉的术语。\n"
            "- 不需要检索：用户描述简单、你有充分把握、只是常规追问。\n"
            "如果需要检索，调用 search_knowledge 工具，关键词用专业术语。\n"
            "如果不需要，直接回复文字说明，不要调用工具。\n"
            "注意：只做是否检索的决策，不要给出最终训练建议。"
        )
        decision_messages = [SystemMessage(content=decision_prompt), *messages]

        decision_response = await chat_model.ainvoke(decision_messages)

        if decision_response.tool_calls:
            for tool_call in decision_response.tool_calls:
                if tool_call["name"] == "search_knowledge":
                    args = tool_call['args']
                    query = args.get("query", user_input)
                    top_k = args.get("top_k", 3)
                    logger.info(f"【健身专家】决策：需要检索，query={query}")
                    knowledge_context = search_knowledge.invoke(
                        {"query": query, "top_k": int(top_k)}
                    )
                    break
        else:
            logger.info("【健身专家】决策：不需要检索，基于已有经验分析")

    except Exception as e:
        logger.info(f"【健身专家】决策轮异常，降级为固定检索：{e}")
        knowledge_context = get_relevant_knowledge(user_input, category="fitness")

    if not knowledge_context:
        knowledge_context = "（本轮未检索知识库，基于已有经验分析）"
    

    # 获取相关知识
    system_prompt = FITNESS_EXPERT_PROMPT.format(knowledge_context=knowledge_context)

    # 调用llm（异步，token 级流式需要）
    response = await llm.achat_with_history(messages, system_prompt, config=config)

    # 解析json（LangChain 内置 JsonOutputParser）
    result = parse_json_response(
        response, 
        {
            "analysis": "解析失败",
            "extracted_info": {},
            "need_more_info": True,
            "response": "抱歉，我刚才有点走神，能再说一遍吗？"
        },
        repair_llm=llm.llm
    )

    # 更新State
    user_profile = state.get("user_profile", {})

    if "extracted_info" in result and result["extracted_info"]:
        fitness_info = user_profile.get("fitness", {})
        fitness_info.update(result["extracted_info"])
        user_profile["fitness"] = fitness_info

    # ===== 增量返回（重要！）=====
    # both 意图时体态/健身专家并行执行，LangGraph 要求并行节点返回的键互斥——
    # 同一键收到两个值会报 INVALID_CONCURRENT_GRAPH_UPDATE。
    # 所以这里只返回健身专家的私有键 + 有 reducer 的键（messages/user_profile），
    # current_response / current_stage / need_more_info / awaiting_confirmation
    # 这些共享键由 merge_expert_responses 汇聚时合并决定。
    updates = {
        "fitness_response": result.get("response", ""),  # 并行汇聚用
        "fitness_need_more_info": result.get("need_more_info", True),
        # 不在这里追加 messages，由汇聚节点统一追加
        "user_profile": user_profile,
    }

    # 如果信息够了，设置目标总结，并标记等待确认
    if not result.get("need_more_info", True):
        updates["fitness_goal_summary"] = result.get("response", "")
        updates["fitness_awaiting"] = "fitness"
        updates["fitness_analysis_confirmed"] = False



    logger.info(f"【健身专家】分析完成，需要更多信息：{result.get('need_more_info', True)}")

    return updates

# ==================== 测试 ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("健身专家多轮测试")
    logger.info("=" * 60)
    
    state = {
        "messages": [],
        "conversation_round": 0,
        "user_profile": {}
    }
    
    user_inputs = [
        "我想增肌",
        "身高175，体重65公斤，之前没怎么练过",
        "每周能练3天，家里有一对哑铃",
        "没有伤病，就是想变壮一点"
    ]
    
    for i, user_input in enumerate(user_inputs):
        logger.info(f"\n{'='*60}")
        logger.info(f"第 {i+1} 轮")
        logger.info(f"{'='*60}")
        logger.info(f"\n用户：{user_input}")
        
        state["messages"].append({
            "role": "user",
            "content": user_input
        })
        state["user_input"] = user_input
        
        state = asyncio.run(fitness_expert(state, None))
        
        logger.info(f"\n教练：{state.get('current_response')}")
        logger.info(f"\n需要更多信息：{state['need_more_info']}")
        logger.info(f"当前提取的信息：{state['user_profile'].get('fitness', {})}")
        
        if not state["need_more_info"]:
            logger.info(f"\n{'='*60}")
            logger.info("咨询完成！")
            logger.info(f"{'='*60}")
            logger.info(f"目标总结：{state.get('fitness_goal_summary')}")
            break
