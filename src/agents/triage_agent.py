"""
导诊Agent（意图模块）
负责分析用户意图，调度对应的专家
"""
import sys
import os
import asyncio

from langchain_core.messages import convert_to_messages, HumanMessage

from src.state import AgentState
from src.utils.llm import LLM, parse_json_response, TEMP_PRECISE

from src.utils.chat_history_utils import build_llm_messages

from src.utils.logger import get_logger
logger = get_logger(__name__)


TRIAGE_AGENT_PROMPT = """
你是一位导诊护士，负责判断用户的需求，调度对应的专家。

你需要分析用户的对话历史，判断：
1. 用户的主要意图是什么？（体态问题/健身需求/两者都有/讨论方案/不确定）
2. 这一轮应该让哪个专家说话？（体态专家/健身专家/两个都要说/生成方案）

【判断规则】
- 如果用户在说身体疼痛、体态问题、不舒服 → 体态专家
- 如果用户在说健身、增肌、减脂、训练 → 健身专家
- 如果用户既说了体态又说了健身 → 两个专家都要说
- 如果信息已经收集得差不多了，可以生成方案了 → 方案生成
- 如果不确定 → 先让体态专家说，或者再问问
- both 意图时，不需要你判断该跑哪个专家，系统会根据两边的问诊进度自动决定并行还是串行。你只需要判断 intent="both"，以及用户是否确认了/要改方案/否定了诊断。

【确认流程判断规则】（重要）
如果对话历史里，上一轮助手刚给出了详细的诊断分析或目标分析（结尾问用户"是否确认"），那么：

- 如果用户回复"好的/对的/没错/开始吧/没问题/可以/确认"等表示同意的话 → next_expert = "plan_generator"，直接开始生成方案
- 如果用户回复了新的症状、新的需求、补充信息（比如"我还有点腰疼""我每周只能练2天"）→ 回到对应的专家继续分析（体态问题回 posture_expert，健身问题回 fitness_expert）
- 如果用户否定了诊断（"不对/不是这样/我觉得不是"）→ 回到对应的专家重新分析
- 如果用户既没确认也没补充，只是闲聊 → 继续让对应的专家回应

【如何区分"生成方案"和"调整方案"（重要）】
- 生成方案：对话历史里还没有出现过训练方案，用户提出想制定方案
- 调整方案：对话历史里已经出现过训练方案或验证报告，用户针对方案内容提出具体修改意见。
  即使话里同时带新的身体状况描述（比如"我膝盖不好，把深蹲去掉"），只要核心意图是修改已有方案 → 方案调整
- 注意：有些话听起来像要求（"不要深蹲""换个动作"），但只要它讨论的是已有方案的内容，就属于方案调整，
  不要因为用户没明说"调整"两个字，就当成重新生成方案
- 无论是生成还是调整，next_expert 都填 "plan_generator"（合并节点，内部会自动判断模式）

【输出格式】
你必须严格按照以下 JSON 格式输出：

{{
  "intent": "posture",
  "next_expert": "posture_expert",
  "analysis": "用户主要在说颈椎疼的问题，属于体态问题，应该让体态专家继续问诊"
}}

【intent 可选值】
- posture：主要是体态问题
- fitness：主要是健身需求
- both：两者都有
- plan：讨论方案/要生成方案
- unknown：不确定

【next_expert 可选值】
- posture_expert：体态专家
- fitness_expert：健身专家
- both：两个都要
- plan_generator：方案生成或调整（首次生成方案、用户对已有方案提修改意见，都走这个节点）
- unknown：不确定

【注意】
- 仔细分析完整的对话历史，不要只看最后一句话
- 如果用户之前在说体态，现在转到健身了，要能发现
- 如果信息已经够了，可以建议生成方案
- 必须输出合法的 JSON
"""

async def triage_agent(state: AgentState, config) -> AgentState:
    """
    导诊节点函数

    config 由 LangGraph 注入，必须透传给 LLM 调用，token 级流式才能生效
    """
    logger.info("【导诊】开始分析用户意图...")

    # 初始化llm
    llm = LLM(temperature=TEMP_PRECISE)

    # 获取对话历史（统一转成消息对象，兼容 dict 和对象两种写法）
    messages, new_summary = build_llm_messages(
        state.get("messages", []),
        state.get("chat_summary", ""),
        max_recent_turns=6,
    )

    user_input = state.get("user_input", "")

    # 把用户输入加到 messages 里（如果有的话）
    if user_input and (not messages or messages[-1].type != "human"):
        # 用新列表，避免原地修改被 add_messages 重复追加
        messages = [*messages, HumanMessage(content=user_input)]
        state["messages"] = messages

    # 把对话历史转成文本
    history_text = ""
    for msg in messages:
        role = "用户" if msg.type == "human" else "医生"
        history_text += f"{role}: {msg.content}\n"

    # 获取当前是否在等待确认
    awaiting = state.get("awaiting_confirmation", "")
    awaiting_text = ""
    if awaiting == "posture":
        awaiting_text = "\n【当前状态】上一轮体态专家刚给出诊断分析，正在等待用户确认。"
    elif awaiting == "fitness":
        awaiting_text = "\n【当前状态】上一轮健身专家刚给出目标分析，正在等待用户确认。"
    elif awaiting == "both":
        awaiting_text = "\n【当前状态】上一轮体态专家和健身专家都刚给出分析，正在等待用户确认。"

    user_message = f"""
对话历史：
{history_text}
{awaiting_text}

请分析用户的意图，判断下一个该调用哪个专家。
"""

    # 调用大模型（异步，token 级流式需要）
    response = await llm.achat(user_message, TRIAGE_AGENT_PROMPT, config=config)

    # 解析JSON（LangChain 内置 JsonOutputParser）
    result = parse_json_response(
        response, 
        {
            "intent": "unknown",
            "next_expert": "posture_expert",
            "analysis": "解析失败，默认调用体态专家"
        },
        repair_llm=llm.llm
    )

    # 更新状态（增量返回，不返回整个 state）
    updates = {
        "intent": result.get("intent", "unknown"),
        "next_expert": result.get("next_expert", "posture_expert"),
        "current_stage": "triage",
        "conversation_round": state.get("conversation_round", 0) + 1,
        "messages": messages,
        "chat_summary": new_summary,
    }

    logger.info(f"【导诊Agent】意图：{updates['intent']}，下一个专家：{updates['next_expert']}")

    return updates

# ==================== 测试 ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("导诊Agent测试")
    logger.info("=" * 60)

    # 测试1：纯体态问题
    logger.info("\n--- 测试1：纯体态问题 ---")
    state1 = {
        "messages": [
            {"role": "user", "content": "我脖子疼"}
        ],
        "user_input": "我脖子疼"
    }
    state1 = asyncio.run(triage_agent(state1, None))
    logger.info(f"意图：{state1['intent']}")
    logger.info(f"下一个专家：{state1['next_expert']}")

    # 测试2：纯健身问题
    logger.info("\n--- 测试2：纯健身问题 ---")
    state2 = {
        "messages": [
            {"role": "user", "content": "我想增肌"}
        ],
        "user_input": "我想增肌"
    }
    state2 = asyncio.run(triage_agent(state2, None))
    logger.info(f"意图：{state2['intent']}")
    logger.info(f"下一个专家：{state2['next_expert']}")

    # 测试3：两者都有
    logger.info("\n--- 测试3：两者都有 ---")
    state3 = {
        "messages": [
            {"role": "user", "content": "我脖子疼，还想增肌"}
        ],
        "user_input": "我脖子疼，还想增肌"
    }
    state3 = asyncio.run(triage_agent(state3, None))
    logger.info(f"意图：{state3['intent']}")
    logger.info(f"下一个专家：{state3['next_expert']}")

    # 测试4：对已有方案提修改意见（历史里有方案，应识别为方案调整，走 plan_generator）
    logger.info("\n--- 测试4：方案修改意见 ---")
    state4 = {
        "messages": [
            {"role": "user", "content": "帮我制定一个训练方案"},
            {"role": "assistant", "content": "【训练方案】Day1：深蹲 3组10次，卧推 3组12次……（验证报告：风险等级 low）"},
            {"role": "user", "content": "我膝盖不太好，把深蹲去掉"}
        ],
        "user_input": "我膝盖不太好，把深蹲去掉"
    }
    state4 = asyncio.run(triage_agent(state4, None))
    logger.info(f"意图：{state4['intent']}")
    logger.info(f"下一个专家：{state4['next_expert']}")

