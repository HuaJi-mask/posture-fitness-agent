"""
双验证机制
医生视角 + 教练视角，双重验证方案的安全性和有效性
"""
import sys
import os
import json
import asyncio

from langchain_core.messages import convert_to_messages

from src.state import AgentState
from src.utils.llm import LLM, parse_json_response, TEMP_DEFAULT

from src.utils.logger import get_logger
logger = get_logger(__name__)


# ==================== 验证器A：医生视角 ====================

DOCTOR_VALIDATOR_PROMPT = """
你是一位有10年经验的康复科医生，负责检查训练方案的安全性。

你会看到：
- 完整的问诊对话记录
- 体态专家给出的诊断结论
- 用户画像
- 待验证的训练方案

请做双重审核：

【第一部分：回溯审核诊断】
1. 体态专家的诊断合理吗？有没有医学依据？
2. 有没有遗漏的问题？（比如用户提到了某个症状但专家没跟进）
3. 用户的描述有没有被误解或过度解读？
4. 如果诊断有问题，请指出来。
5. 训练方案是否符合用户的体态问题和伤病情况？
6. 是否对用户的体态诊断有错误的分析？

【第二部分：审核训练方案安全性】
1. 有没有不安全的动作？
2. 会不会加重用户的体态问题或伤病？
3. 动作安排合理吗？有没有风险？
4. 有什么需要注意的地方？

【输出格式】
你必须严格按照以下 JSON 格式输出：

{{
  "is_safe": true,
  "risk_level": "low",
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "summary": "总体评价",
  "response": "给用户看的验证结论摘要"
}}

【risk_level 可选值】low / medium / high

【重要提示】
- 你是独立审核者，不要默认前面的诊断一定正确
- 要基于原始对话记录自己判断
- 如果发现前面的诊断有严重问题，要明确指出来
- 要结合用户的体态问题和伤病情况来判断
- 要具体，不要泛泛而谈
- 必须输出合法的 JSON
- "response" 必须是 JSON 的最后一个键
"""

# 辩论模式的 Prompt（医生看到教练意见后）
DOCTOR_DEBATE_MODE_PROMPT = """
你是一位有10年经验的康复科医生。

你之前已经对训练方案做了安全性验证，现在教练提出了他的看法。
请你重新审视你的结论，回应教练的意见：

1. 你需要验证教练的意见和分析结果是否存在错误之处，是否遗漏、忽略了某些重要信息？
2. 教练提出的问题，你认同吗？如果认同，请修正你的结论。
3. 如果你不认同，请给出你的反驳理由和医学依据。
4. 有没有之前没考虑到的新问题？
5. 最终，你对这个方案的安全性结论是什么？
6. 对用户自身情况的分析正确吗？

【输出格式】
你必须严格按照以下 JSON 格式输出：

{{
  "is_safe": true,
  "risk_level": "low",
  "issues": ["修正后的问题列表"],
  "suggestions": ["修正后的建议列表"],
  "summary": "最终结论总结",
  "agrees_with_coach": true,
  "rebuttal": "对教练意见的反驳（如果不同意，没有就填空）",
  "response": "给用户看的回应摘要"
}}

【注意】
- 要客观，不要为了赢而赢，该认同就认同
- 要有依据，不要空泛反驳
- 必须输出合法的 JSON
- "response" 必须是 JSON 的最后一个键
"""

async def doctor_validator(state: AgentState, config) -> AgentState:
    """
    医生视角验证器（支持独立验证和辩论两种模式）
    合并了原来的 enter_debate 节点逻辑：辩论轮次+1

    config 由 LangGraph 注入，必须透传给 LLM 调用，token 级流式才能生效
    """
    coach_result = state.get("validation_b_result", {})

    # ===== 合并 enter_debate 逻辑：有教练验证结果说明是辩论轮，轮次+1 =====
    if coach_result:
        current_round = state.get("debate_round", 0)
        if current_round == 0:
            state["debate_history"] = []  # 首次进入辩论，清空历史
        state["debate_round"] = current_round + 1
        logger.info(f"【医生验证器】进入第 {state['debate_round']} 轮辩论")

    debate_round = state.get("debate_round", 0)

    # 判断模式
    is_debate_mode = debate_round > 0 and coach_result

    if is_debate_mode:
        logger.info(f"【医生验证器】第 {debate_round} 轮辩论模式")
        prompt = DOCTOR_DEBATE_MODE_PROMPT
    else:
        logger.info("【医生验证器】独立验证模式")
        prompt = DOCTOR_VALIDATOR_PROMPT

    llm = LLM(temperature=TEMP_DEFAULT)

    # 收集信息
    user_profile = state.get("user_profile", {})
    training_plan = state.get("training_plan", "")
    posture_diagnosis = state.get("posture_diagnosis", "")
    doctor_original = state.get("validation_a_result", {})
    debate_history = state.get("debate_history", [])
    messages = convert_to_messages(state.get("messages", []))

    # 把对话记录格式化成字符串
    conversation_history = "\n".join([
        f"{'用户' if msg.type == 'human' else '助手'}：{msg.content}"
        for msg in messages[-20:]
    ])

    if is_debate_mode:
        # 辩论模式：带上自己之前的结论 + 教练的结论 + 辩论历史
        debate_context = "\n".join([
            f"【第{history['round']}轮】{history['speaker']}：{history['content']}"
            for history in debate_history
        ])

        user_message = f"""
【完整问诊对话记录】
{conversation_history}

【用户情况】
【体态诊断】{posture_diagnosis}
【详细信息】{json.dumps(user_profile, ensure_ascii=False)}

【训练方案】
{training_plan}

【你之前的验证结论】
风险等级：{doctor_original.get('risk_level', '')}
问题：{json.dumps(doctor_original.get('issues', []), ensure_ascii=False)}
建议：{json.dumps(doctor_original.get('suggestions', []), ensure_ascii=False)}

【教练的验证结论】
效果等级：{coach_result.get('effectiveness_level', '')}
问题：{json.dumps(coach_result.get('issues', []), ensure_ascii=False)}
建议：{json.dumps(coach_result.get('suggestions', []), ensure_ascii=False)}

【之前的辩论记录】
{debate_context if debate_context else "（这是第一轮辩论）"}

请回应教练的意见，重新审视你的结论。
"""
    else:
        # 独立验证模式
        user_message = f"""
【用户情况】
【体态诊断】{posture_diagnosis}
【详细信息】{json.dumps(user_profile, ensure_ascii=False)}

【训练方案】
{training_plan}

请从康复医学的角度，检查这个方案是否安全。
1. 回溯审核：体态专家的诊断合理吗？有没有遗漏或误判？用户的描述有没有被误解？
2. 方案审核：这个训练方案安全吗？会不会加重问题？
"""

    response = await llm.achat(user_message, prompt, config=config)

    result = parse_json_response(
        response, 
        {
            "is_safe": True,
            "risk_level": "medium",
            "issues": [],
            "suggestions": ["解析失败，建议人工检查"],
            "summary": "解析失败",
            "agrees_with_coach": False,
            "rebuttal": ""
        },
        repair_llm=llm.llm
    )

    # 保存结果（辩论模式覆盖之前的，独立模式新建）
    state["validation_a_result"] = result
    state["current_stage"] = "validation" if not is_debate_mode else "debate"
    state["current_response"] = result.get("response", "")

    # 辩论模式下记录辩论历史
    if is_debate_mode:
        debate_history.append({
            "round": debate_round,
            "speaker": "doctor",
            "content": result.get("response", ""),
            "agrees": result.get("agrees_with_coach", False)
        })
        state["debate_history"] = debate_history

    logger.info(f"【医生验证器】完成，风险等级：{result.get('risk_level', 'unknown')}")

    return state

# ==================== 验证器B：教练视角 ====================

COACH_VALIDATOR_PROMPT = """
你是一位有8年经验的健身教练，负责检查训练方案的有效性。

你会看到：
- 完整的问诊对话记录
- 健身专家总结的目标和需求
- 用户画像
- 待验证的训练方案

请做双重审核：

【第一部分：回溯审核需求分析】
1. 健身专家对用户目标的理解准确吗？
2. 有没有遗漏用户提到的限制条件（伤病、时间、设备等）？
3. 用户的水平判断合理吗？
4. 你的训练方案是否符合用户的水平和目标？

【第二部分：审核训练方案有效性】
1. 训练安排科学吗？符合增肌/减脂的原理吗？
2. 动作选择合理吗？有没有遗漏的部位？
3. 组数次数安排合适吗？
4. 训练频率和恢复时间合理吗？
5. 对用户的水平来说难度合适吗？


【输出格式】
你必须严格按照以下 JSON 格式输出：

{{
  "is_effective": true,
  "effectiveness_level": "good",
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "summary": "总体评价",
  "response": "给用户看的验证结论摘要"
}}

【effectiveness_level 可选值】excellent / good / fair / poor

【注意】
- 要结合用户的水平和目标来判断
- 要具体，不要泛泛而谈
- 必须输出合法的 JSON
- "response" 必须是 JSON 的最后一个键
"""

COACH_DEBATE_MODE_PROMPT = """
你是一位有8年经验的健身教练。

你之前已经对训练方案做了有效性验证，现在医生提出了他的看法。
请你重新审视你的结论，回应医生的意见：

1. 你需要验证医生的意见和分析结果是否存在错误之处，是否遗漏、忽略了某些重要信息？
2. 医生提出的安全顾虑，你认同吗？如果认同，请修正你的训练建议。
3. 如果你不认同，请给出你的反驳理由和训练学依据。
4. 有没有之前没考虑到的新问题？
5. 最终，你对这个方案的有效性结论是什么？
6. 对用户自身情况的分析正确吗？

【输出格式】
你必须严格按照以下 JSON 格式输出：

{{
  "is_effective": true,
  "effectiveness_level": "good",
  "issues": ["修正后的问题列表"],
  "suggestions": ["修正后的建议列表"],
  "summary": "最终结论总结",
  "agrees_with_doctor": true,
  "rebuttal": "对医生意见的反驳（如果不同意，没有就填空）",
  "response": "给用户看的回应摘要"
}}

【重要提示】
- 你是独立审核者，不要默认前面的分析一定正确
- 要基于原始对话记录自己判断
- 要客观，不要为了赢而赢，该认同就认同
- 安全第一，如果医生说有风险，要认真考虑
- 必须输出合法的 JSON
- "response" 必须是 JSON 的最后一个键
"""

async def coach_validator(state: AgentState, config) -> AgentState:
    """
    教练视角验证器

    config 由 LangGraph 注入，必须透传给 LLM 调用，token 级流式才能生效
    """
    debate_round = state.get("debate_round", 0)
    doctor_result = state.get("validation_a_result", {})

    # 判断模式：有医生意见且不是第一轮，就是辩论模式
    is_debate_mode = debate_round > 0

    if is_debate_mode:
        logger.info(f"【教练验证器】第 {debate_round} 轮辩论模式")
        prompt = COACH_DEBATE_MODE_PROMPT
    else:
        logger.info("【教练验证器】独立验证模式")
        prompt = COACH_VALIDATOR_PROMPT

    llm = LLM(temperature=TEMP_DEFAULT)

    # 收集信息
    user_profile = state.get("user_profile", {})
    training_plan = state.get("training_plan", "")
    fitness_goal_summary = state.get("fitness_goal_summary", "")
    coach_original = state.get("validation_b_result", {})
    debate_history = state.get("debate_history", [])
    messages = convert_to_messages(state.get("messages", []))

    # 把对话记录格式化成字符串
    conversation_history = "\n".join([
        f"{'用户' if msg.type == 'human' else '助手'}：{msg.content}"
        for msg in messages[-20:]
    ])

    if is_debate_mode:
        debate_context = "\n".join(
            f"【第{history['round']}轮】{history['speaker']}：{history['content']}"
            for history in debate_history
        )

        user_message = f"""
【用户情况】

【完整问诊对话记录】
{conversation_history}

【健身目标】{fitness_goal_summary}
【详细信息】{json.dumps(user_profile, ensure_ascii=False)}

【训练方案】
{training_plan}

【你之前的验证结论】
效果等级：{coach_original.get('effectiveness_level', '')}
问题：{json.dumps(coach_original.get('issues', []), ensure_ascii=False)}
建议：{json.dumps(coach_original.get('suggestions', []), ensure_ascii=False)}

【医生的验证结论】
风险等级：{doctor_result.get('risk_level', '')}
问题：{json.dumps(doctor_result.get('issues', []), ensure_ascii=False)}
建议：{json.dumps(doctor_result.get('suggestions', []), ensure_ascii=False)}

【之前的辩论记录】
{debate_context}

请回应医生的意见，重新审视你的结论。记住：安全第一。
"""
    else:
        user_message = f"""
【用户情况】
【健身目标】{fitness_goal_summary}
【详细信息】{json.dumps(user_profile, ensure_ascii=False)}

【训练方案】
{training_plan}

请从专业训练的角度，检查这个方案是否有效。
"""

    response = await llm.achat(user_message, prompt, config=config)

    # 解析JSON
    result = parse_json_response(
        response, 
        {
            "is_effective": True,
            "effectiveness_level": "fair",
            "issues": [],
            "suggestions": ["解析失败，建议人工检查"],
            "summary": "解析失败",
            "agrees_with_doctor": False,
            "rebuttal": ""
        },
        repair_llm=llm.llm
    )

    state["validation_b_result"] = result
    state["current_stage"] = "validation" if not is_debate_mode else "debate"
    state["current_response"] = result.get("response", "")

    if is_debate_mode:
        debate_history.append({
            "round": debate_round,
            "speaker": "coach",
            "content": result.get("response", ""),
            "agrees": result.get("agrees_with_doctor", False)
        })
        state["debate_history"] = debate_history

    logger.info(f"【教练验证器】完成，有效性等级：{result.get('effectiveness_level', 'unknown')}")
    return state


# ==================== 分歧判断函数 ====================

def check_disagreement(state: AgentState) -> bool:
    """判断是否有分歧，需要进入辩论"""
    doctor_result = state.get("validation_a_result", {})
    coach_result = state.get("validation_b_result", {})

    doctor_risk = doctor_result.get("risk_level", "medium")
    doctor_issues = doctor_result.get("issues", [])
    coach_issues = coach_result.get("issues", [])

    # 高风险必须辩论
    if doctor_risk == "high":
        return True

    # 双方都有问题，可能需要讨论
    if len(doctor_issues) > 0 and len(coach_issues) > 0:
        return True

    # 任意一方问题较多
    if len(doctor_issues) >= 2 or len(coach_issues) >= 2:
        return True

    return False


def check_debate_should_continue(state: AgentState) -> str:
    """判断辩论是否应该继续，返回'continue'/'stop'"""
    debate_round = state.get("debate_round", 1)
    doctor_result = state.get("validation_a_result", {})
    coach_result = state.get("validation_b_result", {})

    # 超过三轮，强制结束
    if debate_round >= 3:
        return "stop"

    # 双方都认同对方，达成共识
    if doctor_result.get("agrees_with_coach", False) and coach_result.get("agrees_with_doctor", False):
        return "stop"

    return "continue"


# ==================== 测试 ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("双验证机制测试")
    logger.info("=" * 60)

    # 构造测试数据
    state = {
        "user_profile": {
            "posture": {
                "problem_areas": [{"area": "颈椎", "symptoms": ["脖子疼"]}]
            },
            "fitness": {
                "goals": ["增肌"],
                "level": "新手"
            }
        },
        "posture_diagnosis": "颈椎生理曲度变直，上交叉综合征",
        "fitness_goal_summary": "增肌，新手，每周3天",
        "training_plan": """
每周3天训练：
Day 1：哑铃深蹲 3组10次，哑铃卧推 3组12次，哑铃划船 3组12次
Day 2：硬拉 3组8次，哑铃推举 3组10次，哑铃弯举 3组15次
Day 3：...
        """
    }

    # 医生验证
    state = asyncio.run(doctor_validator(state, None))
    logger.info(f"\n医生验证：风险等级 = {state['validation_a_result']['risk_level']}")

    # 教练验证
    state = asyncio.run(coach_validator(state, None))
    logger.info(f"教练验证：效果等级 = {state['validation_b_result']['effectiveness_level']}")
