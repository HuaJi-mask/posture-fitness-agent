"""
方案生成Agent
负责生成训练方案
"""

import sys
import os
import json
import asyncio
from src.state import AgentState
from src.utils.llm import LLM, parse_json_response, TEMP_DEFAULT

from src.utils.logger import get_logger
logger = get_logger(__name__)


PLAN_GENERATOR_PROMPT = """
你是一位专业的健身教练和康复师，擅长结合体态矫正和健身目标制定训练方案。

请根据用户的体态问题和健身目标，制定一份个性化的训练方案。

【输出格式】
你必须严格按照以下 JSON 格式输出：

{{
  "training_plan": "完整的训练计划，包括每周安排、具体动作、组数次数等",
  "diet_advice": "饮食建议",
  "precautions": "注意事项",
  "summary": "方案总结",
  "response": "给用户看的完整回复（包含训练方案、饮食建议、注意事项的完整展示文本）。必须使用 Markdown 格式（硬性要求）：训练计划用表格（| 动作 | 组数 | 次数 | 说明 |）或列表，小标题用###，重点内容用**加粗**"
}}

【注意】
- 训练方案要结合体态矫正和健身目标，不要割裂
- 动作要具体，要有组数次数
- 要考虑用户的基础和可用器械
- 必须输出合法的 JSON，不要有多余的文字
- "response" 必须是 JSON 的最后一个键
"""

async def plan_generator(state: AgentState, config) -> AgentState:
    """
    方案生成节点函数

    config 由 LangGraph 注入，必须透传给 LLM 调用，token 级流式才能生效
    """
    logger.info("【方案生成】开始生成方案...")

    # 初始化llm
    llm = LLM(temperature=TEMP_DEFAULT)

    # 收集信息
    user_profile = state.get("user_profile", {})
    posture_diagnosis = state.get("posture_diagnosis", "暂无体态诊断")
    correction_approach = state.get("correction_approach", "暂无矫正思路")
    fitness_goal_summary = state.get("fitness_goal_summary", "暂无健身目标总结")


    # 判断是否需要修改方案
    self_check_suggestions = state.get("self_check_suggestions", "")
    is_revision = bool(self_check_suggestions)

    if is_revision:
        # 修改轮：带上原方案和修改建议
        old_plan = state.get("training_plan", "")
        user_message = f"""
用户信息：
- 体态诊断：{posture_diagnosis}
- 矫正思路：{correction_approach}
- 健身目标：{fitness_goal_summary}
- 详细信息：{json.dumps(user_profile, ensure_ascii=False)}

原方案：
{old_plan}

自审发现的问题和修改建议：
{self_check_suggestions}

请根据以上修改建议，修改原方案，输出修改后的完整方案。
"""
        logger.info("【方案生成】自审修改轮，根据建议调整方案")
    else:
        # 首次生成：原来的逻辑
        user_message = f"""
用户信息：
- 体态诊断：{posture_diagnosis}
- 矫正思路：{correction_approach}
- 健身目标：{fitness_goal_summary}
- 详细信息：{json.dumps(user_profile, ensure_ascii=False)}

请根据以上信息，制定一份个性化的训练方案。
"""

    # 调用llm（异步，token 级流式需要）
    response = await llm.achat(user_message, PLAN_GENERATOR_PROMPT, config=config)

    # 解析json（LangChain 内置 JsonOutputParser）
    result = parse_json_response(
        response, 
        {
            "training_plan": response,
            "diet_advice": "暂无",
            "precautions": "暂无",
            "summary": "解析失败，使用原始回复"
        },
        repair_llm=llm.llm
    )

    # 更新状态（优先取 response 字段：它就是给用户看的完整文案，也是流式输出的内容）
    state["training_plan"] = result.get("training_plan", "")
    state["diet_plan"] = result.get("diet_advice", "")
    state["plan_summary"] = result.get("summary", "")
    state["current_stage"] = "planning"
    state["current_response"] = result.get("response") or result.get("training_plan", "")
    state["self_check_suggestions"] = ""  # 重置自审建议


    logger.info("【方案生成】方案生成完成")

    return state

# ==================== 测试 ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("方案生成测试")
    logger.info("=" * 60)
    
    # 构造测试数据
    state = {
        "user_profile": {
            "posture": {
                "problem_areas": [{"area": "颈椎", "symptoms": ["脖子疼"]}]
            },
            "fitness": {
                "goals": ["增肌"],
                "fitness_level": "新手",
                "weekly_days": 3
            }
        },
        "posture_diagnosis": "颈椎生理曲度变直，上交叉综合征倾向",
        "correction_approach": "松解紧张肌群 + 强化薄弱肌群 + 姿势矫正",
        "fitness_goal_summary": "增肌，新手，每周3天，居家哑铃训练"
    }
    
    # 调用方案生成
    state = asyncio.run(plan_generator(state, None))
    
    logger.info(f"\n训练计划：\n{state.get('training_plan')}")
    logger.info(f"\n饮食建议：\n{state.get('diet_plan')}")
    logger.info(f"\n注意事项：\n{state.get('precautions')}")
