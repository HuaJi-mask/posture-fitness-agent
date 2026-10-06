"""
内容融合 Agent
负责把多个专家/验证器的输出融合成一段连贯自然的回复。
这是主流多 Agent 项目的通用做法（aggregator/editor 节点），
而不是简单拼接。
"""

import sys

from langchain_core.messages import AIMessage
from src.state import AgentState
from src.utils.llm import LLM, TEMP_CREATIVE, TEMP_DEFAULT

from src.utils.logger import get_logger
logger = get_logger(__name__)


# ==================== 专家回复融合 ====================
async def merge_expert_responses(state: AgentState, config) -> AgentState:
    """
    融合体态专家和健身专家的回复（fan-in）
    调用 LLM 把两段独立分析整合成一段连贯自然的回复。
    """
    logger.info("融合专家回复")

    posture_resp = state.get("posture_response", "")
    fitness_resp = state.get("fitness_response", "")

    updates = {
        "posture_response": "",
        "fitness_response": "",
    }

    posture_need = bool(state.get("posture_need_more_info", False))
    fitness_need = bool(state.get("fitness_need_more_info", False))
    is_diagnosis_phase = not(posture_need and fitness_need)

    if posture_resp and fitness_resp:
        # both 场景：LLM 融合
        phase_hint = (
            "当前是确诊阶段，要点明体态问题和健身目标的关联"
            if is_diagnosis_phase
            else "当前还在了解情况，要自然地引出需要确认的问题"
        )

        merge_prompt = f"""
你是一位专业的健康顾问，擅长把体态矫正和健身规划结合起来给用户建议。

请把以下两位专家的分析整合成一段连贯、自然的回复，要求：
1. 不要用【体态方面】【健身方面】这样的标签分割，用自然的语言过渡
2. {phase_hint}
3. 保留两位专家的核心信息和提问，不要遗漏
4. 语气亲切专业，像一个顾问在跟用户说话
5. 不要加额外的前言或总结，直接开始正文
6. 整合后的回复必须使用 Markdown 格式（硬性要求，不要输出纯文本大段落）：小标题用###，重点内容用**加粗**，分点用-或数字列表

【体态专家的分析】
{posture_resp}

【健身专家的分析】
{fitness_resp}

请输出融合后的完整回复：
"""
        try:
            llm = LLM(temperature=TEMP_CREATIVE)
            merged = await llm.achat(merge_prompt, config = config)
            merged = merged.strip()

            if not merged:
                raise ValueError("LLM 返回为空")
        except Exception as e:
            logger.info(f"融合专家回复失败，退回简单拼接：{e}")
            merged = f"体态方面：\n{posture_resp}\n\n健身方面：\n{fitness_resp}"

        updates["current_response"] = merged
        updates["current_stage"] = "both"

    elif posture_resp:
        updates["current_response"] = posture_resp
        updates["current_stage"] = "posture"
    elif fitness_resp:
        updates["current_response"] = fitness_resp
        updates["current_stage"] = "fitness"

    updates["need_more_info"] = posture_need or fitness_need

    posture_awaiting = state.get("posture_awaiting", "")
    fitness_awaiting = state.get("fitness_awaiting", "")

    if posture_awaiting and fitness_awaiting:
        updates["awaiting_confirmation"] = "both"
    elif posture_awaiting:
        updates["awaiting_confirmation"] = "posture"
    elif fitness_awaiting:
        updates["awaiting_confirmation"] = "fitness"

    final_response = updates.get("current_response", "")
    if final_response:
        updates["message"] = [AIMessage(content=final_response)]

    logger.info("【融合节点】专家回复融合完成")
    return updates


# ==================== 验证结果融合 ====================

async def merge_validation_results(state: AgentState, config) -> AgentState:
    """
    融合两个验证器的结果（含辩论历史）
    调用 LLM 把医生/教练的验证意见和辩论过程整合成一份连贯的验证报告。
    """
    logger.info("【融合节点】融合验证结果...")

    doctor_result = state.get("validation_a_result", {}) or {}
    coach_result = state.get("validation_b_result", {}) or {}
    debate_history = state.get("debate_history", []) or []
    debate_round = state.get("debate_round", 0)
    has_disagreement = state.get("has_disagreement", False)
    doctor_final = state.get("doctor_final", {}) or {}
    coach_final = state.get("coach_final", {}) or {}

    # 取最终结论（辩论后用 final，没辩论用原始 result）
    doctor_final_view = doctor_final if doctor_final else doctor_result
    coach_final_view = coach_final if coach_final else coach_result

    # 格式化辩论历史
    debate_text = ""
    if debate_history:
        debate_lines = []
        for item in debate_history:
            speaker = "医生" if item["speaker"] == "doctor" else "教练"
            content = item.get("content", "")
            agrees = "（认同对方）" if item.get("agrees") else ""
            debate_lines.append(f"第{item.get('round', '?')}轮 {speaker}{agrees}：{content}")

        debate_text = "\n".join(debate_lines)

    # 总体风险判断
    risk_level = doctor_final_view.get("risk_level", doctor_result.get("risk_level", "medium"))
    validation_passed = risk_level != "high"

    merge_prompt = f"""
你是一位严谨的医疗健康审核员，请把以下两位验证者的意见整合成一份清晰的验证报告，
并在最后加上"反思与改进"部分。

【医生验证意见（安全性视角）】
风险等级：{doctor_final_view.get('risk_level', '未知')}
是否需要就医：{doctor_final_view.get('need_medical_attention', '否')}
发现的问题：{doctor_final_view.get('issues', [])}
建议：{doctor_final_view.get('suggestions', [])}
总结：{doctor_final_view.get('summary', '')}

【教练验证意见（有效性视角）】
效果等级：{coach_final_view.get('effectiveness_level', '未知')}
发现的问题：{coach_final_view.get('issues', [])}
建议：{coach_final_view.get('suggestions', [])}
总结：{coach_final_view.get('summary', '')}

【辩论过程】（如果有）
{debate_text if debate_text else "无辩论，双方独立验证后意见一致"}

请输出一份连贯的验证报告，要求：
1. 开头先给总体结论（方案是否通过验证、总体风险等级）
2. 然后分别说明安全性评估和有效性评估
3. 列出发现的问题和改进建议
4. 如果有辩论，简要说明辩论的焦点和最终共识
5. 语气客观专业，让用户清楚知道方案是否安全有效
6. 使用 Markdown 格式（硬性要求）：小标题用###，重点内容用**加粗**，分点用-或数字列表，不要用 emoji
7. 反思与改进：
   - 这次辩论的核心分歧点是什么？
   - 为什么会产生这个分歧？（信息不足？视角差异？方案本身有缺陷？）
   - 最终是怎么解决的？
   - 这个方案还有哪些潜在问题或改进空间？
   - 下次制定类似方案时应该注意什么？

验证报告：
"""

    try:
        llm = LLM(temperature=TEMP_DEFAULT)
        final_report = await llm.achat(merge_prompt, config=config)
        final_report = final_report.strip()
        if not final_report:
            raise ValueError("LLM返回空")

        # 提取反思与改进部分
        reflection_summary = ""
        if "### 反思与改进" in final_report:
            parts = final_report.split("### 反思与改进", 1)
            if len(parts) > 1:
                reflection_summary = "### 反思与改进" +parts[1]
    except Exception as e:
        logger.info(f"【融合节点】验证报告LLM生成失败，退回硬编码：{e}")
        # 兜底：硬编码拼接
        all_issues = doctor_final_view.get("issues", []) + coach_final_view.get("issues", [])
        all_suggestions = doctor_final_view.get("suggestions", []) + coach_final_view.get("suggestions", [])
        final_report = (
            f"方案验证报告\n\n"
            f"总体结论：{'通过' if validation_passed else '高风险，建议修改'}\n"
            f"风险等级：{risk_level}\n\n"
            f"安全性评估：{doctor_final_view.get('summary', '')}\n\n"
            f"有效性评估：{coach_final_view.get('summary', '')}\n\n"
            f"发现的问题：\n" +
            ("\n".join([f"- {i}" for i in all_issues]) if all_issues else "- 暂无") +
            f"\n\n改进建议：\n" +
            ("\n".join([f"- {s}" for s in all_suggestions]) if all_suggestions else "- 暂无")
        )

    updates = {
        "validation_report": final_report,
        "validation_passed": validation_passed,
        "current_response": final_report,
        "current_stage": "validation",
        "messages": [AIMessage(content=final_report)],
        "reflection_summary": reflection_summary,
    }

    logger.info("【融合节点】验证结果融合完成")
    return updates


