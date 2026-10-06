"""
主状态机（深度合并版）

节点（8个）：
1. triage - 导诊/意图识别
2. posture_expert - 体态专家（ReAct模式）
3. fitness_expert - 健身专家（ReAct模式）
4. plan_generator - 方案生成（合并了首次生成+用户调整+自审循环）
5. doctor_validator - 医生验证器（合并了enter_debate逻辑）
6. coach_validator - 教练验证器
7. merge_validation - 验证结果整合
8. merge_expert_responses - 专家回复汇聚（both并行场景）
"""
import sys
import os
import asyncio


from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END, START

from src.state import AgentState
from src.agents.triage_agent import triage_agent
from src.agents.posture_agent import posture_expert
from src.agents.fitness_agent import fitness_expert
from src.agents.plan_generator import plan_generator
from src.agents.validators import (
    check_disagreement,
    check_debate_should_continue,
    doctor_validator,
    coach_validator,
)
from src.agents.merge_agent import merge_expert_responses, merge_validation_results

from src.utils.logger import get_logger
logger = get_logger(__name__)



def build_main_graph():
    """构建主图"""

    graph = StateGraph(AgentState)

    # ===== 添加节点（8个）=====
    graph.add_node("triage", triage_agent)
    graph.add_node("posture_expert", posture_expert)
    graph.add_node("fitness_expert", fitness_expert)
    graph.add_node("plan_generator", plan_generator)
    graph.add_node("doctor_validator", doctor_validator)
    graph.add_node("coach_validator", coach_validator)
    graph.add_node("merge_validation", merge_validation_results)
    graph.add_node("merge_expert_responses", merge_expert_responses)

    # ===== 路由函数 =====

    def route_after_triage(state: AgentState) -> str:
        """根据 triage 结果，选择后续节点"""
        next_expert = state.get("next_expert", "posture_expert")
        intent = state.get("intent", "unknown")

        # 方案生成或调整（合并后统一走 plan_generator）
        if next_expert in ("plan_generator", "plan_adjuster"):
            return "plan_generator"

        # both 意图：根据问诊进度决定并行还是串行
        if intent == "both" or next_expert == "both":
            posture_need = bool(state.get("posture_need_more_info", True))
            fitness_need = bool(state.get("fitness_need_more_info", True))

            if posture_need or fitness_need:
                return ["posture_expert", "fitness_expert"]
            elif posture_need:
                return "posture_expert"
            elif fitness_need:
                return "fitness_expert"
            else:
                if next_expert == "posture_expert":
                    return "posture_expert"
                elif next_expert == "fitness_expert":
                    return "fitness_expert"
                return ["posture_expert", "fitness_expert"]

        if next_expert == "fitness_expert":
            return "fitness_expert"
        return "posture_expert"

    def route_after_coach(state):
        """教练验证后：有分歧→回医生辩论；无分歧→整合"""
        debate_round = state.get("debate_round", 0)

        if debate_round == 0:
            # 第一轮验证结束：有分歧 → 进入辩论（回 doctor_validator，它内部会轮次+1）
            return "doctor_validator" if check_disagreement(state) else "merge_validation"
        else:
            # 辩论轮结束：继续辩论 → 回 doctor_validator；达成共识/超轮次 → 整合
            return "doctor_validator" if check_debate_should_continue(state) == "continue" else "merge_validation"

    # ===== 添加边 =====
    graph.add_edge(START, "triage")

    # triage → 路由
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "posture_expert": "posture_expert",
            "fitness_expert": "fitness_expert",
            "plan_generator": "plan_generator",
        }
    )

    # 专家节点 → 汇聚节点（both并行时，两个专家都跑完才进入汇聚）
    graph.add_edge("posture_expert", "merge_expert_responses")
    graph.add_edge("fitness_expert", "merge_expert_responses")
    graph.add_edge("merge_expert_responses", END)

    # 方案生成 → 医生验证（自审已在 plan_generator 内部完成）
    graph.add_edge("plan_generator", "doctor_validator")

    # 医生验证 → 教练验证
    graph.add_edge("doctor_validator", "coach_validator")

    # 教练验证 → 路由（辩论循环 or 整合）
    graph.add_conditional_edges(
        "coach_validator",
        route_after_coach,
        {
            "doctor_validator": "doctor_validator",
            "merge_validation": "merge_validation",
        }
    )

    # 验证整合 → END
    graph.add_edge("merge_validation", END)

    app = graph.compile()
    return app


# ==================== 测试 ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("主图测试 - 交互式对话（深度合并版）")
    logger.info("=" * 60)
    logger.info("输入 'quit' 退出\n")

    app = build_main_graph()

    state = {
        "messages": [],
        "conversation_round": 0,
        "user_profile": {}
    }

    while True:
        user_input = input("你：")

        if user_input.lower() == "quit":
            logger.info("\n再见！")
            break

        if not user_input.strip():
            continue

        state["user_input"] = user_input
        state["messages"].append({
            "role": "user",
            "content": user_input
        })

        # 节点已是异步，同步环境里用 asyncio.run 包一层 ainvoke
        result = asyncio.run(app.ainvoke(state))
        state = result

        logger.info(f"\n助手：{state.get('current_response', '抱歉，我没有理解你的意思。')}\n")
        logger.info(f"（意图：{state.get('intent', 'unknown')}，阶段：{state.get('current_stage', 'unknown')}）\n")
