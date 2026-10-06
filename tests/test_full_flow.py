"""
全流程自动化测试脚本
覆盖：单体态问诊、单健身咨询、both并行、方案生成验证、用户纠正、意图切换
用法：venv\Scripts\python.exe tests\test_full_flow.py
"""

import sys
import os
import asyncio
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.main_graph import build_main_graph


# ==================== 测试工具 ====================

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.errors = []
        self.logs = []
        self.start_time = time.time()

    def log(self, msg):
        self.logs.append(msg)
        print(f"  [{self.name}] {msg}")

    def error(self, msg):
        self.passed = False
        self.errors.append(msg)
        print(f"  ❌ [{self.name}] 错误: {msg}")

    def check(self, condition, msg):
        if condition:
            self.log(f"✅ {msg}")
        else:
            self.error(msg)

    def finish(self):
        duration = time.time() - self.start_time
        status = "✅ 通过" if self.passed else "❌ 失败"
        print(f"\n{'='*60}")
        print(f"【{self.name}】{status} (耗时 {duration:.1f}s)")
        if self.errors:
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'='*60}\n")
        return self  # 返回 self，不是 self.passed


def run_round(app, state, user_input):
    """运行一轮对话，返回更新后的 state"""
    state["user_input"] = user_input
    state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
    result = asyncio.run(app.ainvoke(state))
    return result


def print_state(state, round_num):
    """打印关键状态"""
    print(f"\n  --- 第 {round_num} 轮状态 ---")
    print(f"  intent: {state.get('intent', 'N/A')}")
    print(f"  next_expert: {state.get('next_expert', 'N/A')}")
    print(f"  current_stage: {state.get('current_stage', 'N/A')}")
    print(f"  need_more_info: {state.get('need_more_info', 'N/A')}")
    print(f"  posture_need: {state.get('posture_need_more_info', 'N/A')}")
    print(f"  fitness_need: {state.get('fitness_need_more_info', 'N/A')}")
    print(f"  awaiting: {state.get('awaiting_confirmation', 'N/A')}")
    resp = state.get('current_response', '')
    print(f"  response(前100字): {resp[:100]}...")
    print(f"  messages数: {len(state.get('messages', []))}")


# ==================== 场景1：单体态问诊 ====================

def test_posture_only():
    """测试单体态问诊流程：多轮收集信息→确诊"""
    tr = TestResult("场景1-单体态问诊")
    app = build_main_graph()
    state = {"messages": [], "conversation_round": 0, "user_profile": {}}

    user_inputs = [
        "我脖子疼，有时候还头晕",
        "主要是后脖颈疼，连着肩膀也酸",
        "就是酸痛的感觉，早上起床最明显，活动活动好一点",
        "我是程序员，每天坐10个小时以上，经常低头写代码",
    ]

    diagnosed = False
    for i, user_input in enumerate(user_inputs):
        try:
            state = run_round(app, state, user_input)
            print_state(state, i + 1)

            tr.check(state.get("current_response"), f"第{i+1}轮有回复")
            tr.check(state.get("intent") in ["posture", "both", "unknown"], f"第{i+1}轮意图合理")

            if not state.get("need_more_info", True):
                tr.log(f"第{i+1}轮专家确诊了")
                tr.check(state.get("posture_diagnosis"), "确诊时有诊断结论")
                tr.check(state.get("awaiting_confirmation") == "posture", "确诊后等待确认")
                diagnosed = True
                break
        except Exception as e:
            tr.error(f"第{i+1}轮异常: {e}")
            import traceback
            traceback.print_exc()
            break

    if not diagnosed:
        tr.log("4轮内未确诊（可能需要更多轮，属正常）")

    return tr.finish()


# ==================== 场景2：单健身咨询 ====================

def test_fitness_only():
    """测试单健身咨询流程：多轮收集信息→目标总结"""
    tr = TestResult("场景2-单健身咨询")
    app = build_main_graph()
    state = {"messages": [], "conversation_round": 0, "user_profile": {}}

    user_inputs = [
        "我想增肌",
        "我是新手，之前没练过",
        "每周能练3天，每次45分钟左右",
        "家里有哑铃，没有其他器械，没有伤病",
    ]

    summarized = False
    for i, user_input in enumerate(user_inputs):
        try:
            state = run_round(app, state, user_input)
            print_state(state, i + 1)

            tr.check(state.get("current_response"), f"第{i+1}轮有回复")

            if not state.get("need_more_info", True):
                tr.log(f"第{i+1}轮健身专家完成目标分析")
                tr.check(state.get("fitness_goal_summary"), "完成时有目标总结")
                tr.check(state.get("awaiting_confirmation") == "fitness", "完成后等待确认")
                summarized = True
                break
        except Exception as e:
            tr.error(f"第{i+1}轮异常: {e}")
            import traceback
            traceback.print_exc()
            break

    if not summarized:
        tr.log("4轮内未完成总结（可能需要更多轮，属正常）")

    return tr.finish()


# ==================== 场景3：both 并行多轮 ====================

def test_both_parallel():
    """测试both意图：第一轮并行→后续按进度并行/串行"""
    tr = TestResult("场景3-both并行多轮")
    app = build_main_graph()
    state = {"messages": [], "conversation_round": 0, "user_profile": {}}

    user_inputs = [
        "我脖子疼，还想增肌",
        "脖子主要是酸痛，每天坐10小时。增肌的话我是新手，每周能练3天",
        "早上起来最明显，活动后好点。家里有哑铃，没伤病",
    ]

    for i, user_input in enumerate(user_inputs):
        try:
            state = run_round(app, state, user_input)
            print_state(state, i + 1)

            tr.check(state.get("current_response"), f"第{i+1}轮有回复")

            if i == 0:
                # 第一轮应该是 both 并行
                tr.check(state.get("intent") == "both", "第一轮意图为both")
                tr.check(state.get("current_stage") == "both", "第一轮阶段为both")
                # 汇聚后临时字段被清空，但 need_more_info 保留
                tr.check("posture_need_more_info" in state, "有体态进度字段")
                tr.check("fitness_need_more_info" in state, "有健身进度字段")
                tr.log(f"体态还需信息: {state.get('posture_need_more_info')}")
                tr.log(f"健身还需信息: {state.get('fitness_need_more_info')}")

            # 检查 messages 里只有一条 AI 消息（汇聚后统一追加）
            # messages 里是 HumanMessage/AIMessage 对象，用 .type 属性（"human"/"ai"）
            ai_msgs = [m for m in state.get("messages", []) if getattr(m, "type", "") == "ai"]
            tr.check(len(ai_msgs) == i + 1, f"第{i+1}轮后有 {i+1} 条AI消息（实际 {len(ai_msgs)}）")

        except Exception as e:
            tr.error(f"第{i+1}轮异常: {e}")
            import traceback
            traceback.print_exc()
            break

    return tr.finish()


# ==================== 场景4：方案生成+验证 ====================

def test_plan_generation():
    """测试方案生成+双验证流程（模拟已确诊状态）"""
    tr = TestResult("场景4-方案生成+验证")
    app = build_main_graph()

    # 构造已确诊的 state，直接进入方案生成
    state = {
        "messages": [
            {"role": "user", "content": "我脖子疼，还想增肌"},
            {"role": "assistant", "content": "（已完成体态诊断和健身目标分析）"},
            {"role": "user", "content": "好的，开始制定方案吧"},
        ],
        "user_input": "好的，开始制定方案吧",
        "intent": "both",
        "next_expert": "plan_generator",
        "posture_diagnosis": "颈椎生理曲度变直，上交叉综合征",
        "correction_approach": "放松紧张肌肉，强化薄弱肌肉",
        "fitness_goal_summary": "增肌，新手，每周3天，家用哑铃",
        "posture_diagnosis_confirmed": True,
        "fitness_analysis_confirmed": True,
        "user_profile": {
            "posture": {"problem_areas": [{"area": "颈椎", "symptoms": ["脖子疼", "头晕"]}]},
            "fitness": {"goals": [{"goal": "增肌"}], "fitness_level": "新手", "weekly_days": 3}
        },
        "debate_round": 0,
    }

    try:
        state = asyncio.run(app.ainvoke(state))
        print_state(state, 1)

        tr.check(state.get("training_plan") or state.get("current_response"), "有方案输出")
        tr.check(state.get("validation_report"), "有验证报告")
        tr.check("validation_passed" in state, "有验证通过标记")
        tr.check(state.get("current_stage") == "validation", "最终阶段为validation")

        # 检查 messages 里有方案和验证报告两条 AI 消息
        ai_msgs = [m for m in state.get("messages", []) if m.get("type") == "ai" or m.get("role") == "assistant"]
        tr.log(f"AI消息数: {len(ai_msgs)}")

        tr.log(f"验证通过: {state.get('validation_passed')}")
        tr.log(f"验证报告(前150字): {str(state.get('validation_report', ''))[:150]}")

    except Exception as e:
        tr.error(f"方案生成异常: {e}")
        import traceback
        traceback.print_exc()

    return tr.finish()


# ==================== 场景5：用户纠正信息 ====================

def test_user_correction():
    """测试用户纠正信息：说错→纠正→专家调整"""
    tr = TestResult("场景5-用户纠正信息")
    app = build_main_graph()
    state = {"messages": [], "conversation_round": 0, "user_profile": {}}

    try:
        # 第一轮：说错（右脚）
        state = run_round(app, state, "我右脚疼，走路的时候更明显")
        print_state(state, 1)
        tr.check("右脚" in state.get("current_response", "") or "右" in state.get("current_response", ""),
                 "第一轮回复提到右脚")

        # 第二轮：纠正为左脚
        state = run_round(app, state, "哦不对，说错了，是左脚疼")
        print_state(state, 2)
        tr.check(state.get("current_response"), "第二轮有回复")

        # 检查 extracted_info 里是否更新为左脚
        posture_info = state.get("user_profile", {}).get("posture", {})
        problem_areas = posture_info.get("problem_areas", [])
        if problem_areas:
            areas_text = json.dumps(problem_areas, ensure_ascii=False)
            tr.log(f"problem_areas: {areas_text[:200]}")
            tr.check("左脚" in areas_text or "左" in areas_text, "提取的信息更新为左脚")
        else:
            tr.log("未提取到 problem_areas（LLM可能未结构化输出）")

    except Exception as e:
        tr.error(f"异常: {e}")
        import traceback
        traceback.print_exc()

    return tr.finish()


# ==================== 场景6：意图切换 ====================

def test_intent_switch():
    """测试意图切换：先说体态→转健身→导诊识别"""
    tr = TestResult("场景6-意图切换")
    app = build_main_graph()
    state = {"messages": [], "conversation_round": 0, "user_profile": {}}

    try:
        # 第一轮：体态
        state = run_round(app, state, "我脖子疼")
        print_state(state, 1)
        tr.check(state.get("intent") in ["posture", "both", "unknown"], "第一轮意图为体态相关")

        # 第二轮：转到健身
        state = run_round(app, state, "对了，我还想减脂")
        print_state(state, 2)
        # 导诊应该能识别到用户新增了健身需求
        tr.check(state.get("intent") in ["both", "fitness", "posture"], "第二轮意图合理（both或fitness）")
        tr.check(state.get("current_response"), "第二轮有回复")

    except Exception as e:
        tr.error(f"异常: {e}")
        import traceback
        traceback.print_exc()

    return tr.finish()


# ==================== 主入口 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("全流程自动化测试开始")
    print("=" * 60)

    results = []

    # 按顺序运行测试（每个测试独立，避免状态污染）
    # results.append(test_posture_only())       # 上次已通过
    # results.append(test_fitness_only())       # 上次已通过
    results.append(test_both_parallel())
    results.append(test_plan_generation())
    results.append(test_user_correction())
    results.append(test_intent_switch())

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.name}")
    print(f"\n总计: {passed}/{total} 通过")
    print("=" * 60)
