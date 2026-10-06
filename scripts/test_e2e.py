"""
端到端全流程测试
验证：多轮对话、指代消解、对话历史管理、意图识别、专家节点
"""
import sys
import os
import asyncio

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.main_graph import build_main_graph


# 测试用例：多轮对话
TEST_CASES = [
    # 第1轮：体态问题
    {
        "input": "我有扁平足，走路走久了脚底疼",
        "expect_intent": "posture",
        "expect_keywords": ["扁平足", "足弓", "下肢"],
        "description": "体态问题首轮提问"
    },
    # 第2轮：指代消解测试（"它"指扁平足）
    {
        "input": "那它会导致膝盖疼吗？",
        "expect_intent": "posture",
        "expect_keywords": ["膝盖", "力线", "代偿"],
        "description": "指代消解测试（'它'指扁平足）"
    },
    # 第3轮：继续追问
    {
        "input": "那我平时走路需要注意什么？",
        "expect_intent": "posture",
        "expect_keywords": ["走路", "鞋子", "姿势"],
        "description": "继续追问体态建议"
    },
    # 第4轮：切换到健身问题
    {
        "input": "我想练一下臀部，有什么动作推荐？",
        "expect_intent": "fitness",
        "expect_keywords": ["臀", "动作", "训练"],
        "description": "切换到健身问题"
    },
    # 第5轮：指代消解测试（"它"指臀部训练）
    {
        "input": "那它一周练几次比较好？",
        "expect_intent": "fitness",
        "expect_keywords": ["一周", "次", "频率"],
        "description": "指代消解测试（'它'指臀部训练）"
    },
]


async def run_test():
    print("=" * 70)
    print("端到端全流程测试")
    print("=" * 70)
    print(f"共 {len(TEST_CASES)} 轮对话测试\n")

    app = build_main_graph()

    state = {
        "messages": [],
        "conversation_round": 0,
        "user_profile": {}
    }

    passed = 0
    failed = 0

    for i, case in enumerate(TEST_CASES, 1):
        print(f"--- 第 {i}/{len(TEST_CASES)} 轮 ---")
        print(f"用户：{case['input']}")
        print(f"说明：{case['description']}")

        state["user_input"] = case["input"]
        state["messages"].append({
            "role": "user",
            "content": case["input"]
        })

        try:
            result = await app.ainvoke(state)
            state = result

            response = state.get("current_response", "")
            intent = state.get("intent", "unknown")

            print(f"助手：{response[:150]}...")
            print(f"意图：{intent}")

            # 检查意图
            intent_ok = (intent == case["expect_intent"])

            # 检查关键词
            keywords_found = sum(1 for kw in case["expect_keywords"] if kw in response)
            keywords_ok = keywords_found >= 1  # 至少命中1个关键词

            if intent_ok and keywords_ok:
                print("✅ 通过")
                passed += 1
            else:
                print(f"❌ 失败")
                print(f"   意图预期: {case['expect_intent']}, 实际: {intent}")
                print(f"   关键词预期: {case['expect_keywords']}, 命中: {keywords_found}")
                failed += 1

        except Exception as e:
            print(f"❌ 异常: {e}")
            failed += 1

        print()

    # 结果统计
    print("=" * 70)
    print("测试结果统计")
    print("=" * 70)
    print(f"总用例数: {len(TEST_CASES)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"通过率: {passed / len(TEST_CASES) * 100:.1f}%")
    print()

    # 对话历史检查
    print("对话历史管理检查:")
    print(f"  消息总数: {len(state.get('messages', []))}")
    print(f"  对话轮次: {state.get('conversation_round', 0)}")
    print(f"  对话摘要: {state.get('chat_summary', '(无)')[:100]}...")
    print()

    if failed == 0:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️ 有 {failed} 个测试失败")


if __name__ == "__main__":
    asyncio.run(run_test())
