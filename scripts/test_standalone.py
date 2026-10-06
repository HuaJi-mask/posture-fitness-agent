# scripts/test_standalone.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.query_rewriter import make_query_standalone

# 测试用例
test_cases = [
    {
        "chat_history": [
            ("user", "我有扁平足"),
            ("assistant", "扁平足是一种常见的足部问题...")
        ],
        "query": "那它会导致膝盖疼吗？",
        "expected_contains": ["扁平足"]
    },
    {
        "chat_history": [
            ("user", "上交叉综合征有什么症状"),
            ("assistant", "上交叉综合征的主要症状包括圆肩、头前引...")
        ],
        "query": "那这个怎么矫正？",
        "expected_contains": ["上交叉综合征"]
    },
    {
        "chat_history": [],
        "query": "深蹲标准动作怎么做",
        "expected_contains": ["深蹲"]
    },
]

print("=" * 60)
print("指代消解测试")
print("=" * 60)

for i, case in enumerate(test_cases, 1):
    print(f"\n--- 测试 {i} ---")
    print(f"对话历史: {case['chat_history']}")
    print(f"原始问题: {case['query']}")
    
    result = make_query_standalone(case["query"], case["chat_history"])
    print(f"改写后: {result}")
    
    # 检查是否包含期望的关键词
    for keyword in case["expected_contains"]:
        if keyword in result:
            print(f"  ✓ 包含 '{keyword}'")
        else:
            print(f"  ✗ 缺少 '{keyword}'")
