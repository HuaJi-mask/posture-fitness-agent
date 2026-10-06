# scripts/test_chat_history.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.chat_history_utils import build_llm_messages

# 模拟多轮对话历史（10轮，超过6轮窗口，应该触发摘要）
test_messages = [
    {"role": "user", "content": "我脖子疼，有时候还头晕"},
    {"role": "assistant", "content": "根据你说的脖子疼和头晕，我来分析一下...你是做什么工作的？"},
    {"role": "user", "content": "我是程序员，每天坐10个小时"},
    {"role": "assistant", "content": "长期久坐会导致颈椎前屈...有没有试过什么缓解方法？"},
    {"role": "user", "content": "没有，就是有时候自己揉揉"},
    {"role": "assistant", "content": "按摩只能暂时缓解...头晕是什么情况下出现的？"},
    {"role": "user", "content": "转头的时候有时候会晕"},
    {"role": "assistant", "content": "转头时头晕可能和椎动脉受压有关..."},
    {"role": "user", "content": "那这个严重吗？"},
    {"role": "assistant", "content": "从你描述的情况来看，大概率是肌肉劳损..."},
]

print("=" * 60)
print("对话历史管理测试")
print("=" * 60)

# 测试1：对话不长（4轮），应该全部保留
print("\n--- 测试1：对话不长（4轮），应该全部保留 ---")
short_messages = test_messages[:4]
print(f"原始消息数：{len(short_messages)}")

result_messages, summary = build_llm_messages(
    short_messages, 
    chat_summary="",
    max_recent_turns=4
)
print(f"裁剪后消息数：{len(result_messages)}")
print(f"摘要：{summary if summary else '(空，没触发压缩)'}")
print(f"是否触发压缩：{'是' if summary else '否'}")

# 测试2：对话很长（10轮），应该触发摘要+最近对话
print("\n--- 测试2：对话很长（10轮），应该触发摘要+最近对话 ---")
print(f"原始消息数：{len(test_messages)}")

result_messages2, summary2 = build_llm_messages(
    test_messages,
    chat_summary="",
    max_recent_turns=4
)
print(f"裁剪后消息数：{len(result_messages2)}")
print(f"摘要长度：{len(summary2) if summary2 else 0} 字")
print(f"摘要内容：{summary2[:100] if summary2 else '(空)'}...")
print(f"\n裁剪后的消息列表：")
for i, msg in enumerate(result_messages2):
    role = getattr(msg, 'type', 'unknown')
    content = msg.content[:50] if hasattr(msg, 'content') else str(msg)[:50]
    print(f"  {i+1}. [{role}] {content}...")

# 测试3：增量更新摘要（已有摘要，再加新对话）
print("\n--- 测试3：增量更新摘要（已有摘要，再加新对话）---")
print(f"已有摘要长度：{len(summary2) if summary2 else 0} 字")

# 再加4轮新对话
new_messages = test_messages + [
    {"role": "user", "content": "那我应该怎么矫正？"},
    {"role": "assistant", "content": "矫正需要从放松紧张肌肉和强化薄弱肌肉两方面入手..."},
]

result_messages3, summary3 = build_llm_messages(
    new_messages,
    chat_summary=summary2,
    max_recent_turns=4
)
print(f"新摘要长度：{len(summary3) if summary3 else 0} 字")
print(f"摘要是否更新：{'是' if summary3 != summary2 else '否'}")
print(f"新摘要前100字：{summary3[:100] if summary3 else '(空)'}...")
