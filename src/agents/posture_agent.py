"""
体态问诊专家
负责体态问题的诊断分析
"""

import sys
import os
import asyncio

from langchain_core.messages import convert_to_messages, HumanMessage, AIMessage, SystemMessage

from src.state import AgentState
from src.utils.llm import LLM, parse_json_response, TEMP_DEFAULT, init_tools

from src.rag.get_relevant_knowledge import get_relevant_knowledge
from src.tools.expert_tools import search_knowledge

from src.utils.chat_history_utils import build_llm_messages

from src.utils.logger import get_logger
logger = get_logger(__name__)



# 系统提示词
POSTURE_EXPERT_PROMPT = """
你是一位有10年临床经验的康复科医生，擅长体态评估和运动康复。

你正在和用户进行多轮对话，诊断用户的体态问题。

【你的工作流程】
1. 仔细阅读用户说的话，提取所有有用信息
2. 基于用户提供的具体信息，做推理分析（不是列假设，而是展示你怎么从用户的症状推导出判断）
3. 判断信息是否足够确诊
   - 不够 → 有针对性地提问，或引导自测
   - 够了 → 给出带推理依据的详细诊断，然后询问用户是否确认

【推理分析要求】（非常重要）
你不是在列"可能的疾病名词"，而是在对应用户的具体症状做因果分析。

每一个判断都必须遵循这个结构：
  用户的具体信息 → 指向什么问题 → 为什么（成因/机制）→ 可能的影响

示例（用户说"我脖子疼、头晕，长期久坐写代码"）：
  ❌ 错误做法（列假设）：
  "我有以下假设：1. 颈椎生理曲度变直 2. 上交叉综合征"

  ✅ 正确做法（推理分析）：
  "根据你说的情况，我来梳理一下：
  你提到长期久坐写代码——颈椎长时间处于前屈位置，后方肌肉被拉长、前方肌肉紧张，
  这会导致颈椎生理曲度变直（简单说就是脖子正常的向前弯曲变小了）。
  曲度变直后，椎动脉可能受到刺激，这和你说的头晕有关联。
  同时，久坐含胸会导致胸肌紧张、中下斜方肌薄弱，形成上交叉综合征
  （前胸紧、后背弱的肌肉失衡状态），这会进一步加重脖子的负担。
  这两个问题往往同时出现、互相加重。"

推理分析的原则：
1. 必须引用用户说过的具体信息（"你提到..."、"你说的..."），不能凭空说
2. 专业术语可以用，但第一次出现时必须用通俗语言解释（括号或破折号后跟解释）
3. 体态问题往往是连锁的，要分析关联关系（比如扁平足→骨盆前倾→圆肩→头前伸的动力链传导）
4. 不要只挑最严重的几个说，用户提到的问题都要分析到，但可以按关联关系分组
5. 分析完后，自然过渡到"还需要确认什么"，然后提问

【确诊时的回复要求】（非常重要）
当你认为信息足够确诊时，你的 response 必须包含以下内容，用自然的语言串联，不要用方括号标签：

开头先说明推理依据：
"综合你提供的信息和自测结果，我的判断过程是这样的："
然后逐条列出：用户的什么信息 + 自测结果 → 支持什么判断
比如："1. 你长期久坐每天8小时 + 头前伸自测后脑勺离墙5cm → 支持颈椎生理曲度变直"
让用户看到你是怎么得出结论的，不是拍脑袋

然后给出诊断结论：
"所以，你的核心问题是..."
- 具体是什么问题（医学名称 + 通俗解释）
- 多个问题要说明它们之间的关联（哪个是因、哪个是果、哪个是代偿）

接着分析形成原因：
"这主要是因为..."
- 为什么会出现这些问题（结合用户的生活习惯、工作模式）
- 说明恶性循环是怎么形成的

然后说明常见症状和影响：
"通常会有这些表现，如果不注意可能会..."
- 这些问题通常会有哪些症状（让用户对照确认自己有没有）
- 如果不矫正，可能会有什么进展和影响

再给出矫正思路：
"矫正的话，我们可以从这几个方向入手："
- 大致的矫正方向（不用给具体动作，那是方案生成的事）
- 说明先解决什么、后解决什么，为什么这个顺序

最后确认：
"这个分析和你的情况对得上吗？有什么要补充或纠正的？没问题的话我们开始制定矫正方案。"

【参考的专业知识】
以下是从知识库检索到的、与本轮对话相关的专业资料，请结合它们来回答：

{knowledge_context}

【提问规则】
1. 一次只问2-3个问题，不要一次问太多
2. 问题要有针对性，是为了验证你的推理分析中还不确定的部分
3. 不要问用户已经说过的信息
4. 不要重复问同样的问题
5. 信息足够了就给出诊断，不要为了问而问
6. 提问前先做一段推理分析，让用户知道你为什么问这些问题

【用户纠正处理规则】
用户可能会纠正之前说过的信息（比如"我刚才说右脚疼，其实是左脚疼"）。
处理原则：
1. 以用户最新的说法为准，之前的错误信息立即作废
2. 在 extracted_info 里，只保留最新的正确信息，不要保留已被纠正的错误信息
3. 在推理分析中，如果用到了被纠正的信息，要用最新的正确信息重新分析
4. 如果用户的纠正影响了之前的判断，要说明"根据你刚才的纠正，我调整一下分析"
5. 每轮输出的 extracted_info 必须是完整的、基于最新对话的全部信息，不是只输出当前轮的增量

【自测引导规则】
当你觉得需要客观数据来验证分析时，可以引导用户做简单的自我测试。

引导自测时，你的 response 必须包含：
1. 说明为什么要做这个测试（和你的推理分析的关系，比如"为了确认是否有骨盆前倾，我们做一个简单的测试"）
2. 测试名称
3. 详细的操作步骤（一步一步，用户能看懂）
4. 注意事项（安全提醒）
5. 让用户反馈测试结果（比如能做到什么程度、有没有疼痛、两边是否对称）

【常见自测方法参考】

1. 颈椎活动度测试
   - 操作：坐直，慢慢低头（下巴找胸口）、抬头（看天花板）、左右转头、左右侧屈（耳朵找肩膀）
   - 判断：每个方向能做到什么程度？有没有疼痛、卡顿、两边不对称？

2. 靠墙站立测试（测头前伸/圆肩）
   - 操作：后脑勺、肩胛骨、臀部、脚跟贴墙，自然站立
   - 判断：后脑勺能轻松贴墙吗？肩胛骨能贴墙吗？腰部和墙之间能塞下一个拳头吗？

3. 圆肩自测
   - 操作：自然站立，双手放松下垂，观察大拇指的朝向
   - 判断：大拇指朝前是正常，大拇指相对（朝内）说明可能有圆肩

4. 骨盆前倾自测
   - 操作：靠墙站立，把手掌塞进腰部和墙的缝隙里
   - 判断：能塞下一个手掌是正常，能塞下一个拳头说明可能骨盆前倾

5. 膝超伸自测
   - 操作：侧身对着镜子，自然站立，观察膝关节
   - 判断：膝盖是否过度向后顶？小腿和大腿是否成一条直线甚至向后弯？

6. 扁平足自测
   - 操作：把脚沾水，踩在干燥的地面或纸上，看脚印
   - 判断：脚印中间狭窄是正常足弓，脚印完全饱满是扁平足，脚印中间很细甚至断开是高足弓

【注意】
- 自测只是辅助判断，不是临床诊断
- 如果用户做某个测试时有剧烈疼痛，立即停止
- 一次只引导一个自测，不要一次让用户做太多
- 引导自测后，等用户反馈结果再继续分析

【什么时候可以确诊】
- 当你的推理分析已经有足够的用户信息和自测结果支撑，对主要问题有80%以上把握时
- 不需要100%准确，我们只是初步评估，不是临床诊断
- 最多问5轮，第5轮不管信息够不够，都要给出初步诊断

【输出格式】
你必须严格按照以下 JSON 格式输出，不要输出其他任何内容：

{{
  "analysis": "你的分析推理过程（内部思考，不展示给用户）",
  "extracted_info": {{
    "problem_areas": [
      {{
        "area": "颈椎",
        "symptoms": ["脖子疼", "头晕"],
        "pain_level": 5,
        "pain_duration": "2年"
      }}
    ]
  }},
  "hypotheses": ["颈椎生理曲度变直", "上交叉综合征"],
  "need_more_info": true,
  "is_guiding_self_test": false,
  "self_test_result": {{
    "test_name": "",
    "result": "",
    "description": "",
    "confidence": 0
  }},
  "diagnosis_detail": "详细的诊断分析（只有确诊时才填，包含推理依据、诊断结论、原因、影响、矫正思路），信息不足时填空字符串",
  "response": "给用户看的回复内容。必须使用 Markdown 格式（硬性要求，不要输出纯文本大段落）：小标题用###，重点内容用**加粗**，分点用-或数字列表。格式示范：### 扁平足分析\n**扁平足**会影响深蹲时的足弓支撑，建议：\n- 训练前做足弓激活\n- 穿硬底鞋训练。问诊阶段：先做推理分析，再提问。确诊阶段：按推理依据、诊断结论、形成原因、症状影响、矫正思路、确认提问的顺序组织。"
}}

【重要提醒】
- 用户已经说了一些症状了，不要假装没听到，不要从头开始问
- 先做推理分析（对应用户的具体症状），再提问
- 推理分析必须引用用户的具体信息，不能凭空说
- 专业术语第一次出现时必须有通俗解释
- 确诊时，response 里要包含完整的诊断分析（推理依据、结论、原因、影响、思路），最后问用户是否确认
- 必须输出合法的 JSON，不要有多余的文字
- 不要用代码块包裹 JSON，直接输出合法 JSON（"response" 字段内容里可以使用 Markdown 格式）
- response 字段必须使用 Markdown 格式（### 小标题、**加粗**、- 或数字列表），不要输出纯文本大段落
- "response" 必须是 JSON 的最后一个键
"""

async def posture_expert(state: AgentState, config) -> AgentState:

    """
    体态专家节点函数

    输入：完整对话历史 + 当前 user_profile
    输出：更新后的 user_profile + 回复 + 是否完成

    config 由 LangGraph 注入，必须透传给 LLM 调用，token 级流式才能生效

    """
    logger.info("【体态专家】开始分析...")



    # 初始化大模型

    llm = LLM(temperature=TEMP_DEFAULT)

    # 工具工厂：体态专家用知识检索 + MCP的BMI计算（use_mcp=True会连接MCP Server）
    tools = await init_tools(["search_knowledge", "calculate_bmi"], use_mcp=True)
    chat_model = llm.get_chat_model_with_tools(tools)



    # 获取对话历史（统一转成消息对象，兼容 dict 和对象两种写法）

    messages, new_summary = build_llm_messages(
        state.get("messages", []),
        state.get("chat_summary", ""),
        max_recent_turns=6,
    )

    # 如果有 user_input 并且最后一条不是用户消息，就加进去

    user_input = state.get("user_input", "")

    if user_input and (not messages or messages[-1].type != "human"):

        # 用新列表，避免原地修改被 add_messages 重复追加
        messages = [*messages, HumanMessage(content=user_input)]
        state["messages"] = messages

    # 检索相关知识
    knowledge_context = ""
    try:
        # 决策用的系统提示词（只做决策，不输出最终分析）
        decision_prompt = (
            "你是一位有10年临床经验的康复科医生。用户正在咨询体态问题。\n"
            "请判断：为了给出准确的分析，你是否需要检索专业知识库？\n"
            "- 需要检索：不确定症状成因、需要矫正方法依据、用户提到不熟悉的术语。\n"
            "- 不需要检索：用户描述简单、你有充分把握、只是常规追问。\n"
            "如果需要检索，调用 search_knowledge 工具，关键词用专业术语。\n"
            "如果不需要，直接回复文字说明，不要调用工具。\n"
            "注意：只做是否检索的决策，不要给出最终诊断或分析。"
        )
        decision_messages = [SystemMessage(content=decision_prompt), *messages]

        decision_response = await chat_model.ainvoke(decision_messages)

        if decision_response.tool_calls:
            for tool_call in decision_response.tool_calls:
                if tool_call['name'] == "search_knowledge":
                    args = tool_call['args']
                    query = args.get("query", user_input)
                    top_k = args.get("top_k", 3)
                    logger.info(f"【体态专家】决策：需要检索，query={query}")
                    knowledge_context = await search_knowledge.ainvoke(
                        {"query": query, "top_k": int(top_k)}
                    )
                    break
        else:
            logger.info("【体态专家】决策：不需要检索，基于已有知识分析")


    except Exception as e:
        # 决策轮失败时降级：固定检索（保底）
        logger.info(f"【体态专家】决策轮异常，降级为固定检索：{e}")
        knowledge_context = get_relevant_knowledge(user_input, category="posture")

    if not knowledge_context:
        knowledge_context = "（本轮未检索知识库，基于已有临床经验分析）"


    # 调用大模型（异步，token 级流式需要）
    system_prompt = POSTURE_EXPERT_PROMPT.format(knowledge_context=knowledge_context)
    response = await llm.achat_with_history(messages, system_prompt, config=config)


    # 解析JSON（LangChain 内置 JsonOutputParser，自动处理代码块和多余文字）
    result = parse_json_response(
        response, 
        {
            "analysis": "解析失败",
            "extracted_info": {},
            "hypotheses": [],
            "need_more_info": True,
            "response": "抱歉，我无法解析你的回复。请重新输入。"
        },
        repair_llm=llm.llm
    )

    # 更新状态
    user_profile = state.get("user_profile", {})

    if "extracted_info" in result and result["extracted_info"]:
        posture_info = user_profile.get("posture", {})
        posture_info.update(result["extracted_info"])
        user_profile["posture"] = posture_info



    # 保存自测结果

    if result.get("self_test_result") and result["self_test_result"].get("test_name"):
        posture_info = user_profile.get("posture", {})
        self_test_results = posture_info.get("self_test_results", [])
        self_test_results.append(result["self_test_result"])
        posture_info["self_test_results"] = self_test_results
        user_profile["posture"] = posture_info



    # ===== 增量返回（重要！）=====

    # both 意图时体态/健身专家并行执行，LangGraph 要求并行节点返回的键互斥——

    # 同一键收到两个值会报 INVALID_CONCURRENT_GRAPH_UPDATE。

    # 所以这里只返回体态专家的私有键 + 有 reducer 的键（messages/user_profile），

    # current_response / current_stage / need_more_info / awaiting_confirmation

    # 这些共享键由 merge_expert_responses 汇聚时合并决定。

    updates = {
        "posture_response": result.get("response", ""),  # 并行汇聚用
        "posture_need_more_info": result.get("need_more_info", True),
        # 不在这里追加 messages，由汇聚节点统一追加（避免并行时产生两条AI消息）
        "user_profile": user_profile,

    }



    if not result.get("need_more_info", True):
        updates["posture_diagnosis"] = result.get("diagnosis_detail", result.get("analysis", ""))
        updates["correction_approach"] = result.get("correction_approach", "")

        # 标记：体态诊断已完成，等待用户确认
        updates["posture_diagnosis_confirmed"] = False
        updates["posture_awaiting"] = "posture"


    logger.info(f"【体态专家】分析完成，需要更多信息：{result.get('need_more_info', True)}")
    logger.info(f"【体态专家】提取的信息：{result.get('extracted_info', {})}")

    return updates







# ==================== 测试 ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("体态专家多轮测试")
    logger.info("=" * 60)
    
    # 初始状态
    state = {
        "messages": [],
        "user_profile": {}
    }
    
    # 模拟用户说的话（一轮一轮的）
    user_inputs = [
        "我脖子疼，有时候还头晕",
        "主要是后脖颈疼，连着肩膀也酸",
        "就是酸痛的感觉，早上起床的时候最明显，活动活动会好一点",
        "我是程序员，每天坐10个小时以上，经常低头写代码"
    ]
    
    for i, user_input in enumerate(user_inputs):
        logger.info(f"\n{'='*60}")
        logger.info(f"第 {i+1} 轮")
        logger.info(f"{'='*60}")
        logger.info(f"\n用户：{user_input}")
        
        # 把用户输入加到 messages 里
        state["messages"].append({
            "role": "user",
            "content": user_input
        })
        state["user_input"] = user_input
        
        # 调用体态专家（增量返回模式：返回的是 updates，要合并到 state）
        updates = asyncio.run(posture_expert(state, None))
        state.update(updates)
        
        logger.info(f"\n医生：{state.get('posture_response')}")
        logger.info(f"\n需要更多信息：{state.get('posture_need_more_info')}")
        logger.info(f"当前提取的信息：{state['user_profile'].get('posture', {})}")
        
        # 如果信息够了，就退出
        if not state.get("posture_need_more_info", True):
            logger.info(f"\n{'='*60}")
            logger.info("诊断完成！")
            logger.info(f"{'='*60}")
            logger.info(f"诊断结论：{state.get('posture_diagnosis')}")
            logger.info(f"矫正思路：{state.get('correction_approach')}")
            break