"""
大模型调用工具
封装所有 LLM 相关的调用

规范：
- 温度用语义化常量，不用魔法数字
- ReAct 节点用 get_chat_model_with_tools() 便捷方法
- 工具通过 init_tools() 工厂函数统一组装（Tool Factory Pattern）
"""
import re
import json
import os

from src.utils.config import config
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import Optional

from src.utils.logger import get_logger
logger = get_logger(__name__)


# ==================== 温度常量（语义化，不用魔法数字）====================
TEMP_PRECISE = 0.1   # 精确判断：triage、自审、验证（低随机性，要求稳定）
TEMP_DEFAULT = 0.3   # 默认：大多数生成任务（平衡质量和稳定性）
TEMP_CREATIVE = 0.4  # 创造性：融合、改写、方案生成（需要一定发散）

_json_parser = JsonOutputParser()


class LLM:
    """
    大模型调用工具
    封装所有 LLM 相关的调用
    """
    def __init__(self, model: str = None, temperature: float = TEMP_DEFAULT):
        self.model = model or config.MODEL_NAME
        self.temperature = temperature

        self.llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            temperature=temperature,
            timeout=120,  # 模型卡住时最多等120秒，超时快速失败，避免请求无限挂起
            streaming=True,  # 开启流式，配合 graph.astream(stream_mode="messages") 做 token 级输出
        )

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """
        调用大模型进行单轮对话

        Args:
            prompt: 用户消息
            system_prompt: 系统提示（可选）

        Returns:
            大模型回复
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self.llm.invoke(messages)
        return response.content

    def chat_with_history(self, messages: list, system_prompt: str = None) -> str:
        """
        带历史记录的对话

        Args:
            messages: 对话消息列表（dict 或消息对象均可，
                      用 LangChain 内置的 convert_to_messages 统一转换）
            system_prompt: 系统提示（可选）

        Returns:
            大模型回复
        """
        langchain_messages = convert_to_messages(messages)
        if system_prompt:
            langchain_messages = [SystemMessage(content=system_prompt), *langchain_messages]

        response = self.llm.invoke(langchain_messages)
        return response.content

    def get_chat_model(self) -> ChatOpenAI:
        """
        获取底层 ChatOpenAI 实例。
        用于 bind_tools、create_agent 等需要直接操作 ChatModel 的场景。
        """
        return self.llm

    def get_chat_model_with_tools(self, tools: list[BaseTool]) -> ChatOpenAI:
        """
        获取绑定了工具的 ChatOpenAI（便捷方法，避免每个节点重复写 bind_tools）。

        Args:
            tools: 要绑定的工具列表

        Returns:
            绑定了工具的 ChatOpenAI 实例
        """
        return self.llm.bind_tools(tools)

    # ==================== 异步方法（token 级流式用） ====================
    # 说明：agent 节点必须改成 async def xxx(state, config) 并调用这些异步方法，
    # 且必须把图注入的 config 透传给 ainvoke —— 这样 graph.astream(stream_mode="messages")
    # 才能截获逐 token 的 chunk（已验证：同步 invoke 只能拿到聚合后的完整消息）。

    async def achat(self, prompt: str, system_prompt: str = None, config=None) -> str:
        """
        异步单轮对话（配合 stream_mode="messages" 做 token 级流式）
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = await self.llm.ainvoke(messages, config=config)
        return response.content

    async def achat_with_history(self, messages: list, system_prompt: str = None, config=None) -> str:
        """
        异步带历史对话（配合 stream_mode="messages" 做 token 级流式）
        """
        langchain_messages = convert_to_messages(messages)
        if system_prompt:
            langchain_messages = [SystemMessage(content=system_prompt), *langchain_messages]

        response = await self.llm.ainvoke(langchain_messages, config=config)
        return response.content


# ==================== 工具工厂（Tool Factory Pattern）====================
# 主流做法：用统一的工厂函数组装工具集，不同 Agent 注入不同工具子集

async def init_tools(tool_names: list[str] = None, use_mcp: bool = False) -> list[BaseTool]:
    """
    工具工厂（异步）：根据名称列表组装工具集。
    在 async 节点函数里用 await init_tools(...) 调用。

    Args:
        tool_names: 需要的工具名称列表，None 表示加载所有本地工具
        use_mcp: 是否同时加载 MCP 工具（默认 False，MCP 主要对外暴露）

    Returns:
        工具列表，可直接传给 bind_tools

    用法：
        # 体态专家用的工具（在 async 函数里）
        tools = await init_tools(["search_knowledge"])
        # 健身专家用的工具（含 MCP 的计算工具）
        tools = await init_tools(["search_knowledge", "calculate_bmi"], use_mcp=True)
    """
    from src.tools.expert_tools import search_knowledge

    # 本地工具注册表
    local_tools = {
        "search_knowledge": search_knowledge,
    }

    # 选择需要的本地工具
    if tool_names is None:
        tools = list(local_tools.values())
    else:
        tools = [local_tools[name] for name in tool_names if name in local_tools]

    # 可选：加载 MCP 工具（带缓存，只加载一次）
    if use_mcp:
        try:
            from src.tools.mcp_client import load_mcp_tools
            mcp_tools = await load_mcp_tools()
            # 去重：MCP 工具里也有 search_knowledge，避免重复
            local_names = {t.name for t in tools}
            for t in mcp_tools:
                if t.name not in local_names:
                    tools.append(t)
        except Exception as e:
            logger.info(f"【工具工厂】MCP 工具加载失败，使用本地工具：{e}")

    return tools


def init_tools_sync(tool_names: list[str] = None) -> list[BaseTool]:
    """
    工具工厂（同步版本）：只加载本地工具，不加载 MCP。
    在非 async 环境（如测试、脚本）里用。

    Args:
        tool_names: 需要的工具名称列表，None 表示加载所有本地工具

    Returns:
        工具列表
    """
    from src.tools.expert_tools import search_knowledge

    local_tools = {
        "search_knowledge": search_knowledge,
    }

    if tool_names is None:
        return list(local_tools.values())
    return [local_tools[name] for name in tool_names if name in local_tools]


# ==================== JSON 解析工具 ====================

# 全局复用同一个解析器（JsonOutputParser 无状态）
_json_parser = JsonOutputParser()


def parse_json_response(response: str, fallback: dict, repair_llm: Optional[LLM] = None) -> dict:
    """
    解析大模型输出的 JSON（四层防护）
    
    第一层：基础清洗 + 正则提取
    第二层：智能修复（调用 LLM 修复格式错误）
    第三层：降级兜底（把纯文本当 response）
    
    Args:
        response: 大模型原始输出
        fallback: 解析失败时的默认结果
        repair_llm: 用于修复的 LLM 实例（可选，传入则启用智能修复）
    
    Returns:
        解析后的 dict
    """

    if not response or not response.strip():
        logger.info("【JSON解析】响应为空，直接降级")
        return fallback


    # 第一层：基础清洗 + 正则提取
    try:
        # 去掉markdown代码块
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*$', '', cleaned)

        # 清理双大括号
        cleaned = cleaned.replace("{{", "{").replace("}}", "}")

        # 正则提取
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("未找到 JSON 字符串")

        # 尝试解析
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result
        else:
            raise ValueError("JSON 不是对象类型")

    except Exception as e:
        logger.info(f"【第一层解析失败】{e}")


    # 第二层：智能修复（调用 LLM 修复格式错误）
    if repair_llm is not None:
        try:
            repair_prompt = f"""
你是一个 JSON 格式修复专家。下面是大模型输出的文本，里面包含一段非标准 JSON。
请把它修复成**标准的、可直接 json.loads 的 JSON 对象**。

原始输出：
{response[:2000]}  # 截断，避免 token 超限

要求：
1. 只输出修复后的 JSON，不要有任何解释文字
2. 保持原有的字段和内容不变
3. 修复常见问题：尾逗号、引号不匹配、单引号转双引号、多余文字等
"""
            repaired = repair_llm.invoke(repair_prompt)
            repaired_text = repaired.content if hasattr(repaired, 'content') else str(repaired)

            # 重新提取 + 解析
            match = re.search(r'\{[\s\S]*\}', repaired_text)
            if match:
                result = json.loads(match.group(0))
                if isinstance(result, dict):
                    logger.info("【智能修复成功】")
                    return result
        except Exception as e:
            logger.info(f"【智能修复失败】{e}")

    # 第三层：降级兜底（把纯文本当 response）
    logger.info("【JSON解析最终失败，降级为纯文本模式】")

    # 把整个纯文本当作response字段，其他字段为默认值
    degraded = fallback.copy()
    degraded["response"] = response
    degraded["need_more_info"] = True
    degraded["extracted_info"] = {}
    degraded["analysis"] = fallback.get("analysis", "解析失败，降级为纯文本输出")
    
    return degraded


class JsonResponseStreamer:
    """
    增量 JSON 流式解析器。

    从 LLM 逐 token 吐出的原始 JSON 中，实时提取最后一个 "response" 字段
    的可见文本。前提约定（写在所有 agent 的 prompt 里）：response 是 JSON
    的最后一个键，内容就是给用户看的完整回复。

    用法：API 层对每个节点维护一个 streamer，把 chunk 喂进去，拿到的返回值
    就是本次新增的可见文本（可直接发 SSE token 事件）。
    """
    def __init__(self):
        self.raw = ""       # 已累计的原始 JSON token
        self.emitted = ""   # 已发送出去的可见文本

    def feed(self, chunk: str) -> str:
        """喂入一个新 token，返回本次新增的可见文本（可能为空串）"""
        self.raw += chunk
        visible = self._extract_visible()
        if len(visible) <= len(self.emitted):
            return ""
        delta = visible[len(self.emitted):]
        self.emitted = visible
        return delta

    def _extract_visible(self) -> str:
        raw = self.raw
        # 1. 找最后一个 "response" 键（response 是 JSON 最后一个键）
        i = raw.rfind('"response"')
        if i == -1:
            return ""
        # 2. 跳过冒号和空白，必须是字符串起始引号
        j = raw.find(':', i)
        if j == -1:
            return ""
        j += 1
        while j < len(raw) and raw[j] in ' \t\r\n':
            j += 1
        if j >= len(raw) or raw[j] != '"':
            return ""
        # 3. 从起始引号往后扫，找闭合引号（跳过 \\ 转义）
        k = j + 1
        escaped = False
        while k < len(raw):
            c = raw[k]
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                break
            k += 1
        if k >= len(raw):
            # 字符串还没闭合：取到末尾；若末尾是悬空反斜杠（转义未完成），丢弃它
            seg = raw[j+1:-1] if raw[-1] == '\\' else raw[j+1:]
        else:
            seg = raw[j+1:k]
        # 4. 轻量反转义（qwen 中文是直接 UTF-8 输出，主要处理 \\n \\" \\\\）
        return seg.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')


if __name__ == "__main__":
    llm = LLM(temperature=TEMP_DEFAULT)
    chat_model = llm.get_chat_model()
    logger.info(f"拿到了 ChatOpenAI 实例：{type(chat_model)}")

    # 测试工具工厂（同步版本，只加载本地工具）
    tools = init_tools_sync(["search_knowledge"])
    logger.info(f"工具工厂加载了 {len(tools)} 个工具：{[t.name for t in tools]}")





