"""
MCP Client 统一封装
用官方 MultiServerMCPClient 连接多个 MCP Server，统一获取工具
带缓存，连接生命周期全局管理

主流做法：
- MCP Server 独立运行（mcp_server.py），对外暴露工具
- MCP Client 连接多个 Server，统一获取工具列表
- 工具加载带缓存，避免重复建立连接
- 连接全局持有，程序退出时关闭
"""
import os
import sys
import asyncio

from langchain_core.tools import BaseTool

from src.utils.logger import get_logger

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = get_logger(__name__)


# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MCP_SERVER_PATH = os.path.join(PROJECT_ROOT, "src", "tools", "mcp_server.py")

# 全局缓存
_tools_cache = None
_mcp_client = None  # 全局持有 MCP Client，保持连接打开
_mcp_client_lock = asyncio.Lock()  # 用于异步安全访问 _mcp_client


async def load_mcp_tools() -> list[BaseTool]:
    """
    连接所有配置的 MCP Server，获取所有 MCP 工具。
    带缓存，只连接一次。连接全局持有，程序退出时调用 close_mcp_client() 关闭。

    Returns:
        所有 MCP 工具列表（LangChain BaseTool 格式，可直接 bind_tools）
    """
    global _tools_cache, _mcp_client, _mcp_client_lock


    async with _mcp_client_lock:
        if _tools_cache is not None:
            return _tools_cache


        # MCP Server 配置（统一在一个地方管理，主流做法）
        mcp_servers = {
            # 我们自己的 MCP Server（stdio 方式，开发环境用）
            "posture-fitness": {
                "transport": "stdio",
                "command": sys.executable,  # 当前 Python 解释器
                "args": ["-m", "src.tools.mcp_server"],
                "cwd": PROJECT_ROOT,
            },
            # 外部 MCP Server 示例（HTTP 方式，生产环境用）
            # "weather": {
            #     "transport": "streamable_http",
            #     "url": "http://localhost:8000/mcp",
            # },
        }

        logger.info(f"【MCP Client】正在连接 {len(mcp_servers)} 个 MCP Server...")

        # MultiServerMCPClient 不能用作上下文管理器，直接创建 client 调用 get_tools()
        # 连接由 client 内部管理，工具调用时自动保持连接
        client = MultiServerMCPClient(mcp_servers)
        _mcp_client = client  # 全局持有，方便程序退出时清理

        tools = await client.get_tools()

        logger.info(f"【MCP Client】加载完成，共 {len(tools)} 个工具：")
        for t in tools:
            logger.info(f"  - {t.name}")

        _tools_cache = tools
        return tools


async def close_mcp_client():
    """
    关闭 MCP Client 连接，释放子进程。
    程序退出时调用，避免子进程挂起。
    """
    global _mcp_client, _tools_cache
    if _mcp_client is not None:
        try:
            # MultiServerMCPClient 没有统一的 close 方法，删除引用让 GC 清理
            # 子进程会在主进程退出时自动终止
            _mcp_client = None
            _tools_cache = None
            logger.info("【MCP Client】连接已释放")
        except Exception as e:
            logger.info(f"【MCP Client】释放连接时出错：{e}")


def get_mcp_tools_by_names(names: list[str]) -> list[BaseTool]:
    """
    按名称获取 MCP 工具子集（同步包装，方便在非 async 函数里用）。

    Args:
        names: 需要的工具名称列表

    Returns:
        工具列表
    """
    import asyncio
    all_tools = asyncio.run(load_mcp_tools())
    name_set = set(names)
    return [t for t in all_tools if t.name in name_set]


# ==================== 测试 ====================
if __name__ == "__main__":
    import asyncio

    async def test():
        logger.info("=" * 60)
        logger.info("MCP Client 测试")
        logger.info("=" * 60)

        # 测试加载所有工具
        tools = await load_mcp_tools()
        logger.info(f"\n共加载 {len(tools)} 个工具")

        # 测试调用 calculate_bmi（MCP 工具只支持异步 ainvoke）
        logger.info("\n" + "=" * 60)
        logger.info("测试调用 calculate_bmi：")
        bmi_tool = next((t for t in tools if t.name == "calculate_bmi"), None)
        if bmi_tool:
            result = await bmi_tool.ainvoke({"weight_kg": 70, "height_cm": 175})
            # MCP 工具返回的是内容列表，提取 text
            if isinstance(result, list) and len(result) > 0:
                logger.info(result[0].get("text", str(result)))
            else:
                logger.info(result)
        else:
            logger.info("未找到 calculate_bmi 工具")

        # 测试按名称筛选
        logger.info("\n" + "=" * 60)
        logger.info("测试按名称筛选：")
        filtered = [t for t in tools if t.name in ["calculate_bmi", "calculate_calories"]]
        logger.info(f"筛选到 {len(filtered)} 个工具：{[t.name for t in filtered]}")

        # 关闭连接
        await close_mcp_client()
        logger.info("\n测试完成！")

    asyncio.run(test())
