"""
统一日志配置
"""
import logging
import sys

# 配置日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

# 创建根 logger
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
)

# 屏蔽一些过于啰嗦的第三方库日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取 logger 实例
    
    Args:
        name: 模块名，通常用 __name__
    
    Returns:
        Logger 实例
    """
    return logging.getLogger(name)
