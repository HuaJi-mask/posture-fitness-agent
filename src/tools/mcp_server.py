"""
MCP Server - 体态矫正与健身规划工具集
把项目工具封装成 MCP 标准接口，供支持 MCP 的客户端调用

使用新版 FastMCP（mcp >= 1.0）
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os


from mcp.server.fastmcp import FastMCP
from src.rag.get_relevant_knowledge import get_relevant_knowledge

from src.utils.logger import get_logger
logger = get_logger(__name__)




# 创建 MCP 实例（FastMCP 是新版推荐用法）
mcp = FastMCP("PostureFitnessTools")


# ===== 工具1：知识检索 =====
@mcp.tool()
def search_knowledge(query: str, top_k: int = 3) -> str:
    """
    从体态/健身知识库检索相关专业知识。

    当需要专业知识支撑分析、不确定症状成因、想查矫正方法、
    用户提到不熟悉的专业术语时调用。

    Args:
        query: 检索关键词，要用专业术语，比如"上交叉综合征的成因和矫正方法"
        top_k: 返回最相关的几条，默认3条

    Returns:
        检索到的知识文本
    """
    result = get_relevant_knowledge(query, k=top_k)
    if not result or result == "":
        return "未命中相关知识"
    return "检索到的资料：\n" + result


# ===== 工具2：BMI 计算 =====
@mcp.tool()
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """
    计算用户的 BMI 指数并给出评价。

    Args:
        weight_kg: 体重（公斤）
        height_cm: 身高（厘米）

    Returns:
        BMI 值和对应的健康评价
    """
    if height_cm <= 0 or weight_kg <= 0:
        return "错误：身高和体重必须大于0"

    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)

    if bmi < 18.5:
        category = "偏瘦"
        advice = "建议适当增加营养摄入，结合力量训练增肌"
    elif bmi < 24:
        category = "正常"
        advice = "体重正常，继续保持规律运动和均衡饮食"
    elif bmi < 28:
        category = "超重"
        advice = "建议控制饮食热量，增加有氧运动，每周至少150分钟中等强度运动"
    else:
        category = "肥胖"
        advice = "建议在医生指导下制定减重计划，循序渐进，避免过度节食"

    return f"""BMI 计算结果：
- BMI 值：{bmi:.1f}
- 分类：{category}
- 建议：{advice}
（中国标准：偏瘦<18.5，正常18.5-23.9，超重24.0-27.9，肥胖≥28）"""


# ===== 工具3：热量计算 =====
@mcp.tool()
def calculate_calories(weight_kg: float, height_cm: float, age: int, gender: str, activity_level: str) -> str:
    """
    计算用户的每日基础代谢率(BMR)和总能量消耗(TDEE)。

    Args:
        weight_kg: 体重（公斤）
        height_cm: 身高（厘米）
        age: 年龄（岁）
        gender: 性别，"male"或"female"
        activity_level: 活动水平，可选：sedentary（久坐）/light（轻度活动）
                        /moderate（中度活动）/active（高度活动）/very_active（极高活动）

    Returns:
        BMR和TDEE计算结果，以及增肌/减脂的热量建议
    """
    # Mifflin-St Jeor 公式
    if gender.lower() == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    activity_factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    factor = activity_factors.get(activity_level.lower(), 1.2)
    tdee = bmr * factor

    return f"""热量计算结果：
- 基础代谢率(BMR)：{bmr:.0f} 大卡/天
- 每日总消耗(TDEE)：{tdee:.0f} 大卡/天
- 增肌建议：{tdee + 300:.0f} 大卡/天（热量盈余300）
- 减脂建议：{tdee - 500:.0f} 大卡/天（热量缺口500）
- 维持建议：{tdee:.0f} 大卡/天"""


# ===== 启动 Server =====
if __name__ == "__main__":
    logger.info("MCP Server 启动中...")
    logger.info("工具列表：search_knowledge, calculate_bmi, calculate_calories")
    mcp.run()
