"""
状态定义
"""


from typing import TypedDict, List, Dict, Optional
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


def merge_user_profile(a: Optional[dict], b: Optional[dict]) -> dict:
    """合并用户画像

    体态/健身专家并行时，各自更新 user_profile 的不同子键（posture / fitness），
    需要 reducer 才能合并（否则同一键收到两个值会报
    INVALID_CONCURRENT_GRAPH_UPDATE）。子键也是 dict 时做一层深合并，避免相互覆盖。
    """
    merged = dict(a or {})
    for key, value in (b or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


class ProblemArea(TypedDict, total=False):
    # 单个问题部位的详细信息
    area: str  # 问题部位
    problem: str  # 问题描述
    symptoms: List[str]  # 症状列表
    pain_level: int  # 疼痛等级
    pain_duration: str  # 疼痛持续时间
    pain_type: str  # 疼痛类型
    pain_pattern: str  # 疼痛模式
    severity: str  # 严重程度

class SelfTestResult(TypedDict, total=False):
    # 单个自测结果
    test_name: str  # 自测项目名称
    result: str  # 自测结果
    description: str  # 用户自测结果描述
    confidence: float  # 自测结果置信度

class DailyHabits(TypedDict, total=False):
    # 日常习惯
    occupation: str  # 用户职业
    work_pattern: str  # 用户工作模式
    daily_sitting_hours: float  # 用户每天坐的时间
    sleep_duration: float  # 用户每天睡的时间
    sleep_quality: str  # 用户睡眠质量
    exercise_frequency: str  # 用户运动频率
    screen_time: float  # 用户每天使用屏幕的时间
    other_habits: List[str]  # 用户其他习惯

class PostureInfo(TypedDict, total=False):
    # 体态信息（问诊收集到的）
    problem_area: List[ProblemArea]  # 问题部位列表
    self_test_results: List[SelfTestResult]  # 自测结果列表
    daily_habits: DailyHabits  # 日常习惯信息
    medical_history: str  # 用户病史
    has_seen_doctor: bool  # 是否已就诊过医生
    doctor_diagnosis: str  # 医生诊断结果
    other_info: str  # 用户其他信息

class FitnessGoal(TypedDict, total=False):
    # 单个健身目标
    goal: str  # 健身目标
    priority: str  # 目标优先级
    target_time: str  # 目标时间
    target_detail: str  # 目标详细描述

class ExerciseHistory(TypedDict, total=False):
    # 运动历史
    years_of_training: float  # 用户运动年数
    break_duration: str  # 停练多久
    previous_training_type: str  # 上一个运动类型
    max_lifts: Dict[str, str]  # 最大重量记录
    current_training: str  # 当前运动类型

class FitnessInfo(TypedDict, total=False):
    # 健身需求信息
    goals: List[FitnessGoal]  # 健身目标列表
    fitness_level: str  # 健身等级
    exercise_history: ExerciseHistory  # 运动历史
    available_equipment: List[str]  # 可用的运动设备
    weekly_available_hours: float  # 每周可用时间
    training_days_per_week: int  # 每周训练天数
    preferred_days: List[str]  # 用户偏好训练天

    # 限制条件
    injury_restrictions: List[str]  # 伤害限制列表
    time_restrictions: List[str]  # 时间限制列表
    equipment_restrictions: List[str]  # 设备限制列表
    other_restrictions: List[str]  # 其他限制列表

    # 饮食相关
    diet_preference: str  # 用户饮食偏好
    diet_allergies: str  # 用户过敏食物
    diet_restrictions: List[str]  # 饮食限制列表

class UserProfile(TypedDict, total=False):
    # 基础信息
    age: int  # 用户年龄
    gender: str  # 用户性别
    height: float  # 用户身高
    weight: float  # 用户体重
    body_fat: float  # 用户体脂率

    # 体态信息
    posture: PostureInfo  # 体态信息
    fitness: FitnessInfo  # 健身需求信息

class ValidationResult(TypedDict, total=False):
    # 验证结果
    is_valid: bool  # 是否有效
    risk_level: str  # 验证反馈
    need_medical_attention: bool  # 是否需要专业医疗

    # 详细内容
    issue: List[Dict]  # 发现的问题列表
    suggestion: List[str]  # 建议的解决方案

    # 共识与分歧
    consensus_point: List[str]  # 共识点
    disagreement_point: List[Dict]  # 不一致点列表

class AgentState(TypedDict):
    user_input: str  # 用户输入
    reorganized_input : str  # 重组后的问题描述

    # 意图识别
    intent: str  # 意图识别结果(posture/finess/both/unknown)

    # 对话相关
    messages: Annotated[list, add_messages]  # 对话消息列表(完整的对话历史)
    chat_summary: str  # 对话摘要(压缩后的对话历史摘要)
    conversation_round: int  # 对话轮次（每跑一轮专家+1）
    current_stage: str  # 当前阶段(triage/posture/fitness/planning/adjustment/validation)
    need_more_info: bool  # 是否需要更多信息
    next_expert: str  # 下一个专家(posture/fitness/unknown)
    current_response: str  # 当前专家的回复

    # 统一用户画像（并行专家各自更新不同子键，用 reducer 合并）
    user_profile: Annotated[UserProfile, merge_user_profile]  # 用户画像(包含用户信息)

    # 诊断与目标
    posture_diagnosis: str  # 诊断结果
    correction_approach: str  # 矫正思路
    fitness_goal_summary: str  # 健身目标总结

    # 并行临时字段（both意图时用，汇聚后清空）
    # 注意：并行节点返回的键必须互斥，所以"需要更多信息/等待确认"也拆成
    # 各自的私有键，由 merge_expert_responses 合并成共享键
    posture_response: str  # 体态专家本轮回复
    fitness_response: str  # 健身专家本轮回复
    posture_need_more_info: bool  # 体态专家是否还需要更多信息（私有，汇聚时合并）
    fitness_need_more_info: bool  # 健身专家是否还需要更多信息（私有，汇聚时合并）
    posture_awaiting: str  # 体态专家等待确认的模块（私有，汇聚时合并）
    fitness_awaiting: str  # 健身专家等待确认的模块（私有，汇聚时合并）

    # 方案相关
    training_plan: str  # 训练计划
    diet_plan: str  # 饮食计划
    plan_summary: str  # 方案总结

    # 方案调整
    adjustment_round: int  # 方案调整轮次
    adjustment_history: list  # 方案调整历史
    user_feedback: str  # 用户反馈
    user_satisfied: str  # 用户满意度

    # 验证相关
    validation_round: int  # 验证轮次
    validation_a_result: ValidationResult  # 验证器A结果
    validation_b_result: ValidationResult  # 验证器B结果
    validation_consensus: bool  # 验证结果是否一致
    validation_report: str  # 验证报告
    validation_passed: bool  # 验证是否通过

    # 辩论相关
    debate_round: int   # 当前辩论轮次
    debate_history: list    # 辩论历史记录
    has_disagreement: bool  # 是否有分歧
    doctor_final: dict  # 医生最终结论
    coach_final: dict   # 教练最终结论

    # 确认流程
    awaiting_confirmation: str  # 正在等待用户确认的模块：posture/fitness/空字符串
    posture_diagnosis_confirmed: bool  # 体态诊断是否已被用户确认
    fitness_analysis_confirmed: bool  # 健身分析是否已被用户确认

    # 最终输出
    final_output: str  # 最终输出

    # 错误信息
    error_message: Optional[str]  # 错误信息

    # 方案自审
    self_check_passed: bool          # 自审是否通过
    self_check_suggestions: str      # 自审修改建议（不通过时填）
    self_check_round: int            # 自审轮次（最多2轮，防止死循环）

    # 反思总结
    reflection_summary: str


