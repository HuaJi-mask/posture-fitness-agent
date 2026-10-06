"""
健身动作数据集转换脚本
将 free-exercise-db 的 JSON 数据转换成结构化 Markdown 知识库文档
按肌群分类，每个肌群一个文件
"""
import json
import os
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = r"E:\智能体开发学习\智能体项目\体态矫正和健身计划规划智能体-手写"
INPUT_JSON = os.path.join(PROJECT_ROOT, "temp_exercises.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge", "fitness", "exercises")

# 肌群英文到中文的映射
MUSCLE_MAP = {
    "abdominals": "腹部",
    "abductors": "外展肌",
    "adductors": "内收肌",
    "biceps": "肱二头肌",
    "calves": "小腿",
    "chest": "胸部",
    "forearms": "前臂",
    "glutes": "臀部",
    "hamstrings": "腘绳肌",
    "lats": "背阔肌",
    "lower_back": "下背部",
    "middle_back": "中背部",
    "neck": "颈部",
    "quadriceps": "股四头肌",
    "shoulders": "肩部",
    "traps": "斜方肌",
    "triceps": "肱三头肌",
}

# 难度映射
LEVEL_MAP = {
    "beginner": "初学者",
    "intermediate": "中级",
    "expert": "高级",
}

# 器械映射
EQUIPMENT_MAP = {
    "body only": "徒手",
    "dumbbell": "哑铃",
    "barbell": "杠铃",
    "cable": "绳索",
    "machine": "器械",
    "kettlebells": "壶铃",
    "bands": "弹力带",
    "medicine ball": "药球",
    "exercise ball": "健身球",
    "foam roll": "泡沫轴",
    "e-z curl bar": "EZ曲杆",
    "other": "其他",
}


def format_exercise(exercise):
    """格式化单个动作为 Markdown"""
    name = exercise.get("name", "未知动作")
    level = LEVEL_MAP.get(exercise.get("level", ""), exercise.get("level", ""))
    equipment = EQUIPMENT_MAP.get(exercise.get("equipment", ""), exercise.get("equipment", ""))
    mechanic = exercise.get("mechanic", "")
    force = exercise.get("force", "")
    primary = ", ".join([MUSCLE_MAP.get(m, m) for m in exercise.get("primaryMuscles", [])])
    secondary = ", ".join([MUSCLE_MAP.get(m, m) for m in exercise.get("secondaryMuscles", [])])
    instructions = exercise.get("instructions", [])

    md = f"## {name}\n\n"
    md += f"- **难度**：{level}\n"
    md += f"- **器械**：{equipment}\n"
    md += f"- **主要肌群**：{primary}\n"
    if secondary:
        md += f"- **次要肌群**：{secondary}\n"
    if mechanic:
        md += f"- **动作类型**：{mechanic}\n"
    if force:
        md += f"- **发力类型**：{force}\n"
    md += "\n"

    if instructions:
        md += "**动作步骤**：\n\n"
        for i, step in enumerate(instructions, 1):
            # 清理乱码字符
            step = step.replace("戮", "3/4").replace("?", "")
            md += f"{i}. {step}\n"
        md += "\n"

    return md


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取 JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        exercises = json.load(f)

    print(f"共加载 {len(exercises)} 个动作")

    # 按主要肌群分组
    muscle_groups = defaultdict(list)
    for ex in exercises:
        primary_muscles = ex.get("primaryMuscles", [])
        if primary_muscles:
            # 取第一个主要肌群作为分类
            muscle_groups[primary_muscles[0]].append(ex)
        else:
            muscle_groups["other"].append(ex)

    print(f"共 {len(muscle_groups)} 个肌群分类")

    # 每个肌群生成一个 Markdown 文件
    total_written = 0
    for muscle_en, exercises_list in sorted(muscle_groups.items()):
        muscle_cn = MUSCLE_MAP.get(muscle_en, muscle_en)
        filename = f"{muscle_en}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # 文件头
        content = f"# {muscle_cn}训练动作库\n\n"
        content += f"> 来源：free-exercise-db（开源健身动作数据集）\n"
        content += f"> 动作数量：{len(exercises_list)}\n"
        content += f"> 更新日期：2026-09-28\n\n"
        content += "---\n\n"

        # 按难度排序
        level_order = {"beginner": 0, "intermediate": 1, "expert": 2}
        exercises_list.sort(key=lambda x: level_order.get(x.get("level", ""), 3))

        # 写入每个动作
        for ex in exercises_list:
            content += format_exercise(ex)
            content += "---\n\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        total_written += len(exercises_list)
        print(f"  ✓ {muscle_cn}：{len(exercises_list)} 个动作 → {filename}")

    print(f"\n转换完成！共写入 {total_written} 个动作到 {OUTPUT_DIR}")

    # 清理临时文件
    if os.path.exists(INPUT_JSON):
        os.remove(INPUT_JSON)
        print(f"已清理临时文件：{INPUT_JSON}")


if __name__ == "__main__":
    main()
