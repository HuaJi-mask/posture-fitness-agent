"""
RAG 生成层评估脚本（LLM-as-Judge）

用大模型当评委，评估 RAG 生成回答的质量：
- Faithfulness（忠实度）：回答是否基于检索上下文，有无幻觉
- Answer Relevancy（回答相关性）：回答是否切题
- Context Precision（上下文精确率）：检索到的上下文有多少被回答用到

不需要人工标注标准答案，用检索到的上下文作为评判依据。

用法：python scripts/evaluate_generation.py
"""

import sys
import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.knowledge_base import KnowledgeBase
from src.utils.llm import LLM, TEMP_PRECISE

@dataclass
class GenerationResult:
    """单条查询的生成层评估结果"""
    query: str
    answer: str
    context: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    judge_reason: str

@dataclass
class GenerationSummary:
    """评估总结"""
    total_queries: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_overall: float  # 三个指标的平均值

# 评委系统提示：定义评分标准
JUDGE_SYSTEM_PROMPT = """你是一位严格的 RAG 系统评估专家。你的任务是根据检索到的上下文，评估 AI 助手回答的质量。

请从以下三个维度打分，每个维度 0-1 分（保留两位小数），并给出简短理由：

1. Faithfulness（忠实度）：回答中的信息是否都能在上下文中找到依据？
   - 1.0：回答完全基于上下文，没有编造
   - 0.5：大部分基于上下文，有少量延伸但不违背事实
   - 0.0：回答包含上下文没有的信息（幻觉）

2. Answer Relevancy（回答相关性）：回答是否直接回应了用户的问题？
   - 1.0：完全切题，直接回答了问题
   - 0.5：部分相关，有跑题或遗漏
   - 0.0：答非所问

3. Context Precision（上下文精确率）：检索到的上下文有多少被回答真正用到？
   - 1.0：回答充分利用了上下文中的关键信息
   - 0.5：用到了部分上下文，有一些上下文没被使用
   - 0.0：回答几乎没有用到上下文

输出严格为 JSON 格式，不要输出其他内容：
{
  "faithfulness": 0.00,
  "answer_relevancy": 0.00,
  "context_precision": 0.00,
  "reason": "简短的评分理由"
}"""

# 生成回答的系统提示
GENERATION_SYSTEM_PROMPT = """你是一位专业的体态矫正和健身规划助手。请根据提供的参考资料，用中文回答用户的问题。

要求：
1. 回答必须基于参考资料，不要编造资料中没有的信息
2. 如果参考资料不足以回答问题，请明确说明
3. 回答要结构清晰，重点突出
4. 适当使用加粗和分点，方便阅读"""


def genrate_answer(llm: LLM, query: str, context: str) -> str:
    """基于检索上下文生成回答"""
    prompt = f"""参考资料：
{context}

用户问题：{query}

请根据参考资料回答用户问题。"""
    return llm.chat(prompt, system_prompt=GENERATION_SYSTEM_PROMPT)



def judge_answer(llm: LLM, query: str, context: str, answer: str) -> Tuple[float, float, float, str]:
    """
    用 LLM 当评委，给回答打分。
    返回 (faithfulness, answer_relevancy, context_precision, reason)
    """
    judge_prompt = f"""用户问题：{query}

检索到的上下文：
{context}

AI 助手的回答：
{answer}

请根据评分标准对以上回答进行评估。"""

    response = llm.chat(judge_prompt, system_prompt=JUDGE_SYSTEM_PROMPT)

    # 健壮的 JSON 解析：处理 markdown 代码块包裹、不完整 JSON 等情况
    try:
        # 去掉 ```json 和 ``` 包裹
        cleaned = response.strip()
        if cleaned.startswith('```'):
            # 去掉第一行 ```json 或 ```
            lines = cleaned.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            cleaned = '\n'.join(lines)

        # 找第一个 { 和最后一个 }，提取 JSON 主体
        start = cleaned.find('{')
        end = cleaned.rfind('}')

        if start != -1 and end != -1 and end > start:
            json_str = cleaned[start:end + 1]
            data = json.loads(json_str)
        else:
            # 没找到完整 JSON，尝试直接解析
            data = json.loads(cleaned)

        faithfulness = float(data.get('faithfulness', 0.0))
        answer_relevancy = float(data.get('answer_relevancy', 0.0))
        context_precision = float(data.get('context_precision', 0.0))
        reason = data.get('reason', '')

        # 限制在 0-1 之间
        faithfulness = max(min(faithfulness, 1.0), 0.0)
        answer_relevancy = max(min(answer_relevancy, 1.0), 0.0)
        context_precision = max(min(context_precision, 1.0), 0.0)

        return faithfulness, answer_relevancy, context_precision, reason

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"    [警告] 评委输出解析失败: {e}")
        print(f"    原始输出前200字: {response[:200]}")
        return 0.0, 0.0, 0.0, f"解析失败: {e}"



def evaluate_single_query(
        kb: KnowledgeBase,
        gen_llm: LLM,
        judge_llm: LLM,
        query: str,
        k: int = 4
) -> GenerationResult:
    """评估单条查询的生成质量"""
    # 检索上下文
    docs = kb.search(query, k=k)
    context = "\n\n---\n\n".join(docs) if docs else "（未检索到相关资料）"

    # 生成回答
    answer = genrate_answer(gen_llm, query, context)

    # 评委打分
    faithfulness, answer_relevancy, context_precision, reason = judge_answer(judge_llm, query, context, answer)

    return GenerationResult(
        query=query,
        answer=answer,
        context=context,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        judge_reason=reason
    )


def summarize_generation(results: List[GenerationResult]) -> GenerationSummary:
    """汇总生成层评估结果"""
    n = len(results)
    if n == 0:
        return GenerationSummary(0, 0, 0, 0, 0)

    avg_faith = sum(r.faithfulness for r in results) / n
    avg_relev = sum(r.answer_relevancy for r in results) / n
    avg_ctx = sum(r.context_precision for r in results) / n
    avg_overall = (avg_faith + avg_relev + avg_ctx) / 3

    return GenerationSummary(
        total_queries=n,
        avg_faithfulness=avg_faith,
        avg_answer_relevancy=avg_relev,
        avg_context_precision=avg_ctx,
        avg_overall=avg_overall
    )


def print_generation_report(summary: GenerationSummary, results: List[GenerationResult]):
    """打印生成层评估报告"""
    print("=" * 70)
    print("RAG 生成层评估报告（LLM-as-Judge）")
    print("=" * 70)
    print(f"  测试用例数:          {summary.total_queries}")
    print(f"  Faithfulness:        {summary.avg_faithfulness:.4f}  (回答忠实度，有无幻觉)")
    print(f"  Answer Relevancy:    {summary.avg_answer_relevancy:.4f}  (回答是否切题)")
    print(f"  Context Precision:   {summary.avg_context_precision:.4f}  (上下文利用率)")
    print(f"  综合得分:            {summary.avg_overall:.4f}  (三项平均)")
    print()

    # 低分案例分析（低于0.6的）
    low_score = [r for r in results
                 if (r.faithfulness + r.answer_relevancy + r.context_precision) / 3 < 0.6]
    if low_score:
        print("=" * 70)
        print(f"低分案例（{len(low_score)}条综合分<0.6）")
        print("=" * 70)
        for r in low_score:
            print(f"\n  查询: {r.query}")
            print(f"  忠实度={r.faithfulness:.2f} 相关性={r.answer_relevancy:.2f} "
                  f"上下文利用={r.context_precision:.2f}")
            print(f"  评委理由: {r.judge_reason[:150]}")
            print(f"  回答预览: {r.answer[:100]}...")


def save_generation_results(summary: GenerationSummary, results: List[GenerationResult], output_path: str):
    """保存生成层评估结果"""
    output = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results]
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"已保存生成层评估结果到 {output_path}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(project_root, "data", "evaluation", "test_queries.jsonl")
    output_path = os.path.join(project_root, "data", "evaluation", "generation_results.json")

    # 加载知识库
    print("加载知识库...")
    kb = KnowledgeBase()
    if not kb.load():
        print("[错误] 无法加载向量库")
        return

    # 加载测试集
    print(f"加载测试集: {test_path}")
    test_queries = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            try:
                test_queries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [警告] 第{line_num}行解析失败，已跳过: {e}")

    print(f"共 {len(test_queries)} 条测试用例\n")

    # 初始化 LLM
    gen_llm = LLM(temperature=0.3)
    judge_llm = LLM(temperature=TEMP_PRECISE)

    # 逐条评估
    results = []
    for i, item in enumerate(test_queries, 1):
        query = item["query"]
        print(f"  [{i}/{len(test_queries)}] {query[:40]}...")
        result = evaluate_single_query(kb, gen_llm, judge_llm, query, k=4)
        results.append(result)
        print(f"    忠实度={result.faithfulness:.2f} 相关性={result.answer_relevancy:.2f} "
              f"上下文利用={result.context_precision:.2f}")

    # 汇总结果
    print()
    summary = summarize_generation(results)
    print_generation_report(summary, results)
    save_generation_results(summary, results, output_path)

if __name__ == '__main__':
    main()