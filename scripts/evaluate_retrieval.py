"""
RAG 检索层评估脚本

评估指标：
- Precision@K：前K条结果中相关结果的占比
- Recall@K：前K条结果覆盖了多少应命中的相关文档
- F1@K：Precision和Recall的调和平均
- Hit Rate：前K条至少有1条相关的查询占比
- MRR：第一个相关结果排名的倒数平均

用法：python scripts/evaluate_retrieval.py
"""

import sys
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.knowledge_base import KnowledgeBase

@ dataclass
class TestQuery:
    """单条测试用例"""
    query: str
    relevant_docs: List[str]    # 应命中的文档文件名
    relevant_keywords: List[str]    # 关键词
    category: str = "general"    # 分类

@ dataclass
class QueryResult:
    """单条查询的评估结果"""
    query: str
    category: str
    precision: float
    recall: float
    f1: float
    hit: int
    mrr_rank: Optional[int] # 第一个相关结果的排名，None表示未命中
    relevance_list: List[bool]    # 相关文档列表，每个文档包含 source, content, score
    retrieved_sources: List[str] = field(default_factory=list) # 检索到的来源文件名

@ dataclass
class EvaluationSummary:
    """评估总结"""
    total_queries: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    hit_rate: float
    mrr: float
    by_category: dict = field(default_factory=dict) # 每个分类的评估结果



def load_test_queries(path: str) -> List[TestQuery]:
    """加载测试集，每行一个JSON"""
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                queries.append(TestQuery(
                    query=data['query'],
                    relevant_docs=data.get('relevant_docs', []),
                    relevant_keywords=data.get('relevant_keywords', []),
                    category=data.get('category', 'general')
                ))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[警告] 第{line_num}行解析失败: {e}")
    return queries



def is_result_relevant(
        content: str,
        source: str,
        relevant_docs: List[str],
        relevant_keywords: List[str]
) -> bool:
    """
    判断一条检索结果是否相关。
    满足以下任一条件即算相关：
    1. source 文件名在 relevant_docs 中
    2. 内容包含至少一个 relevant_keywords
    """
    # 文件名匹配
    source_filename = os.path.basename(source) if source else ""
    for doc in relevant_docs:
        if doc in source_filename or doc in source:
                       return True

    # 关键词匹配
    content_lower = content.lower()
    for kw in relevant_keywords:
        if kw.lower() in content_lower:
            return True

    return False



def evaluate_single_query(
        kb: KnowledgeBase,
        test_query: TestQuery,
        k: int = 4
) -> QueryResult:
    """评估单条查询"""
    # 检索
    # 检索（带分类过滤）
    category = test_query.category if test_query.category != "general" else None
    hybrid_docs = kb.hybrid_search(test_query.query, k=k, category=category)

    docs_with_score = [(doc, 0.0) for doc in hybrid_docs]
    

    # 逐条判断相关性
    relevance_list = []
    retrieved_sources = []
    hit_docs = set()     # 记录命中文档的文件名

    for doc, score in docs_with_score:
        source = doc.metadata.get('source', '') if doc.metadata else ''
        source_filename = os.path.basename(source)
        retrieved_sources.append(source_filename)
        
        relevant = is_result_relevant(doc.page_content, source, test_query.relevant_docs, test_query.relevant_keywords)
        relevance_list.append(relevant)

        # 如果这条结果相关，记录它命中了哪个 relevant_doc
        if relevant:
            for doc_name in test_query.relevant_docs:
                if doc_name in source_filename or doc_name in source:
                    hit_docs.add(doc_name)
                    break

    # 计算指标
    retrieved_relevant_count = sum(relevance_list)
    hit_docs_count = len(hit_docs)
    total_relevant_docs = len(test_query.relevant_docs)

    precision = retrieved_relevant_count / k if k > 0 else 0.0
    recall = hit_docs_count / total_relevant_docs if total_relevant_docs > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    hit = 1 if retrieved_relevant_count > 0 else 0

    # 计算 MRR: 第一个相关结果的倒数排名
    mrr_rank = None
    for i, relevant in enumerate(relevance_list):
        if relevant:
            mrr_rank = i + 1
            break

    return QueryResult(
        query=test_query.query,
        category=test_query.category,
        precision=precision,
        recall=recall,
        f1=f1,
        hit=hit,
        mrr_rank=mrr_rank,
        relevance_list=relevance_list,
        retrieved_sources=retrieved_sources
    )

def summarize(results: List[QueryResult]) -> EvaluationSummary:
    """汇总评估结果"""
    n = len(results)
    if n == 0:
        return EvaluationSummary(0, 0, 0, 0, 0, 0)

    avg_precision = sum(r.precision for r in results) / n
    avg_recall = sum(r.recall for r in results) / n
    avg_f1 = sum(r.f1 for r in results) / n
    hit_rate = sum(r.hit for r in results) / n

    mrr_sum = sum(1 / r.mrr_rank for r in results if r.mrr_rank is not None)
    mrr = mrr_sum / n

    # 按分类统计
    by_category = {}
    categories = set(r.category for r in results)
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cn = len(cat_results)
        by_category[cat] = {
             "count": cn,
             "precision": sum(r.precision for r in cat_results) / cn,
             "recall": sum(r.recall for r in cat_results) / cn,
             "hit_rate": sum(r.hit for r in cat_results) / cn,
             }

    return EvaluationSummary(
        total_queries=n,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        avg_f1=avg_f1,
        hit_rate=hit_rate,
        mrr=mrr,
        by_category=by_category
    )



def print_report(summary: EvaluationSummary, results: List[QueryResult], k: int):
    """打印评估报告"""
    print("=" * 70)
    print(f"RAG 检索层评估报告（K={k}）")
    print("=" * 70)
    print(f"  测试用例数:    {summary.total_queries}")
    print(f"  Precision@{k}:  {summary.avg_precision:.4f}  (前{k}条中相关结果占比)")
    print(f"  Recall@{k}:     {summary.avg_recall:.4f}  (应命中文档被覆盖的比例)")
    print(f"  F1@{k}:         {summary.avg_f1:.4f}  (Precision与Recall调和平均)")
    print(f"  Hit Rate:       {summary.hit_rate:.4f}  (至少命中1条的查询占比)")
    print(f"  MRR:            {summary.mrr:.4f}  (第一个相关结果排名倒数平均)")
    print()

    print("-" * 70)
    print("按分类统计")
    print("-" * 70)
    for cat, stats in summary.by_category.items():
        print(f"  [{cat}] {stats['count']}条 | "
              f"P@{k}={stats['precision']:.4f} | "
              f"R@{k}={stats['recall']:.4f} | "
              f"Hit={stats['hit_rate']:.4f}")
    print()

    # Bad Case 分析
    bad_cases = [r for r in results if r.hit == 0 or r.precision < 0.5]
    if bad_cases:
        print("=" * 70)
        print(f"Bad Case（{len(bad_cases)}条未命中或精确率<50%）")
        print("=" * 70)
        for r in bad_cases:
            print(f"\n  查询: {r.query}")
            print(f"  P={r.precision:.2f} R={r.recall:.2f} "
                  f"Hit={'✓' if r.hit else '✗'} "
                  f"首条相关排名={r.mrr_rank}")
            print(f"  检索来源: {r.retrieved_sources}")
            print(f"  相关性:   {r.relevance_list}")


def save_results(summary: EvaluationSummary, results: List[QueryResult], output_path: str):
    """保存评估结果到JSON文件"""
    output = {
        "summary": asdict(summary),
        "details": [asdict(r) for r in results],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"已保存评估结果到 {output_path}")



# 主入口
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(project_root, "data", "evaluation", "test_queries.jsonl")
    output_path = os.path.join(project_root, "data", "evaluation", "retrieval_result.json")

    # 加载知识库
    print("加载知识库...")
    kb = KnowledgeBase()
    if not kb.load():
        print("[错误] 无法加载向量库，请先运行 scripts/rebuild_knowledge_base.py")
        return

    # 加载测试集
    print("加载测试集...")
    test_queries = load_test_queries(test_path)
    print(f"成功加载 {len(test_queries)} 条测试用例")

    # 逐条评估
    K = 4
    results = []
    for i, tq in enumerate(test_queries, 1):
        print(f"  [{i}/{len(test_queries)}] {tq.query[:40]}...")
        result = evaluate_single_query(kb, tq, k=K)
        results.append(result)

    # 汇总结果
    summary = summarize(results)
    print_report(summary, results, K)
    save_results(summary, results, output_path)


if __name__ == "__main__":
    main()
