"""
RAG 检索评估脚本
评估指标：Precision@k、Recall@k、F1@k、命中率、MRR
用法：python scripts/evaluate_rag.py
"""
import sys
import os
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.knowledge_base import KnowledgeBase


def load_test_queries(path):
    """加载测试集，每行一个JSON"""
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def is_relevant(result_content, relevant_docs, relevant_keywords, source=None):
    """
    判断一条检索结果是否相关
    两种判断方式（满足其一即可）：
    1. source 文件名在 relevant_docs 中
    2. 内容包含至少一个 relevant_keywords
    """
    # 方式1：通过 source 判断（如果 metadata 里有 source）
    if source:
        for doc in relevant_docs:
            if doc in source:
                return True

    # 方式2：通过关键词判断
    content_lower = result_content.lower()
    for kw in relevant_keywords:
        if kw.lower() in content_lower:
            return True

    return False


def evaluate_query(kb, query_item, k=4):
    """
    评估单个查询
    返回：precision, recall, f1, hit, mrr_rank（第一个相关结果的排名，无则为None）
    """
    query = query_item['query']
    relevant_docs = query_item.get('relevant_docs', [])
    relevant_keywords = query_item.get('relevant_keywords', [])

    # 检索（带source）
    docs_with_score = kb.vector_store.similarity_search_with_score(query, k=k)

    # 判断每条结果是否相关
    relevance_list = []
    for doc, score in docs_with_score:
        source = doc.metadata.get('source', '') if doc.metadata else ''
        relevant = is_relevant(doc.page_content, relevant_docs, relevant_keywords, source)
        relevance_list.append(relevant)

    # 计算指标
    retrieved_relevant = sum(relevance_list)
    total_relevant = len(relevant_docs)  # 用相关文档数作为分母

    precision = retrieved_relevant / k if k > 0 else 0
    recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    hit = 1 if retrieved_relevant > 0 else 0

    # MRR：第一个相关结果的排名倒数
    mrr_rank = None
    for i, rel in enumerate(relevance_list):
        if rel:
            mrr_rank = i + 1
            break

    return {
        'query': query,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hit': hit,
        'mrr_rank': mrr_rank,
        'relevance_list': relevance_list,
        'results': [(doc.page_content[:60].replace('\n', ' '), doc.metadata.get('source', '').split('\\')[-1] if doc.metadata else '') for doc, _ in docs_with_score],
    }


def main():
    print("=" * 70)
    print("RAG 检索评估")
    print("=" * 70)

    # 初始化知识库
    kb = KnowledgeBase()
    if not kb.load():
        print("错误：无法加载向量库，请先运行 scripts/rebuild_knowledge_base.py")
        return

    # 加载测试集
    test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'evaluation', 'test_queries.jsonl')
    test_queries = load_test_queries(test_path)
    print(f"加载测试集：{len(test_queries)} 条查询\n")

    # 评估每个查询
    k = 4
    all_results = []
    bad_cases = []

    for item in test_queries:
        result = evaluate_query(kb, item, k=k)
        all_results.append(result)

        # 收集 bad case（未命中或精确率低）
        if result['hit'] == 0 or result['precision'] < 0.5:
            bad_cases.append(result)

    # 汇总指标
    avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
    avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
    avg_f1 = sum(r['f1'] for r in all_results) / len(all_results)
    hit_rate = sum(r['hit'] for r in all_results) / len(all_results)

    # MRR
    mrr_sum = 0
    for r in all_results:
        if r['mrr_rank'] is not None:
            mrr_sum += 1 / r['mrr_rank']
    mrr = mrr_sum / len(all_results)

    # 按主题分类统计
    posture_queries = [r for r in all_results if any(kw in r['query'] for kw in ['扁平足', '上交叉', '骨盆', '头前引', '体态', '圆肩', '塌腰', '脖子疼', '高低肩', '头晕', '腰疼'])]
    fitness_queries = [r for r in all_results if r not in posture_queries]

    def avg_metric(results, metric):
        return sum(r[metric] for r in results) / len(results) if results else 0

    print("=" * 70)
    print(f"总体评估结果（k={k}）")
    print("=" * 70)
    print(f"  Precision@{k}:  {avg_precision:.4f}  (前{k}条中相关结果占比)")
    print(f"  Recall@{k}:     {avg_recall:.4f}  (前{k}条覆盖了多少应命中的文档)")
    print(f"  F1@{k}:         {avg_f1:.4f}  (Precision和Recall的调和平均)")
    print(f"  命中率:         {hit_rate:.4f}  (前{k}条至少有1条相关的比例)")
    print(f"  MRR:            {mrr:.4f}  (第一个相关结果排名的倒数平均)")
    print()

    print("-" * 70)
    print(f"体态类查询（{len(posture_queries)}条）")
    print(f"  Precision@{k}:  {avg_metric(posture_queries, 'precision'):.4f}")
    print(f"  Recall@{k}:     {avg_metric(posture_queries, 'recall'):.4f}")
    print(f"  命中率:         {avg_metric(posture_queries, 'hit'):.4f}")
    print()
    print(f"健身类查询（{len(fitness_queries)}条）")
    print(f"  Precision@{k}:  {avg_metric(fitness_queries, 'precision'):.4f}")
    print(f"  Recall@{k}:     {avg_metric(fitness_queries, 'recall'):.4f}")
    print(f"  命中率:         {avg_metric(fitness_queries, 'hit'):.4f}")
    print()

    # Bad Case 分析
    print("=" * 70)
    print(f"Bad Case 分析（{len(bad_cases)}条未命中或精确率<50%）")
    print("=" * 70)
    for r in bad_cases:
        print(f"\n  查询: {r['query']}")
        print(f"  精确率: {r['precision']:.2f} | 召回率: {r['recall']:.2f} | 命中: {'是' if r['hit'] else '否'}")
        print(f"  检索结果来源: {[src for _, src in r['results']]}")
        print(f"  相关性: {r['relevance_list']}")

    # 保存详细结果
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'evaluation', 'evaluation_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'hit_rate': hit_rate,
            'mrr': mrr,
            'total_queries': len(all_results),
            'bad_cases_count': len(bad_cases),
            'details': [{k: v for k, v in r.items() if k != 'results'} for r in all_results],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
