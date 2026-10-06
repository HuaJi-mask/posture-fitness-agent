"""临时检索测试脚本"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
kb.load()

queries = ['扁平足怎么矫正', '上交叉综合征训练', '骨盆前倾怎么办', '头前引矫正', '体态评估方法']
for q in queries:
    print(f'\n查询: {q}')
    results = kb.search_with_score(q, k=2)
    for content, score in results:
        preview = content[:100].replace('\n', ' ')
        print(f'  分数: {score:.4f} | {preview}...')
