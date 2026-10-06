"""
知识库重建脚本
用法：python scripts/rebuild_knowledge_base.py
"""
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.knowledge_base import KnowledgeBase


def main():
    print("=" * 60)
    print("开始重建知识库")
    print("=" * 60)

    index_path = "data/faiss_index"

    # 1. 删除旧索引
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        print(f"已删除旧索引: {index_path}")

    # 2. 重新构建
    kb = KnowledgeBase(index_path=index_path)
    kb.build_from_directory(
        docs_dir="data/knowledge",
        chunk_size=500,
        chunk_overlap=50,
    )

    # 3. 简单验证
    print("\n" + "=" * 60)
    print("验证检索")
    print("=" * 60)
    test_queries = [
        "颈椎疼怎么办？",
        "扁平足怎么矫正？",
        "新手怎么增肌？",
        "胸肌训练动作有哪些？",
        "深蹲怎么做？",
    ]
    for query in test_queries:
        results = kb.search(query, k=2)
        print(f"\n查询: {query}")
        print(f"  检索到 {len(results)} 条结果")
        if results:
            print(f"  第一条: {results[0][:100]}...")

    print("\n" + "=" * 60)
    print("知识库重建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
