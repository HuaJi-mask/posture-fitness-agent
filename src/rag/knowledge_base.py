"""
知识库定义
"""


import sys
import os
import pickle


from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from src.utils.config import config

from langchain_community.retrievers import BM25Retriever

import pickle

from sentence_transformers import CrossEncoder

from src.utils.logger import get_logger
logger = get_logger(__name__)


# 加载环境变量
load_dotenv()

class KnowledgeBase:
    """
    知识库类
    """

    def __init__(self, index_path: str = "data/faiss_index"):
        """
        初始化知识库

        Args:
            index_path: 知识库保存路径
        """
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            check_embedding_ctx_length=False,
        )
        self.vector_store = None

        self.bm25_retriever = None  # BM25索引，构建时初始化
        self.all_splits = []    # 保存所有文本块，BM25和混合检索都要用

    def build_from_directory(self, docs_dir: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        从目录构建知识库
        
        Args:
            docs_dir: 文档目录
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        logger.info(f"开始从目录 {docs_dir} 构建知识库...")

        # 加载文档
        loader = DirectoryLoader(
            docs_dir, 
            glob="**/*.md", 
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        logger.info(f"成功加载 {len(documents)} 个文档")

        # 文本块分割
        text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"成功分割 {len(splits)} 个文本块")

        # 自动给每个文本块打分类标签（category + doc_type）
        for doc in splits:
            source_path = doc.metadata.get("source", "").replace("\\", "/")

            # 一级分类：posture / fitness / both
            if "/posture/" in source_path or "posture_" in source_path:
                doc.metadata["category"] = "posture"
            elif "/fitness/" in source_path or "fitness_" in source_path:
                doc.metadata["category"] = "fitness"
            else:
                doc.metadata["category"] = "both"  # 根目录下的基础文件

            # 二级分类：overview 概览 / exercise 具体动作 / guide 专题指南
            if "/exercises/" in source_path:
                doc.metadata["doc_type"] = "exercise"  # 具体动作
            elif "/posture/" in source_path:
                doc.metadata["doc_type"] = "guide"  # 体态专题指南
            else:
                doc.metadata["doc_type"] = "overview"  # 根目录基础文件（概览类）

        # 统计各分类数量
        category_counts = {}
        doc_type_counts = {}
        for doc in splits:
            cat = doc.metadata.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            dtype = doc.metadata.get("doc_type", "unknown")
            doc_type_counts[dtype] = doc_type_counts.get(dtype, 0) + 1
        logger.info(f"一级分类统计: {category_counts}")
        logger.info(f"二级分类统计: {doc_type_counts}")

        # 向量化并存入FAISS（分批处理，避免API批量大小限制）
        logger.info(f"开始向量化，共 {len(splits)} 个文本块，分批处理（每批20个）...")
        import time

        batch_size = 20  # 阿里云Embedding API限制每批最多20条
        total_batches = (len(splits) + batch_size - 1) // batch_size

        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"  处理第 {batch_num}/{total_batches} 批（{len(batch)} 个文本块）...")

            if i == 0:
                # 第一批创建索引
                self.vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                # 后续批次追加到索引
                self.vector_store.add_documents(batch)

            # 每批之间短暂延迟，避免API限流
            if i + batch_size < len(splits):
                time.sleep(0.5)

        # 保存所有 splits 到 self.all_splits（BM25 和混合检索都要用）
        self.all_splits = splits

        # 构建BM25索引
        self.bm25_retriever = BM25Retriever.from_documents(
            splits,
            k=10,
        )

        # 保存向量库
        self.save()
        logger.info("向量化完成")


    def save(self):
        """
        保存向量库
        """
        if self.vector_store:
            self.vector_store.save_local(self.index_path)
            logger.info(f"向量库已保存到 {self.index_path}")

        with open(os.path.join(self.index_path, "bm25_corpus.pkl"), "wb") as f:
            pickle.dump(self.all_splits, f)
        
    def load(self):
        """
        加载向量库
        """
        bm25_path = os.path.join(self.index_path, "bm25_corpus.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                self.all_splits = pickle.load(f)
            self.bm25_retriever = BM25Retriever.from_documents(
                self.all_splits,
                k=10,
            )
            logger.info(f"BM25索引已从 {bm25_path} 加载")


        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,   # 本地文件，信任它
            )
            logger.info(f"向量库已从 {self.index_path}加载")
            return True
        else:
            logger.info(f"向量库文件不存在: {self.index_path}")
            return False

    def search(self, query: str, k: int = 4) -> List[str]:
        """
        检索相关文档片段
        
        Args:
            query: 查询字符串
            k: 返回的文档片段数量
        
        Returns:
            List[str]: 相关文档片段列表
        """
        if not self.vector_store:
            # 尝试加载
            if not self.load():
                return []

        docs = self.vector_store.similarity_search(query, k=k)

        # 返回文档内容
        return [doc.page_content for doc in docs]

    def search_with_score(self, query: str, k: int = 4):
        """
        检索相关文档片段并返回相似度分数
        
        Args:
            query: 查询字符串
            k: 返回的文档片段数量
        
        Returns:
            List[Dict[str, Any]]: 包含文档内容和相似度分数的字典列表
        """
        if not self.vector_store:
            # 尝试加载
            if not self.load():
                return []

        docs_with_score = self.vector_store.similarity_search_with_score(query, k=k)

        return [(doc.page_content, score) for doc, score in docs_with_score]

    def hybrid_search(self, query:str, k: int = 4, alpha: float = 0.75, category: str = None):
        """
        混合检索：向量检索 + BM25 关键词检索，用 RRF 融合
        
        Args:
            query: 查询
            k: 返回结果数
            alpha: 向量检索权重
            category: 可选，按分类过滤（posture/fitness/both），None表示不过滤
        """
        # 多召回更多向量检索结果，用于后面融合排序
        vector_docs = self.vector_store.similarity_search(query, k=k * 10)

        # BM25 检索同样召回更多
        bm25_docs = self.bm25_retriever.invoke(query)[:k * 10]

        # 按分类过滤
        if category:
            vector_docs = [doc for doc in vector_docs if doc.metadata.get("category") in [category, "both"]]
            bm25_docs = [doc for doc in bm25_docs if doc.metadata.get("category") in [category, "both"]]

        # RRF 融合
        rrf_k = 60
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + alpha * (1 / (rank + 1 + rrf_k))

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1 / (rank + 1 + rrf_k))

        # 按融合得分排序，取前k条
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        from src.rag.reranker import rerank

        # 融合后取前 10 条候选，再 rerank
        candidates = [doc_map[doc_id] for doc_id in sorted_ids[:20]]
        final_docs = rerank(query, candidates, top_k=k)

        return final_docs



if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("检索功能测试")
    logger.info("=" * 60)

    kb = KnowledgeBase()
    kb.load()

    # 测试用例：包含正向查询、关键词查询、bad case
    test_queries = [
        "扁平足怎么矫正",
        "新手怎么开始健身",   # 之前的 bad case
        "深蹲标准动作",
        "上交叉综合征训练",
    ]

    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"查询: {query}")
        logger.info('=' * 60)

        # 纯向量检索
        logger.info("\n【纯向量检索】")
        vector_results = kb.search(query, k=4)
        for i, r in enumerate(vector_results):
            logger.info(f"  {i+1}. {r[:80].replace(chr(10), ' ')}...")

        # 混合检索（含 rerank）
        logger.info("\n【混合检索+Rerank】")
        hybrid_results = kb.hybrid_search(query, k=4)
        for i, r in enumerate(hybrid_results):
            logger.info(f"  {i+1}. {r[:80].replace(chr(10), ' ')}...")

        # 简单对比
        if vector_results[0][:50] != hybrid_results[0][:50]:
            logger.info("\n  ⚠️  第一名结果不同")
        else:
            logger.info("\n  ✓  第一名结果相同")


    