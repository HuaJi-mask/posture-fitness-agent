"""
统一配置管理
所有配置都从这里读，不要在各个模块里散落配置
"""
import os
from dotenv import load_dotenv

# 加载 .env（只加载一次）
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


class Config:
    """配置类"""
    
    # 大模型配置
    MODEL_NAME = os.getenv("MODEL_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    
    # 向量库配置
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    
    # 知识库路径
    KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "data/knowledge")
    
    # 验证配置
    MAX_VALIDATION_ROUNDS = int(os.getenv("MAX_VALIDATION_ROUNDS", "3"))


# 全局配置实例
config = Config()