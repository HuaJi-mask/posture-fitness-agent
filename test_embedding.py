import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

print("API Key:", os.getenv("OPENAI_API_KEY")[:10] + "...")
print("Base URL:", os.getenv("OPENAI_BASE_URL"))
print("Model:", os.getenv("EMBEDDING_MODEL"))

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v2"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

print("\n测试单个文本...")
try:
    result = embeddings.embed_query("你好")
    print(f"成功！向量维度：{len(result)}")
except Exception as e:
    print(f"失败：{e}")

print("\n测试多个文本...")
try:
    results = embeddings.embed_documents(["你好", "世界"])
    print(f"成功！返回 {len(results)} 个向量")
except Exception as e:
    print(f"失败：{e}")