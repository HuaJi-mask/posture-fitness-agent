# 安装依赖
FROM python:3.10-slim AS builder

WORKDIR /app

# 安装编译器依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# 运行镜像
FROM python:3.10-slim

WORKDIR /app

# 复制依赖
COPY --from=builder /root/.local /root/.local
ENV PATH="/root/.local/bin:$PATH"

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000')" || exit 1

# 启动命令
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
