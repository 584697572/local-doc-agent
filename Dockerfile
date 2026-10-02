FROM python:3.13-slim

# 日志立即输出，且不在源码目录生成 Python 字节码缓存。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# libgomp1 提供 FAISS / PyTorch 所需的 OpenMP 运行库。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依赖先于源码复制，修改业务代码时可以复用已安装依赖的镜像层。
# 此部署不使用 GPU，先安装 CPU 版 Torch，避免拉取 CUDA 运行库。
COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-runtime.txt \
    && python -m pip check

# 只复制服务需要的代码；私有文档、索引和密钥由运行时提供。
COPY agent.py config.py llm_client.py ./
COPY api/ ./api/
COPY harness/ ./harness/
COPY retrieval/ ./retrieval/
COPY tools/ ./tools/
RUN mkdir -p /app/data /app/.localdoc_index /root/.cache/huggingface

EXPOSE 8000

# 只探测 HTTP 服务，不触发模型下载、检索或付费 LLM 调用。
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); assert r.status == 200"]

# 单进程运行，保持当前 Agent / Retriever 的锁与缓存行为不变。
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
