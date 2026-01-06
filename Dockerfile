# 多阶段构建 - 基础镜像
FROM python:3.9-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 生产环境镜像
FROM base as production

# 复制应用代码
COPY backend /app/backend
COPY frontend /app/frontend
COPY models /app/models
COPY scripts /app/scripts

# 创建必要的目录
RUN mkdir -p /app/logs /app/recordings /app/data

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 默认命令
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

