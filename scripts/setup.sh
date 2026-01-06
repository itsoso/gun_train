#!/bin/bash

# 智能枪械训练系统 - 安装脚本

echo "🚀 开始安装智能枪械训练系统..."

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$python_version" ]]; then
    echo "❌ 未找到Python3，请先安装Python 3.9或更高版本"
    exit 1
fi
echo "✅ Python版本: $python_version"

# 创建虚拟环境
echo "创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建目录结构..."
mkdir -p logs
mkdir -p recordings
mkdir -p models
mkdir -p data/videos
mkdir -p data/clips

# 复制环境变量文件
if [ ! -f .env ]; then
    echo "创建环境变量文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件配置数据库等信息"
fi

# 数据库初始化提示
echo ""
echo "=" 50
echo "数据库设置"
echo "=" * 50
echo "请确保已安装并启动以下服务："
echo "1. PostgreSQL (端口: 5432)"
echo "2. MongoDB (端口: 27017)"
echo "3. Redis (端口: 6379)"
echo ""
echo "数据库初始化命令:"
echo "  python -m backend.db.database"
echo ""

# 下载预训练模型提示
echo "=" * 50
echo "AI模型设置"
echo "=" * 50
echo "需要下载以下模型文件到 models/ 目录:"
echo "1. MediaPipe姿态识别模型 (自动下载)"
echo "2. YOLOv8枪支检测模型 (需要训练)"
echo ""

echo "✅ 安装完成！"
echo ""
echo "启动命令:"
echo "  # 激活虚拟环境"
echo "  source venv/bin/activate"
echo ""
echo "  # 启动API服务器"
echo "  uvicorn backend.api.main:app --reload"
echo ""
echo "  # 启动Celery后台任务"
echo "  celery -A backend.tasks worker --loglevel=info"
echo ""
echo "访问 http://localhost:8000/docs 查看API文档"

