#!/usr/bin/env bash
# ============================================
# macOS / Linux 一键启动脚本
# 自动完成：创建虚拟环境 -> 安装依赖 -> 启动服务
# ============================================
cd "$(dirname "$0")" || exit 1

# 首次运行：创建虚拟环境并安装依赖
if [ ! -d venv ]; then
    echo "[1/3] 首次运行，正在创建虚拟环境..."
    python3 -m venv venv || { echo "创建虚拟环境失败，请确认已安装 Python 3.10+"; exit 1; }
    echo "[2/3] 正在安装依赖（首次约需几分钟）..."
    venv/bin/pip install -r requirements.txt || { echo "依赖安装失败，请检查网络后重试"; exit 1; }
fi

# 检查 .env 是否存在，不存在则从模板创建并提示填写 Key
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "[提示] 已生成 .env 文件，请打开它填入你的 DEEPSEEK_API_KEY 后重新运行本脚本。"
    exit 0
fi

echo "[3/3] 正在启动服务..."
venv/bin/python -m uvicorn backend.main:app --reload
