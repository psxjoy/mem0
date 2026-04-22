#!/bin/bash

# ===== 配置区 =====
APP_DIR="/root/mem0/server"
VENV_PATH="$APP_DIR/.venv"
LOG_FILE="$APP_DIR/mem0.log"
PID_FILE="$APP_DIR/mem0.pid"

# ===== 进入目录 =====
cd $APP_DIR || exit 1

# ===== 激活虚拟环境 =====
source $VENV_PATH/bin/activate

# ===== 检查是否已运行 =====
if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Mem0 已在运行，PID: $PID"
        exit 0
    else
        echo "检测到旧 PID 文件，已清理"
        rm -f $PID_FILE
    fi
fi

# ===== 启动服务 =====
echo "启动 Mem0..."

nohup uvicorn main:app --host 0.0.0.0 --port 8000 > $LOG_FILE 2>&1 &

# ===== 记录 PID =====
echo $! > $PID_FILE

echo "启动成功！"
echo "PID: $(cat $PID_FILE)"
echo "日志: $LOG_FILE"
