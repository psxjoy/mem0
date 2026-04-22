#!/bin/bash

APP_DIR="/root/mem0/server"
PID_FILE="$APP_DIR/mem0.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "没有运行中的 Mem0"
    exit 0
fi

PID=$(cat $PID_FILE)

if ps -p $PID > /dev/null 2>&1; then
    echo "正在停止 Mem0 (PID: $PID)..."
    kill $PID
    rm -f $PID_FILE
    echo "已停止"
else
    echo "进程不存在，清理 PID 文件"
    rm -f $PID_FILE
fi
