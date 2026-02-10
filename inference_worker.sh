#!/bin/bash

QUEUE_FILE="inference_queue.txt"
LOG_DIR="inference_logs"
CHECK_INTERVAL=5
THRESHOLD=500

# 建立 log 資料夾
mkdir -p "$LOG_DIR"

echo "Worker 啟動，正在監控 GPU 並自動記錄 Log 於 $LOG_DIR..."

while true; do
    if [ -s "$QUEUE_FILE" ]; then
        # 取得所有空閒 GPU ID
        all_free_gpus=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v t=$THRESHOLD -F', ' '$2 < t {print $1}')

        for gpu_id in $all_free_gpus; do
            if [ ! -s "$QUEUE_FILE" ]; then
                break
            fi

            # 領取任務並從檔案移除
            cmd=$(head -n 1 "$QUEUE_FILE")
            sed -i '1d' "$QUEUE_FILE"

            if [ -n "$cmd" ]; then
                # 1. 替換卡號
                final_cmd=$(echo "$cmd" | sed "s/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=$gpu_id/")
                
                # 2. 準備 Log 檔案路徑 (檔名包含時間與 GPU ID)
                timestamp=$(date +'%Y%m%d_%H%M%S')
                log_file="$LOG_DIR/log_${timestamp}_gpu${gpu_id}.log"

                echo "[$(date +'%H:%M:%S')] GPU $gpu_id 執行任務，Log: $log_file"

                # 3. 背景執行並寫入 Log
                # { ... } 括號內的東西會被視為一組輸出
                (
                    echo "================================================"
                    echo "START TIME: $(date)"
                    echo "COMMAND: $final_cmd"
                    echo "================================================"
                    echo ""
                    
                    # 執行指令並將 stdout 和 stderr 導向到 log
                    eval "$final_cmd"
                    
                    echo ""
                    echo "================================================"
                    echo "END TIME: $(date)"
                    echo "================================================"
                ) > "$log_file" 2>&1 &
                
                sleep 1
            fi
        done
    fi
    sleep $CHECK_INTERVAL
done