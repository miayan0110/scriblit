#!/bin/bash

# --- 設定區 ---
QUEUE_FILE="inference_queue.txt"
LOG_DIR="inference_logs"
CHECK_INTERVAL=30
IDLE_THRESHOLD=500      
MIN_FREE_MEM=22000      

# --- GPU 編號對照表 (已根據你最新的資訊修正) ---
declare -A GPU_MAP
GPU_MAP=( [0]=3 [1]=4 [2]=0 [3]=5 [4]=1 [5]=6 [6]=2 [7]=7 )

mkdir -p "$LOG_DIR"
touch "$QUEUE_FILE"

echo "========================================================="
echo "  GPU Worker 啟動 (詳細 Log 記錄模式)"
echo "  目標：剩餘空間 > ${MIN_FREE_MEM}MB 且無進程"
echo "========================================================="

while true; do
    if [ -s "$QUEUE_FILE" ]; then
        all_gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader)

        for gpu_id in $all_gpu_indices; do
            if [ ! -s "$QUEUE_FILE" ]; then break; fi

            # 取得顯卡資訊
            free_mem=$(nvidia-smi -i $gpu_id --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
            used_mem=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
            compute_pids=$(nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
            
            if ! [[ "$free_mem" =~ ^[0-9]+$ ]]; then continue; fi

            # 判定條件
            if [ "$free_mem" -ge "$MIN_FREE_MEM" ] && [ "$used_mem" -lt "$IDLE_THRESHOLD" ] && [ -z "$compute_pids" ]; then
                
                # 二次防禦
                sleep 1
                re_check_pids=$(nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
                if [ -n "$re_check_pids" ]; then continue; fi

                # 領取任務
                raw_cmd=$(head -n 1 "$QUEUE_FILE")
                [ -z "$raw_cmd" ] && sed -i '1d' "$QUEUE_FILE" && continue
                sed -i '1d' "$QUEUE_FILE"

                # 轉換編號與指令處理
                target_id=${GPU_MAP[$gpu_id]}
                final_cmd=$(echo "$raw_cmd" | sed "s/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=$target_id/")
                
                # 解析路徑
                target_script=$(echo "$final_cmd" | grep -o '/[^ ]*\.py' | head -n 1)
                target_dir=$( [ -n "$target_script" ] && dirname "$target_script" || echo "$(pwd)" )
                
                # Log 設定
                timestamp=$(date +'%Y%m%d_%H%M%S')
                log_file="$(pwd)/$LOG_DIR/log_${timestamp}_gpu${target_id}.log"

                echo "[$(date +'%H:%M:%S')] GPU $gpu_id (Free: ${free_mem}MB) -> 派發至 $target_id"

                # 執行與詳細記錄
                (
                    echo "------------------------------------------------"
                    echo "實驗啟動時間: $(date)"
                    echo "工作目錄:     $target_dir"
                    echo "原始指令:     $raw_cmd"
                    echo "分配後指令:   $final_cmd"
                    echo "Log 檔案路徑: $log_file"
                    echo "------------------------------------------------"
                    echo ""

                    cd "$target_dir" || exit
                    # 強制 python -u 確保輸出不緩衝
                    exec_cmd=$(echo "$final_cmd" | sed 's/python/python -u/')
                    nohup bash -c "$exec_cmd"
                    
                    echo ""
                    echo "------------------------------------------------"
                    echo "實驗結束時間: $(date)"
                    echo "------------------------------------------------"
                ) >> "$log_file" 2>&1 &
                
                sleep 3
            fi
        done
    fi
    sleep $CHECK_INTERVAL
done