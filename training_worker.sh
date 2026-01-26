#!/bin/bash

## chmod +x training_worker.sh
# 使用方式：bash training_worker.sh <GPU_ID>
# 例如：bash training_worker.sh 0

GPU_ID=$1
QUEUE_FILE="training_queue.txt"
LOCK_FILE="training_queue.lock"

# --- 設定區 ---
PYTHON_BIN="/mnt/HDD3/miayan/paper/envs/scriblit/bin/python3.10"
SCRIPT_PATH="train.py"
VALIDATION_FILE="custom_unet.pth"
CHECK_INTERVAL=30
# 顯存門檻 (MB)：如果 GPU 即使有 process 但吃少於這個數字，視為空閒 (可搶)
MEM_THRESHOLD=10000 

if [ -z "$GPU_ID" ]; then
    echo "❌ 請指定 GPU ID (例如: ./training_worker.sh 0)"
    exit 1
fi

# 自動計算 Port (避免雙卡衝突)
CURRENT_PORT=$((29500 + GPU_ID))
ACC_ARGS="--main_process_port=$CURRENT_PORT"

# ================= 核心函式 =================

# 1. 檢查 GPU 狀態 (回傳: "BUSY_MY", "BUSY_OTHER", "FREE")
check_gpu_status() {
    # 取得該 GPU 上所有 process 的 PID 和 使用者ID
    # 格式: PID, UID, USED_MEMORY
    local proc_info=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)

    if [ -z "$proc_info" ]; then
        echo "FREE"
        return
    fi

    # 讀取每一行 Process
    local is_free="FREE"
    
    while IFS=, read -r pid used_mem; do
        # 檢查是不是我在跑 train.py
        if ps -p $pid -o args= 2>/dev/null | grep -q "$SCRIPT_PATH"; then
            # 檢查 owner 是不是我
            local owner=$(ps -o user= -p $pid)
            if [ "$owner" == "$USER" ]; then
                echo "BUSY_MY"
                return
            fi
        fi

        # 如果不是我的 train.py，檢查顯存佔用
        # 去除空白
        used_mem=$(echo $used_mem | xargs)
        if [ "$used_mem" -gt "$MEM_THRESHOLD" ]; then
            is_free="BUSY_OTHER"
        fi
    done <<< "$proc_info"

    echo "$is_free"
}

# 2. 執行監控與救援 (Watchdog)
run_watchdog() {
    local config_path=$1
    local output_dir=$2
    local target_ckpt=$3
    local target_dir="./$output_dir/$target_ckpt"

    echo "🛡️  進入監控模式 ($target_ckpt)..."

    while true; do
        # 檢查是否還在跑 (只看我自己的 process)
        local my_running=false
        local pids=$(pgrep -u "$USER" -f "$SCRIPT_PATH")
        
        # 這裡要過濾，確保該 PID 真的是跑在目前這張 GPU 上
        for pid in $pids; do
             # 用 nvidia-smi 查這個 pid 有沒有用這張卡
             if nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader | grep -q "$pid"; then
                 my_running=true
                 break
             fi
        done

        if [ "$my_running" = true ]; then
            echo -ne "⏳ GPU $GPU_ID | 正在執行: $output_dir | $(date +'%H:%M')\r"
            sleep $CHECK_INTERVAL
            continue
        fi

        # Process 停了，驗收
        echo ""
        echo "⚠️  GPU $GPU_ID Process 停止！檢查結果..."

        if [ -d "$target_dir" ] && [ -f "$target_dir/$VALIDATION_FILE" ]; then
            echo "✅ 任務完成！"
            return 0 # 成功，返回主迴圈去領下一個任務
        else
            echo "❌ 任務未完成 (OOM或中斷)。"
            echo "🔄 10秒後原地救援重啟..."
            sleep 10
            
            mkdir -p "$output_dir"
            log_file="./$output_dir/train_log_$(date +%Y%m%d_%H%M).txt"
            
            FULL_CMD="export PYTHONUNBUFFERED=1; CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch $ACC_ARGS $SCRIPT_PATH --config $config_path --output_dir $output_dir"
            
            echo "執行: $FULL_CMD"
            nohup bash -c "$FULL_CMD" > "$log_file" 2>&1 &
            
            sleep 20
            echo "👀 已重啟，繼續監控..."
        fi
    done
}

# ================= 主流程 =================

echo "🚀 啟動 GPU $GPU_ID 智慧工人 (閾值: ${MEM_THRESHOLD}MB)"

while true; do
    STATUS=$(check_gpu_status)

    if [ "$STATUS" == "BUSY_MY" ]; then
        echo "🔍 發現 GPU $GPU_ID 已經有我的任務在跑！直接接手監控..."
        # 這裡比較尷尬，因為我們不知道現在跑的是哪個 config
        # 但我們可以「盲目監控」：只要 process 死掉且 queue 裡有東西，就假設舊的跑完了
        # 為了安全，這裡我們做一個簡單的等待迴圈，直到它死掉
        while [ "$(check_gpu_status)" == "BUSY_MY" ]; do
            echo -ne "⏳ 監控既有任務中... $(date +'%H:%M:%S')\r"
            sleep 30
        done
        echo ""
        echo "✅ 既有任務結束 (或中斷)。準備領取新任務..."
        continue

    elif [ "$STATUS" == "BUSY_OTHER" ]; then
        echo -ne "⛔ GPU $GPU_ID 被其他人佔用 (VRAM > ${MEM_THRESHOLD}MB)，等待中... $(date +'%H:%M:%S')\r"
        sleep 60
        continue
    fi

    # === STATUS == FREE (可以領任務了) ===
    
    # 去 queue 搶任務
    NEXT_TASK=""
    (
        flock -x 200
        if [ -s "$QUEUE_FILE" ]; then
            NEXT_TASK=$(head -n 1 "$QUEUE_FILE")
            sed -i '1d' "$QUEUE_FILE"
        fi
    ) 200>"$LOCK_FILE"

    if [ -z "$NEXT_TASK" ]; then
        echo -ne "💤 任務池空了，GPU $GPU_ID 待機中... $(date +'%H:%M:%S')\r"
        sleep 60
    else
        echo "🎉 GPU $GPU_ID 搶到任務！"
        IFS="|" read -r q_cfg q_out q_ckpt <<< "$NEXT_TASK"
        
        # 1. 啟動任務
        mkdir -p "$q_out"
        log_file="./$q_out/train_log_$(date +%Y%m%d_%H%M).txt"
        
        FULL_CMD="export PYTHONUNBUFFERED=1; CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch $ACC_ARGS $SCRIPT_PATH --config $q_cfg --output_dir $q_out"
        
        echo "🚀 啟動: $q_out"
        nohup bash -c "$FULL_CMD" > "$log_file" 2>&1 &
        sleep 20 # 等待啟動
        
        # 2. 進入監控 (直到這個任務做完)
        run_watchdog "$q_cfg" "$q_out" "$q_ckpt"
    fi
done