#!/bin/bash

## chmod +x training_worker.sh
# 使用方式：bash training_worker.sh <GPU_ID>
# 例如：bash training_worker.sh 0

## training queue 格式
# 格式： "Config路徑 | Output資料夾名稱 | 目標Checkpoint名稱"
# 例如：
# /mnt/HDD3/miayan/paper/scriblit/config.yaml|train_ex10_1|checkpoint-235260

GPU_ID=$1
QUEUE_FILE="training_queue.txt"
LOCK_FILE="training_queue.lock"

# ================= 設定區 =================

# --- 排程啟動設定 ---
# ENABLE_SCHEDULE: 是否開啟定時功能？ ("true" = 開啟, "false" = 關閉/立刻執行)
ENABLE_SCHEDULE="false"

# START_TIME: 你想幾點開始跑？ (支援格式: "tomorrow 04:00", "03:00", "now + 5 hours")
# 範例 1: "tomorrow 04:00"  (明天凌晨 4 點)
# 範例 2: "23:30"           (今天的 23:30，如果已經過了會變成明天，視 date 指令而定，建議寫清楚 tomorrow)
START_TIME="tomorrow 00:00"

# --- VIP 迴避名單 ---
# 請在引號內填入 "同學的帳號名稱"，多個人用空白隔開
# 範例: VIP_USERS="alex bob teacher"
# 只要是這些人佔用 GPU，不管顯存大小，腳本都會乖乖等待
VIP_USERS=""

# --- 參數設定 ---
PYTHON_BIN="/mnt/HDD3/miayan/paper/envs/scriblit/bin/python3.10"
SCRIPT_PATH="train.py"
VALIDATION_FILE="custom_unet.pth"
CHECK_INTERVAL=30    # 監控"我自己"的任務：每 30 秒檢查一次 (保持敏銳)
WAIT_INTERVAL=180    # 等待"別人"釋放 GPU：每 3 分鐘檢查一次 (不用太頻繁)
# 顯存門檻 (MB)：如果 GPU 即使有 process 但吃少於這個數字，視為空閒 (可搶)
MEM_THRESHOLD=25000 

if [ -z "$GPU_ID" ]; then
    echo "❌ 請指定 GPU ID (例如: ./training_worker.sh 0)"
    exit 1
fi

# 自動計算 Port (避免雙卡衝突)
CURRENT_PORT=$((29500 + GPU_ID))
ACC_ARGS="--main_process_port=$CURRENT_PORT"

# ================= 輔助函式 (時間與Log解析) =================

# 1. 計算已監控時間
get_monitor_duration() {
    local start_ts=$1
    local now_ts=$(date +%s)
    local diff=$((now_ts - start_ts))
    
    local h=$((diff / 3600))
    local m=$(( (diff % 3600) / 60 ))
    local s=$((diff % 60))
    
    printf "%02d:%02d:%02d" $h $m $s
}

# 2. 解析 Log 檔抓取更詳細資訊
parse_log_progress() {
    local log_file=$1
    if [ ! -f "$log_file" ]; then
        echo "等待 Log..."
        return
    fi

    # 讀取最後包含 % 的一行 (避免讀取整份文件)
    local last_line=$(tail -c 2000 "$log_file" | tr '\r' '\n' | grep "%" | tail -n 1)

    if [ -z "$last_line" ]; then
        echo "分析中..."
        return
    fi

    # 1. 抓取百分比 (例如 51%)
    local pct=$(echo "$last_line" | grep -o "[0-9]\+%" | head -1)
    
    # 2. 抓取步數 (例如 119916/235260)
    local steps=$(echo "$last_line" | grep -o "[0-9]\+/[0-9]\+" | head -1)

    # 3. 抓取時間 [已跑<剩餘]
    local times=$(echo "$last_line" | sed -n 's/.*\[\([^<]*\)< \?\([^,]*\),.*/\1 < \2/p')

    # 組合輸出
    if [ -z "$times" ]; then
        echo "$pct"
    else
        # 格式: 51% [119916/235260] 43:48:50 < 41:54:55
        if [ ! -z "$steps" ]; then
             echo "$pct [$steps] $times"
        else
             echo "$pct | $times"
        fi
    fi
}

# ================= 等待邏輯區塊 =================

if [ "$ENABLE_SCHEDULE" == "true" ]; then
    echo "⏰ 排程模式已開啟！目標啟動時間: $START_TIME"
    
    # 計算現在與目標時間的秒數差
    TARGET_SEC=$(date -d "$START_TIME" +%s)
    NOW_SEC=$(date +%s)
    DIFF_SEC=$((TARGET_SEC - NOW_SEC))
    
    if [ $DIFF_SEC -gt 0 ]; then
        # 把秒數換算成小時分鐘顯示給你看
        WAIT_HRS=$((DIFF_SEC / 3600))
        WAIT_MIN=$(( (DIFF_SEC % 3600) / 60 ))
        
        echo "💤 現在時間: $(date +'%H:%M:%S')"
        echo "⏳ 腳本將進入睡眠，等待 $WAIT_HRS 小時 $WAIT_MIN 分鐘..."
        echo "   (預計於 $(date -d "$START_TIME" +'%Y-%m-%d %H:%M:%S') 醒來開工)"
        
        # 讓腳本睡覺
        sleep $DIFF_SEC
        
        echo ""
        echo "⏰ 鈴鈴鈴！時間到了！工人起床開始檢查 GPU $GPU_ID..."
    else
        echo "⚠️  注意：設定的時間 ($START_TIME) 已經過去了，腳本將立即開始執行！"
    fi
else
    echo "🚀 排程模式未開啟，立即開始執行..."
fi

# ================= 核心函式 =================

# 1. 檢查 GPU 狀態 (回傳: "BUSY_MY", "BUSY_OTHER", "FREE")
check_gpu_status() {
    # 取得 PID 和 Memory
    local proc_info=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)

    if [ -z "$proc_info" ]; then
        echo "FREE"
        return
    fi

    # 預設狀態
    local final_status="FREE"

    # 逐行檢查每個 Process
    while IFS=, read -r pid used_mem; do
        pid=$(echo $pid | xargs); used_mem=$(echo $used_mem | xargs)
        local owner=$(ps -o user= -p $pid); owner=$(echo $owner | xargs)

        # A. 如果是我自己
        if [ "$owner" == "$USER" ]; then
            if ps -p $pid -o args= 2>/dev/null | grep -q "$SCRIPT_PATH"; then
                echo "BUSY_MY"
                return
            fi
        fi

        # B. 如果是 VIP 名單裡的人 (絕對迴避)
        if [[ " $VIP_USERS " =~ " $owner " ]]; then
            echo "BUSY_VIP:$owner"
            return
        fi

        # C. 如果是其他路人 (超過門檻才算忙)
        if [ "$used_mem" -gt "$MEM_THRESHOLD" ]; then
            final_status="BUSY_OTHER:$owner"
        fi
    done <<< "$proc_info"

    echo "$final_status"
}

# 2. 執行監控與救援 (Watchdog)
run_watchdog() {
    local config_path=$1
    local output_dir=$2
    local target_ckpt=$3
    local target_dir="./$output_dir/$target_ckpt"

    local current_log_file=""
    local MONITOR_START_TIME=$(date +%s)
    
    # 取得資料夾名稱 (只顯示最後一層，比較乾淨)
    local DIR_NAME=$(basename "$output_dir")

    echo "🛡️  進入監控模式 ($target_ckpt)..."

    while true; do
        # 檢查是否還在跑 (只看我自己的 process)
        local my_running=false
        local pids=$(pgrep -u "$USER" -f "$SCRIPT_PATH")
        
        for pid in $pids; do
             if nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader | grep -q "$pid"; then
                 my_running=true
                 break
             fi
        done

        if [ "$my_running" = true ]; then
            # === 顯示詳細進度 ===
            if [ -z "$current_log_file" ]; then
                current_log_file=$(ls -t "$output_dir"/train_log_*.txt 2>/dev/null | head -n 1)
            fi
            
            if [ ! -z "$current_log_file" ]; then
                PROGRESS_INFO=$(parse_log_progress "$current_log_file")
            else
                PROGRESS_INFO="Log初始化中..."
            fi

            DURATION=$(get_monitor_duration $MONITOR_START_TIME)

            # 格式: ⏳ GPU 0 | 正在執行: train_ex8 | 進度: 51% [steps] time | 監控: 00:05:30
            echo -ne "⏳ GPU $GPU_ID | 正在執行: $DIR_NAME | 進度: $PROGRESS_INFO | 監控: $DURATION\r"
            
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
            # === 重啟前的安全檢查 ===
            while true; do
                # 檢查 GPU 狀態
                STATUS_RAW=$(check_gpu_status)
                STATUS=$(echo $STATUS_RAW | cut -d':' -f1)
                OWNER=$(echo $STATUS_RAW | cut -d':' -f2)

                if [ "$STATUS" == "BUSY_VIP" ]; then
                    echo -ne "⛔ 重啟暫停：VIP ($OWNER) 介入 | GPU $GPU_ID 等待中... $(date +'%H:%M:%S')\r"
                    sleep $WAIT_INTERVAL
                elif [ "$STATUS" == "BUSY_OTHER" ]; then
                    echo -ne "⛔ 重啟暫停：路人 ($OWNER) 佔用 | GPU $GPU_ID 等待中... $(date +'%H:%M:%S')\r"
                    sleep $WAIT_INTERVAL
                else
                    echo ""
                    echo "✅ GPU 狀態安全，執行救援重啟..."
                    break
                fi
            done

            mkdir -p "$output_dir"
            local new_log_name="train_log_$(date +%Y%m%d_%H%M).txt"
            current_log_file="./$output_dir/$new_log_name"

            FULL_CMD="export PYTHONUNBUFFERED=1; CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch $ACC_ARGS $SCRIPT_PATH --config $config_path --output_dir $output_dir"
            echo "執行: $FULL_CMD"
            nohup bash -c "$FULL_CMD" > "$current_log_file" 2>&1 &
            
            MONITOR_START_TIME=$(date +%s)
            
            sleep 20
            echo "👀 已重啟，繼續監控..."
        fi
    done
}

# ================= 主流程 =================

echo "🚀 啟動 GPU $GPU_ID 智慧工人 (VIP名單: $VIP_USERS)"
touch "$LOCK_FILE"

while true; do
    STATUS_RAW=$(check_gpu_status)
    STATUS=$(echo $STATUS_RAW | cut -d':' -f1)
    OWNER=$(echo $STATUS_RAW | cut -d':' -f2)

    if [ "$STATUS" == "BUSY_MY" ]; then
        echo "🔍 GPU $GPU_ID 是我自己在跑！接手監控..."
        
        # 尋找最近的 log 來推斷現在在跑哪個實驗
        RECENT_LOG=$(find . -name "train_log_*.txt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')
        MONITOR_START_TIME=$(date +%s)
        
        # 從 Log 路徑推算資料夾名稱
        if [ ! -z "$RECENT_LOG" ]; then
            TASK_NAME=$(basename "$(dirname "$RECENT_LOG")")
        else
            TASK_NAME="手動任務"
        fi

        while [ "$(echo $(check_gpu_status) | cut -d':' -f1)" == "BUSY_MY" ]; do
             if [ ! -z "$RECENT_LOG" ]; then
                PROGRESS_INFO=$(parse_log_progress "$RECENT_LOG")
             else
                PROGRESS_INFO="Log解析中..."
             fi
             DURATION=$(get_monitor_duration $MONITOR_START_TIME)
             
             echo -ne "⏳ GPU $GPU_ID | 正在執行: $TASK_NAME | 進度: $PROGRESS_INFO | 監控: $DURATION\r"
             sleep $CHECK_INTERVAL
        done
        continue

    elif [ "$STATUS" == "BUSY_VIP" ]; then
        echo -ne "⛔ 禮讓 VIP ($OWNER) | GPU $GPU_ID 等待中... $(date +'%H:%M:%S')\r"
        sleep 60
        continue

    elif [ "$STATUS" == "BUSY_OTHER" ]; then
        echo -ne "⛔ 路人 ($OWNER) 佔用高顯存 | GPU $GPU_ID 等待中... $(date +'%H:%M:%S')\r"
        sleep 60
        continue
    fi

    # === FREE (可以搶票) ===
    
    NEXT_TASK=""
    exec 200>"$LOCK_FILE"
    flock -x 200
    if [ -s "$QUEUE_FILE" ]; then
        NEXT_TASK=$(head -n 1 "$QUEUE_FILE" | tr -d '\r')
        sed -i '1d' "$QUEUE_FILE"
    fi
    flock -u 200
    
    if [ -z "$NEXT_TASK" ]; then
        echo -ne "💤 任務池空了，GPU $GPU_ID 待機中... $(date +'%H:%M:%S')\r"
        sleep 60
    else
        echo "🎉 GPU $GPU_ID 搶到任務！"
        echo "   內容: $NEXT_TASK"
        
        IFS="|" read -r q_cfg q_out q_ckpt <<< "$NEXT_TASK"
        
        if [ -z "$q_cfg" ] || [ -z "$q_out" ]; then
            echo "⚠️  格式錯誤跳過..."
            continue
        fi

        mkdir -p "$q_out"
        log_file="./$q_out/train_log_$(date +%Y%m%d_%H%M).txt"
        
        FULL_CMD="export PYTHONUNBUFFERED=1; CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch $ACC_ARGS $SCRIPT_PATH --config $q_cfg --output_dir $q_out"
        
        echo "🚀 啟動: $q_out"
        nohup bash -c "$FULL_CMD" > "$log_file" 2>&1 &
        sleep 20 
        
        run_watchdog "$q_cfg" "$q_out" "$q_ckpt"
    fi
done