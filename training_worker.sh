#!/bin/bash

## chmod +x training_worker.sh
# 使用方式：bash training_worker.sh <GPU_ID>
# 例如：bash training_worker.sh 0

## training queue 格式
# 格式： "Config路徑 | Output資料夾名稱 | 目標Checkpoint名稱"
# 例如：
# /mnt/HDD3/miayan/paper/scriblit/config.yaml|train_ex8_12|checkpoint-235260

GPU_ID=$1
QUEUE_FILE="training_queue.txt"
LOCK_FILE="training_queue.lock"

# ================= 設定區 =================

# --- 排程啟動設定 ---
# ENABLE_SCHEDULE: 是否開啟定時功能？ ("true" = 開啟, "false" = 關閉/立刻執行)
ENABLE_SCHEDULE="true"

# START_TIME: 你想幾點開始跑？ (支援格式: "tomorrow 04:00", "03:00", "now + 5 hours")
# 範例 1: "tomorrow 04:00"  (明天凌晨 4 點)
# 範例 2: "23:30"           (今天的 23:30，如果已經過了會變成明天，視 date 指令而定，建議寫清楚 tomorrow)
START_TIME="21:00"

# --- VIP 迴避名單 ---
# 請在引號內填入 "同學的帳號名稱"，多個人用空白隔開
# 範例: VIP_USERS="alex bob teacher"
# 只要是這些人佔用 GPU，不管顯存大小，腳本都會乖乖等待
VIP_USERS="lin004"

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

# ================= 等待邏輯區塊 =================

if [ "$ENABLE_SCHEDULE" == "true" ]; then
    echo "⏰ 排程模式已開啟！目標啟動時間: $START_TIME"
    
    # 計算現在與目標時間的秒數差
    # date -d 是 Linux 的強大功能，能自動解析文字
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
        # 去除空白
        pid=$(echo $pid | xargs)
        used_mem=$(echo $used_mem | xargs)

        # 1. 取得該 Process 的使用者名稱 (Owner)
        local owner=$(ps -o user= -p $pid)
        owner=$(echo $owner | xargs) # 去空白

        # 2. 判斷邏輯
        # A. 如果是我自己
        if [ "$owner" == "$USER" ]; then
            # 進一步檢查是不是 train.py
            if ps -p $pid -o args= 2>/dev/null | grep -q "$SCRIPT_PATH"; then
                echo "BUSY_MY"
                return
            fi
            # 如果是我自己在跑別的東西 (例如 jupyter)，視為 BUSY_OTHER，以免自己打架
        fi

        # B. 如果是 VIP 名單裡的人 (絕對迴避)
        # 使用 grep 檢查 owner 是否在 VIP_USERS 字串中
        if [[ " $VIP_USERS " =~ " $owner " ]]; then
            echo "BUSY_VIP:$owner" # 回傳特殊狀態，並附上名字
            return
        fi

        # C. 如果是其他路人
        # 只有當顯存大於門檻時，才視為忙碌
        if [ "$used_mem" -gt "$MEM_THRESHOLD" ]; then
            final_status="BUSY_OTHER:$owner" # 標記是被誰佔用
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
            # === 重啟前的安全檢查 ===
            while true; do
                STATUS_RAW=$(check_gpu_status)
                STATUS=$(echo $STATUS_RAW | cut -d':' -f1)
                OWNER=$(echo $STATUS_RAW | cut -d':' -f2)

                if [ "$STATUS" == "BUSY_VIP" ]; then
                    # 遇到 VIP，改用 WAIT_INTERVAL (3分鐘)
                    echo -ne "⛔ 重啟暫停：VIP ($OWNER) 介入 | GPU $GPU_ID 等待中... $(date +'%H:%M:%S')\r"
                    sleep $WAIT_INTERVAL
                elif [ "$STATUS" == "BUSY_OTHER" ]; then
                    # 遇到路人，改用 WAIT_INTERVAL (3分鐘)
                    echo -ne "⛔ 重啟暫停：路人 ($OWNER) 佔用 | GPU $GPU_ID 等待中... $(date +'%H:%M:%S')\r"
                    sleep $WAIT_INTERVAL
                else
                    echo ""
                    echo "✅ GPU 狀態安全，執行救援重啟..."
                    break
                fi
            done

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

echo "🚀 啟動 GPU $GPU_ID 智慧工人 (VIP名單: $VIP_USERS)"
touch "$LOCK_FILE"

while true; do
    STATUS_RAW=$(check_gpu_status)
    
    # 解析狀態，因為可能是 "BUSY_VIP:lin004" 這種格式
    STATUS=$(echo $STATUS_RAW | cut -d':' -f1)
    OWNER=$(echo $STATUS_RAW | cut -d':' -f2)

    if [ "$STATUS" == "BUSY_MY" ]; then
        echo "🔍 GPU $GPU_ID 是我自己在跑！接手監控..."
        while [ "$(echo $(check_gpu_status) | cut -d':' -f1)" == "BUSY_MY" ]; do
            echo -ne "⏳ 監控既有任務中... $(date +'%H:%M:%S')\r"
            sleep 30
        done
        echo ""
        echo "✅ 既有任務結束。準備領取新任務..."
        continue

    elif [ "$STATUS" == "BUSY_VIP" ]; then
        # 遇到同學，絕對等待
        echo -ne "⛔ 禮讓 VIP ($OWNER) | GPU $GPU_ID 等待中... $(date +'%H:%M:%S')\r"
        sleep 60
        continue

    elif [ "$STATUS" == "BUSY_OTHER" ]; then
        # 遇到路人且顯存很高，等待
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