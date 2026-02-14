#!/bin/bash

# --- 設定區 ---
QUEUE_FILE="inference_queue.txt"
LOG_DIR="inference_logs"
CHECK_INTERVAL=5
IDLE_THRESHOLD=500      

# --- GPU 編號對照表 ---
declare -A GPU_MAP
GPU_MAP=( [0]=3 [1]=4 [2]=0 [3]=5 [4]=1 [5]=6 [6]=2 [7]=7 )

mkdir -p "$LOG_DIR"
touch "$QUEUE_FILE"

# 用於控制監控條刷新的旗標
FIRST_RUN=true

echo "========================================================="
echo "  🚀 GPU Inference Worker 智慧分流版 (雙行固定刷新)"
echo "  規則：S (11GB) / L (22GB) 自動適配顯卡"
echo "  [執行中] 僅限您的帳號並顯示黃色/紅/綠變色"
echo "========================================================="

while true; do
    if [ -s "$QUEUE_FILE" ]; then
        all_gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader)
        for gpu_id in $all_gpu_indices; do
            if [ ! -s "$QUEUE_FILE" ]; then break; fi

            full_line=$(head -n 1 "$QUEUE_FILE")
            if [[ "$full_line" == *"|"* ]]; then
                task_type=$(echo "$full_line" | cut -d'|' -f1 | xargs)
                raw_cmd=$(echo "$full_line" | cut -d'|' -f2- | xargs)
            else
                task_type="L"
                raw_cmd="$full_line"
            fi

            if [ "$task_type" == "S" ]; then CURRENT_MIN_FREE=11000; else CURRENT_MIN_FREE=22000; fi

            free_mem=$(nvidia-smi -i $gpu_id --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
            used_mem=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
            compute_pids=$(nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
            
            if [ "$free_mem" -ge "$CURRENT_MIN_FREE" ] && [ "$used_mem" -lt "$IDLE_THRESHOLD" ] && [ -z "$compute_pids" ]; then
                sed -i '1d' "$QUEUE_FILE"
                target_id=${GPU_MAP[$gpu_id]}
                final_cmd=$(echo "$raw_cmd" | sed "s/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=$target_id/")
                timestamp=$(date +'%Y%m%d_%H%M%S')
                log_file="$(pwd)/$LOG_DIR/log_${timestamp}_gpu${target_id}.log"

                # 任務通知
                echo -e "\n[$(date +'%H:%M:%S')] 🚀 GPU $gpu_id 搶到任務 ($task_type): $final_cmd"
                FIRST_RUN=true

                printf -- "------------------------------------------------\n" > "$log_file"
                printf -- "實驗啟動時間: $(date)\n" >> "$log_file"
                printf -- "任務類型:     $task_type (門檻: ${CURRENT_MIN_FREE}MB)\n" >> "$log_file"
                printf -- "工作目錄:     $(pwd)\n" >> "$log_file"
                printf -- "原始指令:     %s\n" "$raw_cmd" >> "$log_file"
                printf -- "分配後指令:   %s\n" "$final_cmd" >> "$log_file"
                printf -- "Log 檔案路徑: %s\n" "$log_file" >> "$log_file"
                printf -- "------------------------------------------------\n\n" >> "$log_file"

                (
                    target_script=$(echo "$final_cmd" | grep -o '/[^ ]*\.py' | head -n 1)
                    target_dir=$( [ -n "$target_script" ] && dirname "$target_script" || echo "$(pwd)" )
                    cd "$target_dir" || exit
                    exec_cmd=$(echo "$final_cmd" | sed 's/python/python -u/')
                    nohup bash -c "$exec_cmd" >> "$log_file" 2>&1 &
                    disown
                ) 
                sleep 5
            fi
        done
    fi

    # 3. 狀態監控資料準備
    STATUS_LINE_1=""
    STATUS_LINE_2=""
    GPU_COUNT=0
    for gpu_idx in $(echo "${!GPU_MAP[@]}" | tr ' ' '\n' | sort -n); do
        pids_on_gpu=$(nvidia-smi -i $gpu_idx --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        used_vram=$(nvidia-smi -i $gpu_idx --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        total_vram=$(nvidia-smi -i $gpu_idx --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
        
        DISPLAY_TEXT=""
        if [ -n "$pids_on_gpu" ]; then
            IS_MY_TASK=false
            for pid in $pids_on_gpu; do
                if [ "$(ps -o user= -p "$pid" 2>/dev/null | xargs)" == "$USER" ]; then IS_MY_TASK=true; break; fi
            done

            if [ "$IS_MY_TASK" = true ]; then
                vram_pct=$(( used_vram * 100 / total_vram ))
                COLOR="\033[32m" # 綠色
                if [ "$vram_pct" -ge 90 ]; then COLOR="\033[31m"; elif [ "$vram_pct" -ge 50 ]; then COLOR="\033[33m"; fi
                DISPLAY_TEXT="${COLOR}G$gpu_idx: 執行中 (MEM: ${vram_pct}%)\033[0m"
            else
                DISPLAY_TEXT="G$gpu_idx: OCC"
            fi
        else
            DISPLAY_TEXT="G$gpu_idx: --"
        fi

        if [ $GPU_COUNT -lt 4 ]; then STATUS_LINE_1+="| $DISPLAY_TEXT "; else STATUS_LINE_2+="| $DISPLAY_TEXT "; fi
        ((GPU_COUNT++))
    done
    
    # 4. 固定三行刷新邏輯 (含剩餘序列)
    QUEUE_COUNT=$(wc -l < "$QUEUE_FILE")
    if [ "$FIRST_RUN" = "true" ]; then
        echo -e "\n⏳ 監控狀態 (剩餘序列: $QUEUE_COUNT)"
        echo -e "   (0-3) $STATUS_LINE_1"
        echo -e "   (4-7) $STATUS_LINE_2"
        FIRST_RUN=false
    else
        # 向上移動三行並清除
        echo -ne "\033[3A"
        echo -e "\r\033[K⏳ 監控狀態 (剩餘序列: $QUEUE_COUNT)"
        echo -e "\r\033[K   (0-3) $STATUS_LINE_1"
        echo -e "\r\033[K   (4-7) $STATUS_LINE_2"
    fi
    
    sleep $CHECK_INTERVAL
done