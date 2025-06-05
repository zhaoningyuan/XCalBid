#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=0

campaigns=("1458" "2259" "2261" "2821" "2997" "3358" "3386" "3427" "3476")
barget_paras=("2" "4" "8" "16")
reward_type="nop"
pids=()
mkdir -p offline_datasets/

cleanup() {
    echo "Received signal, terminating all processes..."
    for pid in "${pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill -TERM $pid
            echo "Killed process $pid"
        fi
    done
    exit 1
}

trap cleanup SIGINT SIGTERM

for campaign_id in "${campaigns[@]}"; do
    for barget_para in "${barget_paras[@]}"; do
        dataset_path="offline_datasets/c${campaign_id}_b${barget_para}_r${reward_type}.npz"
        python sim-offline-data.py --campaign_id=$campaign_id --budget_para=$barget_para --offline_data_path $dataset_path --reward_type=$reward_type &
        pids+=($!)
        echo "Started process $! for campaign $campaign_id with budget_para $barget_para"
    done
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo "All processes completed."