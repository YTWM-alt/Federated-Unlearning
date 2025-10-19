#!/bin/bash
# 顺序遗忘多个客户端（严格保留用户命令参数）

# 依次遗忘的客户端编号列表
FORGET_LIST=(5)

for CID in "${FORGET_LIST[@]}"; do
  echo "=============================="
  echo " 正在遗忘客户端 ${CID} ..."
  echo "=============================="

  python3 main.py \
    --exp_name cifar10_allcnn \
    --dataset cifar10 \
    --optimizer sgd \
    --total_num_clients 10 \
    --num_training_iterations 40 \
    --forget_clients $CID \
    --model allcnn \
    --device cuda \
    --num_workers 0 \
    --lr 0.03 \
    --client_data_distribution dirichlet \
    --num_participating_clients -1 \
    --seed 42 \
    --num_local_epochs 1 \
    --baselines fair_vue \
    --fair_rank_k 12 \
    --fair_tau_mode median \
    --fair_fisher_batches 10 \
    --fair_erase_scale 0.2 \
    --fair_vue_debug \
    --unlearn_only \
    --skip_retraining \
    --full_training_dir ./experiments/cifar10_allcnn/full_training \
    --heal \
    --heal_teacher pre \
    --heal_alpha 0.05

  echo "✅ 客户端 ${CID} 遗忘完成。"
  echo
done

echo "🎯 全部顺序遗忘任务完成！"
