#!/bin/bash
# ============================================
# FairVUE 参数搜索（去除日志写入，仅控制台输出）
# 保持 --exp_name 不变
# ============================================

# 固定参数
DATASET="cifar10"
MODEL="allcnn"
OPTIMIZER="sgd"
TOTAL_CLIENTS=10
ITERS=40
DEVICE="cuda"
LR=0.03
EPOCHS=1
SEED=42
FULL_TRAIN_DIR="./experiments/cifar10_allcnn/full_training"
DISTRIBUTION="dirichlet"
EXP_NAME="cifar10_allcnn"

# 超参数取值范围
FAIR_RANK_LIST=(8 12 16)
FAIR_TAU_MODES=("mean" "median")
FAIR_FISHER_BATCHES=(5 10 20)
FAIR_ERASE_SCALES=(0.2 0.4 0.6)
FORGET_CLIENTS=(5)

# 循环执行实验
for CID in "${FORGET_CLIENTS[@]}"; do
  for RANK_K in "${FAIR_RANK_LIST[@]}"; do
    for TAU_MODE in "${FAIR_TAU_MODES[@]}"; do
      for FISHER_B in "${FAIR_FISHER_BATCHES[@]}"; do
        for ERASE_S in "${FAIR_ERASE_SCALES[@]}"; do

          echo "=============================="
          echo "🚀 正在执行：client=${CID}, k=${RANK_K}, tau=${TAU_MODE}, fb=${FISHER_B}, es=${ERASE_S}"
          echo "=============================="

          # 执行命令（不保存日志，仅输出到控制台）
          python3 main.py \
            --exp_name $EXP_NAME \
            --dataset $DATASET \
            --optimizer $OPTIMIZER \
            --total_num_clients $TOTAL_CLIENTS \
            --num_training_iterations $ITERS \
            --forget_clients $CID \
            --model $MODEL \
            --device $DEVICE \
            --num_workers 0 \
            --lr $LR \
            --client_data_distribution $DISTRIBUTION \
            --num_participating_clients -1 \
            --seed $SEED \
            --num_local_epochs $EPOCHS \
            --baselines fair_vue \
            --fair_rank_k $RANK_K \
            --fair_tau_mode $TAU_MODE \
            --fair_fisher_batches $FISHER_B \
            --fair_erase_scale $ERASE_S \
            --fair_vue_debug true \
            --skip_retraining true \
            --skip_training true \
            --full_training_dir $FULL_TRAIN_DIR

          echo "✅ 完成：client=${CID}, k=${RANK_K}, tau=${TAU_MODE}, fb=${FISHER_B}, es=${ERASE_S}"
          echo
        done
      done
    done
  done
done

echo "🎯 所有实验完成！"
