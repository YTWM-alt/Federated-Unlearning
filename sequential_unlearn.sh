#!/bin/bash
# ============================================
# FairVUE 参数搜索（去除日志写入，仅控制台输出）
# 每个客户端使用独立的 exp_name 和 retrain 路径
# ============================================

# 固定参数
DATASET="cifar100"
MODEL="resnet18"
OPTIMIZER="sgd"
TOTAL_CLIENTS=20
ITERS=200
DEVICE="cuda"
LR=0.1
EPOCHS=1
SEED=42
FULL_TRAIN_DIR="./experiments/cifar100_resnet18/full_training"
DISTRIBUTION="dirichlet"
BASE_EXP_NAME="cifar100_resnet18"

# 超参数取值范围
FAIR_RANK_LIST=(16)
FAIR_TAU_MODES=("mean")
FAIR_FISHER_BATCHES=(10)
FAIR_ERASE_SCALES=(0.15)
FORGET_CLIENTS=(0)

# 循环执行实验
for CID in "${FORGET_CLIENTS[@]}"; do
  EXP_NAME="${BASE_EXP_NAME}_client${CID}"                           
  RETRAIN_MODEL_PATH="./experiments/${EXP_NAME}/retraining"          
  
  for RANK_K in "${FAIR_RANK_LIST[@]}"; do
    for TAU_MODE in "${FAIR_TAU_MODES[@]}"; do
      for FISHER_B in "${FAIR_FISHER_BATCHES[@]}"; do
        for ERASE_S in "${FAIR_ERASE_SCALES[@]}"; do

          echo "=============================="
          echo "🚀 正在执行：client=${CID}, k=${RANK_K}, tau=${TAU_MODE}, fb=${FISHER_B}, es=${ERASE_S}"
          echo "=============================="

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
            --baselines fast_fu \
            --fair_rank_k $RANK_K \
            --fair_tau_mode $TAU_MODE \
            --fair_fisher_batches $FISHER_B \
            --fair_erase_scale $ERASE_S \
            --fair_vue_debug true \
            --skip_training true \
            --skip_retraining true \
            --full_training_dir $FULL_TRAIN_DIR \
            --retraining_dir $RETRAIN_MODEL_PATH \
            --apply_membership_inference true \
            --mia_verbose true \
            --mia_scope all \
            --fair_auto_tune_all true \
            --fair_auto_erase true \


          echo "✅ 完成：client=${CID}, k=${RANK_K}, tau=${TAU_MODE}, fb=${FISHER_B}, es=${ERASE_S}"
          echo
        done
      done
    done
  done
done

echo "🎯 所有实验完成！"
