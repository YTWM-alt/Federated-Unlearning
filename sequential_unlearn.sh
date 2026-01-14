#!/bin/bash
# ============================================
# FairVUE 参数搜索（去除日志写入，仅控制台输出）
# 每个客户端使用独立的 exp_name 和 retrain 路径
# ============================================

# 固定参数
DATASET="cifar10"
MODEL="resnet18"
OPTIMIZER="sgd"
TOTAL_CLIENTS=20
ITERS=150
DEVICE="cuda"
LR=0.01
EPOCHS=1
SEED=42
FULL_TRAIN_DIR="./experiments/cifar10_resnet18_exclusive/full_training"
DISTRIBUTION="exclusive"
BASE_EXP_NAME="cifar10_resnet18_exclusive"

# 超参数取值范围
FAIR_RANK_LIST=(106)
FAIR_TAU_MODES=("median")
FAIR_FISHER_BATCHES=(10)
FAIR_ERASE_SCALES=(0.2)
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
            --pretrained True \
            --device $DEVICE \
            --num_workers 1 \
            --lr $LR \
            --client_data_distribution $DISTRIBUTION \
            --num_participating_clients -1 \
            --seed $SEED \
            --num_local_epochs $EPOCHS \
            --baselines fair_vue\
            --fair_rank_k $RANK_K \
            --fair_tau_mode $TAU_MODE \
            --fair_fisher_batches $FISHER_B \
            --fair_erase_scale $ERASE_S \
            --fair_vue_debug true \
            --skip_training true \
            --skip_retraining true \
            --execution_stage all \
            --full_training_dir $FULL_TRAIN_DIR \
            --retraining_dir $RETRAIN_MODEL_PATH \
            --apply_membership_inference true \
            --mia_verbose false \
            --fair_auto_tune_all false \
            --fair_auto_erase false \
            --fe_max_step_ratio 0.001 \
            --ratio_cutoff 0.185 \
            --dampening_constant 0.8 \
            --dampening_upper_bound 0.98 \
            --conda_lower_bound 0.711 \
            --conda_eps 1e-6 \
            --conda_weights_path ./experiments/cifar10_resnet18_exclusive_client0/full_training \
            --pga_unlearn_lr 0.008 \


          echo "✅ 完成：client=${CID}, k=${RANK_K}, tau=${TAU_MODE}, fb=${FISHER_B}, es=${ERASE_S}"
          echo
        done
      done
    done
  done
done

echo "🎯 所有实验完成！"
