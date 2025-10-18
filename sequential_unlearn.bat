@echo off
setlocal enabledelayedexpansion

set FORGET_LIST=0 1 2 3 4 5 6 7 8 9

for %%C in (%FORGET_LIST%) do (
  echo ==============================
  echo 正在遗忘客户端 %%C ...
  echo ==============================

  python main.py ^
    --exp_name cifar10_allcnn ^
    --dataset cifar10 ^
    --optimizer sgd ^
    --total_num_clients 10 ^
    --num_training_iterations 40 ^
    --forget_clients %%C ^
    --model allcnn ^
    --device cuda ^
    --num_workers 0 ^
    --lr 0.03 ^
    --client_data_distribution dirichlet ^
    --num_participating_clients -1 ^
    --seed 42 ^
    --num_local_epochs 1 ^
    --fair_rank_k 12 ^
    --fair_tau_mode median ^
    --fair_fisher_batches 10 ^
    --fair_erase_scale 0.2 ^
    --fair_vue_debug ^
    --unlearn_only ^
    --full_training_dir ./experiments/cifar10_allcnn/full_training

  echo ✅ 客户端 %%C 遗忘完成。
  echo.
)

echo 🎯 全部顺序遗忘任务完成！
pause
