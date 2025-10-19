@echo off
REM ============================================
REM FairVUE 参数搜索（去除日志写入，仅控制台输出）
REM 保持 --exp_name 不变
REM ============================================

REM 固定参数
set DATASET=cifar10
set MODEL=allcnn
set OPTIMIZER=sgd
set TOTAL_CLIENTS=10
set ITERS=40
set DEVICE=cuda
set LR=0.03
set EPOCHS=1
set SEED=42
set FULL_TRAIN_DIR=./experiments/cifar10_allcnn/full_training
set DISTRIBUTION=dirichlet
set EXP_NAME=cifar10_allcnn

REM 超参数取值范围
set FAIR_RANK_LIST=8 12 16
set FAIR_TAU_MODES=mean median
set FAIR_FISHER_BATCHES=5 10 20
set FAIR_ERASE_SCALES=0.2 0.4 0.6
set FORGET_CLIENTS=5

REM 循环执行实验
for %%C in (%FORGET_CLIENTS%) do (
  for %%R in (%FAIR_RANK_LIST%) do (
    for %%T in (%FAIR_TAU_MODES%) do (
      for %%F in (%FAIR_FISHER_BATCHES%) do (
        for %%E in (%FAIR_ERASE_SCALES%) do (
          echo ==============================
          echo 🚀 正在执行：client=%%C, k=%%R, tau=%%T, fb=%%F, es=%%E
          echo ==============================

          python main.py ^
            --exp_name %EXP_NAME% ^
            --dataset %DATASET% ^
            --optimizer %OPTIMIZER% ^
            --total_num_clients %TOTAL_CLIENTS% ^
            --num_training_iterations %ITERS% ^
            --forget_clients %%C ^
            --model %MODEL% ^
            --device %DEVICE% ^
            --num_workers 0 ^
            --lr %LR% ^
            --client_data_distribution %DISTRIBUTION% ^
            --num_participating_clients -1 ^
            --seed %SEED% ^
            --num_local_epochs %EPOCHS% ^
            --baselines fair_vue ^
            --fair_rank_k %%R ^
            --fair_tau_mode %%T ^
            --fair_fisher_batches %%F ^
            --fair_erase_scale %%E ^
            --fair_vue_debug true ^
            --skip_retraining true ^
            --skip_training true ^
            --full_training_dir %FULL_TRAIN_DIR%

          echo ✅ 完成：client=%%C, k=%%R, tau=%%T, fb=%%F, es=%%E
          echo.
        )
      )
    )
  )
)

echo 🎯 所有实验完成！
pause
