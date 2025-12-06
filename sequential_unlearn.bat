@echo off
setlocal EnableDelayedExpansion

REM ============================================
REM FairVUE 参数搜索（Windows Batch 版）
REM 每个客户端使用独立的 exp_name 和 retrain 路径
REM ============================================

REM 固定参数
SET "DATASET=tinyimagenet"
SET "MODEL=resnet18"
SET "OPTIMIZER=sgd"
SET "TOTAL_CLIENTS=20"
SET "ITERS=200"
SET "DEVICE=cuda"
SET "LR=0.1"
SET "EPOCHS=1"
SET "SEED=42"
SET "FULL_TRAIN_DIR=./experiments/tinyimagenet_resnet18_alpha0.5/full_training"
SET "DISTRIBUTION=dirichlet"
SET "BASE_EXP_NAME=tinyimagenet_resnet18_alpha0.5"

REM 超参数取值范围 (在 Batch 中使用空格分隔列表)
SET "FAIR_RANK_LIST=200"
SET "FAIR_TAU_MODES=median"
SET "FAIR_FISHER_BATCHES=10"
SET "FAIR_ERASE_SCALES=0.02"
SET "FORGET_CLIENTS=0"

REM 循环执行实验
FOR %%C IN (%FORGET_CLIENTS%) DO (
  
  REM 动态设置路径变量
  SET "EXP_NAME=%BASE_EXP_NAME%_client%%C"
  SET "RETRAIN_MODEL_PATH=./experiments/!EXP_NAME!/retraining"

  FOR %%K IN (%FAIR_RANK_LIST%) DO (
    FOR %%T IN (%FAIR_TAU_MODES%) DO (
      FOR %%B IN (%FAIR_FISHER_BATCHES%) DO (
        FOR %%S IN (%FAIR_ERASE_SCALES%) DO (

          echo ==============================
          echo 🚀 正在执行：client=%%C, k=%%K, tau=%%T, fb=%%B, es=%%S
          echo ==============================

          python main.py ^
            --exp_name !EXP_NAME! ^
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
            --fair_rank_k %%K ^
            --fair_tau_mode %%T ^
            --fair_fisher_batches %%B ^
            --fair_erase_scale %%S ^
            --fair_vue_debug true ^
            --skip_training false ^
            --skip_retraining false ^
            --full_training_dir %FULL_TRAIN_DIR% ^
            --retraining_dir !RETRAIN_MODEL_PATH! ^
            --apply_membership_inference true ^
            --mia_verbose false ^
            --mia_scope all ^
            --fair_auto_tune_all false ^
            --fair_auto_erase false ^
            --fe_max_step_ratio 0.26 ^
            --ratio_cutoff 0.185 ^
            --dampening_constant 0.8 ^
            --dampening_upper_bound 0.98 ^
            --conda_lower_bound 0.711 ^
            --conda_eps 1e-6 ^
            --conda_weights_path ./experiments/tinyimagenet_resnet18_alpha0.5_client0/full_training ^
            --pga_unlearn_lr 0.0020

          echo ✅ 完成：client=%%C, k=%%K, tau=%%T, fb=%%B, es=%%S
          echo.
        )
      )
    )
  )
)

echo 🎯 所有实验完成！
pause