import copy
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier

"""
Membership Inference Attack from
https://www.chenwang.net.cn/publications/FedEraser-IWQoS21.pdf
Code: https://www.dropbox.com/s/1lhx962axovbbom/FedEraser-Code.zip?dl=0&e=5&file_subpath=%2FFedEraser-Code
"""

@torch.no_grad
def train_attack_model(shadow_global_model, shadow_client_loaders, shadow_test_loader, dataset, device):
    shadow_model = shadow_global_model
    n_class_dict = dict()
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10
    n_class_dict['cifar100'] = 100

    N_class = n_class_dict[dataset]

    device = torch.device(device)
    shadow_model.to(device)

    shadow_model.eval()
    # ====== MIA 训练阶段：输入规模与设备 ======
    try:
        _mem_total = sum(len(dl.dataset) for dl in shadow_client_loaders.values())
        _nonmem_total = len(getattr(shadow_test_loader, "dataset", []))
        print(f"[MIA-TRAIN] dataset={dataset} device={device} mem_total={_mem_total} nonmem_total={_nonmem_total} "
              f"shadow_nonmem_ds_id={id(getattr(shadow_test_loader, 'dataset', object()))}")
    except Exception as _e:
        print(f"[MIA-TRAIN][WARN] failed to read dataset sizes: {_e}")
    ####
    pred_4_mem_list = []
    for _, dataloader in shadow_client_loaders.items():
        for data, target in dataloader:
            data = data.to(device)
            out = shadow_model(data)
            out = softmax(out, dim=1)
            pred_4_mem_list.append(out.cpu().detach().numpy())

    pred_4_mem = np.concatenate(pred_4_mem_list, axis=0)

    ####
    pred_4_nonmem_list = []
    for data, target in shadow_test_loader:
        data = data.to(device)
        out = shadow_model(data)
        out = softmax(out, dim=1)
        pred_4_nonmem_list.append(out.cpu().detach().numpy())
    pred_4_nonmem = np.concatenate(pred_4_nonmem_list, axis=0)

    # —— 特征统计：max prob / entropy（成员 vs 非成员）——
    def _prob_stats(arr):
        pmax = arr.max(axis=1)
        ent  = -(arr * np.log(arr + 1e-12)).sum(axis=1)
        return float(pmax.mean()), float(pmax.std()), float(ent.mean()), float(ent.std())
    _m_mp_mu, _m_mp_sd, _m_ent_mu, _m_ent_sd = _prob_stats(pred_4_mem)
    _n_mp_mu, _n_mp_sd, _n_ent_mu, _n_ent_sd = _prob_stats(pred_4_nonmem)
    print(f"[MIA-TRAIN] mem shape={pred_4_mem.shape} maxP μ={_m_mp_mu:.3f}±{_m_mp_sd:.3f} "
          f"entropy μ={_m_ent_mu:.3f}±{_m_ent_sd:.3f}")
    print(f"[MIA-TRAIN] nonmem shape={pred_4_nonmem.shape} maxP μ={_n_mp_mu:.3f}±{_n_mp_sd:.3f} "
          f"entropy μ={_n_ent_mu:.3f}±{_n_ent_sd:.3f}")


    n = min(pred_4_mem.shape[0], pred_4_nonmem.shape[0])
    idx_mem    = np.random.RandomState(0).choice(pred_4_mem.shape[0],    size=n, replace=False)
    idx_nonmem = np.random.RandomState(1).choice(pred_4_nonmem.shape[0], size=n, replace=False)
    att_X = np.vstack((pred_4_mem[idx_mem], pred_4_nonmem[idx_nonmem]))
    att_y = np.hstack((np.ones(n), np.zeros(n))).astype(np.int16)
    print(f"[MIA-TRAIN] downsample to 1:1 -> n_each={n}, att_X={att_X.shape}")

    att_X.sort(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        att_X, att_y, test_size=0.1)

    # For possible division by zero error
    scale_pos_weight = 1.0  # 已经 1:1，无需额外权重
    attacker = XGBClassifier(n_estimators=200,
                             n_jobs=-1,
                             max_depth=8,
                             objective='binary:logistic',
                             booster="gbtree",
                             # learning_rate=None,
                             # tree_method = 'gpu_hist',
                             scale_pos_weight=scale_pos_weight,
                             random_state=0
                             )

    print(f"[MIA-TRAIN] XGB params: n_estimators={attacker.n_estimators} max_depth={attacker.max_depth} "
          f"scale_pos_weight={attacker.scale_pos_weight} random_state=0")
    attacker.fit(X_train, y_train)
    # —— holdout 验证，确认攻击器不是“全靠阈值” —— 
    try:
        from sklearn.metrics import roc_auc_score
        _pred = attacker.predict(X_test)
        _proba = attacker.predict_proba(X_test)[:,1]
        _acc = accuracy_score(y_test, _pred)
        _f1  = f1_score(y_test, _pred)
        _auc = roc_auc_score(y_test, _proba)
        print(f"[MIA-TRAIN] holdout acc={_acc:.3f} f1={_f1:.3f} auc={_auc:.3f} (X_test={X_test.shape[0]})")
    except Exception as _e:
        print(f"[MIA-TRAIN][WARN] holdout eval failed: {_e}")
    return attacker


def evaluate_mia_attack(target_model: torch.nn.Module,
                        attack_model: torch.nn.Module,
                        client_loaders,
                        test_loader,  # 保持兼容
                        dataset: str,
                        forget_client_idx: int,
                        device: str,
                        eval_nonmem_loader=None):
    results = {}

    n_class_dict = dict()
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10
    n_class_dict['cifar100'] = 100

    N_class = n_class_dict[dataset]

    target_model.to(device)
    target_model.eval()
    # —— 目标模型“指纹”，确保真的换了模型 —— 
    try:
        _first = next(iter(target_model.parameters()))
        print(f"[MIA-EVAL] target_model device={device} first_param μ={float(_first.data.mean()):.4e} "
              f"σ={float(_first.data.std()):.4e} numel={_first.data.numel()}")
    except Exception:
        pass

    # The predictive output of forgotten user data after passing through the target model.
    unlearn_X_list = []
    with torch.no_grad():
        for data, target in client_loaders[forget_client_idx]:
            data = data.to(device)
            out = target_model(data)
            out = softmax(out, dim=1)
            unlearn_X_list.append(out.cpu().detach().numpy())


    unlearn_X = np.concatenate(unlearn_X_list, axis=0)
    unlearn_X.sort(axis=1)
    unlearn_y = np.ones(unlearn_X.shape[0])
    unlearn_y = unlearn_y.astype(np.int16)

    N_unlearn_sample = len(unlearn_y)
    # —— 成员 softmax 统计 —— 
    try:
        _mp_mu, _mp_sd, _ent_mu, _ent_sd = _prob_stats(unlearn_X)
    except NameError:
        def _prob_stats(arr):
            pmax = arr.max(axis=1)
            ent  = -(arr * np.log(arr + 1e-12)).sum(axis=1)
            return float(pmax.mean()), float(pmax.std()), float(ent.mean()), float(ent.std())
        _mp_mu, _mp_sd, _ent_mu, _ent_sd = _prob_stats(unlearn_X)
    print(f"[MIA-EVAL] member N={N_unlearn_sample} softmax maxP μ={_mp_mu:.3f}±{_mp_sd:.3f} "
          f"entropy μ={_ent_mu:.3f}±{_ent_sd:.3f}")

    # 评估用的“非成员”应来自与 shadow 训练不同的一部分测试数据，避免泄漏
    if eval_nonmem_loader is not None:
        shuffled_test_loader = eval_nonmem_loader
    else:
        # 退路：保持旧行为（但存在泄漏风险）
        shuffled_test_loader = DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, shuffle=True)
    try:
        _ds = getattr(shuffled_test_loader, "dataset", None)
        _idx = getattr(_ds, "indices", None)
        _head = list(_idx[:5]) if _idx is not None else "NA"
        _tail = list(_idx[-5:]) if _idx is not None else "NA"
        print(f"[MIA-EVAL] nonmember ds id={id(_ds)} len={len(_ds) if _ds is not None else 'NA'} "
              f"indices_head={_head} ... tail={_tail}")
    except Exception:
        pass


    # Test data, predictive output obtained after passing the target model
    test_X_list = []
    total_samples_collected = 0
    with torch.no_grad():
        for data, target in shuffled_test_loader:
            data = data.to(device)
            out = target_model(data)
            out = softmax(out, dim=1)
            test_X_list.append(out.cpu().detach().numpy())
            total_samples_collected += out.shape[0]

            if total_samples_collected > N_unlearn_sample:
                break
    

    test_X = np.concatenate(test_X_list, axis=0)[:N_unlearn_sample]
    test_X.sort(axis=1)
    test_y = np.zeros(test_X.shape[0])
    test_y = test_y.astype(np.int16)
    _n_nonmem = test_X.shape[0]
    _n_mem    = unlearn_X.shape[0]
    _n_tot    = _n_nonmem + _n_mem
    _mp_mu_n, _mp_sd_n, _ent_mu_n, _ent_sd_n = _prob_stats(test_X)
    print(f"[MIA-EVAL] nonmember N={_n_nonmem} softmax maxP μ={_mp_mu_n:.3f}±{_mp_sd_n:.3f} "
          f"entropy μ={_ent_mu_n:.3f}±{_ent_sd_n:.3f}")

    # The data of the forgotten user passed through the output of the target model, and the data of the test set passed through the output of the target model were spliced together
    # The balanced data set that forms the 50% train 50% test.
    combined_X = np.vstack((unlearn_X, test_X))
    combined_Y = np.hstack((unlearn_y, test_y))

    pred_Y = attack_model.predict(combined_X)
    pred_proba_Y = attack_model.predict_proba(combined_X)[:, 1]

    # --- 诊断：阈值扫描，看看 F1 是否被 0.5 卡住 ---
    try:
        import numpy as _np
        from sklearn.metrics import f1_score as _f1
        ths = _np.linspace(0.1, 0.9, 9)
        f1_scan = [ _f1(combined_Y, (pred_proba_Y>=t).astype(int)) for t in ths ]
        best_t = float(ths[int(_np.argmax(f1_scan))])
        best_f1 = float(_np.max(f1_scan))
        print(f"[MIA-DBG] F1@0.5={_f1(combined_Y, (pred_proba_Y>=0.5).astype(int)):.3f}; "
            f"best_F1={best_f1:.3f} @ thr={best_t:.2f}")
    except Exception:
        pass

    # —— AUC 与混淆矩阵（0.5 阈值）——
    try:
        from sklearn.metrics import roc_auc_score, confusion_matrix
        _auc = roc_auc_score(combined_Y, pred_proba_Y)
        _cm  = confusion_matrix(combined_Y, pred_Y)
        if _cm.size == 4:
            tn, fp, fn, tp = _cm.ravel()
            print(f"[MIA-EVAL] AUC={_auc:.3f}  CM@0.5: tn={tn} fp={fp} fn={fn} tp={tp}  "
                  f"N={_n_tot} (mem={_n_mem}, nonmem={_n_nonmem})")
        else:
            print(f"[MIA-EVAL] AUC={_auc:.3f}  CM shape={_cm.shape}")
        # 成员与非成员分开看预测概率
        _proba_mem = attack_model.predict_proba(unlearn_X)[:,1]
        _proba_non = attack_model.predict_proba(test_X)[:,1]
        import numpy as _np
        def _q(a): 
            return tuple(float(x) for x in _np.quantile(a, [0.1,0.5,0.9]))
        print(f"[MIA-EVAL] proba(mem) q10/q50/q90={_q(_proba_mem)}  proba(non) q10/q50/q90={_q(_proba_non)}")
    except Exception as _e:
        print(f"[MIA-EVAL][WARN] AUC/CM diagnostics failed: {_e}")


    accuracy = accuracy_score(combined_Y, pred_Y)
    precision = precision_score(combined_Y, pred_Y, pos_label=1)
    recall = recall_score(combined_Y, pred_Y, pos_label=1)
    f1 = f1_score(combined_Y, pred_Y, pos_label=1)

    results['mia_attacker_accuracy'] = accuracy
    results['mia_attacker_precision'] = precision
    results['mia_attacker_recall'] = recall
    results['mia_attacker_f1'] = f1
    results['mia_attacker_predictions'] = pred_Y.tolist()
    results['mia_attacker_probabilities'] = pred_proba_Y.tolist()

    return results
