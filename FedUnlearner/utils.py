import torch
import copy
import os
from typing import Dict
from typeguard import typechecked

def average_weights(weights_path, device,
                    samples_per_client: dict[int, int] | None = None,
                    weight_exponent: float = 0.5,
                    min_weight_floor: float = 0.05):
    """
    聚合客户端权重。
    - 若 samples_per_client=None -> 退化为等权平均（保持兼容）
    - 若提供 samples：按 n_k^beta 做幂次加权（beta=weight_exponent），并给每个客户端一个最小权重下限
      min_weight_floor 表示相对于“均匀权重(1/M)”的下限比例。例如 0.05 表示每个客户端至少拿到均匀权重的 5%。
    """
    import os, re, torch
    w_files = [f for f in os.listdir(weights_path)
               if f.startswith("client_") and f.endswith(".pth")]
    assert len(w_files) > 0, f"Weights path is empty: {weights_path}"

    # 读取 state_dict 与客户端 id
    states, ids = [], []
    for f in w_files:
        m = re.search(r"client_(\d+)\.pth", f)
        if not m:
            continue
        cid = int(m.group(1))
        states.append(torch.load(os.path.join(weights_path, f), map_location=device, weights_only=True))
        ids.append(cid)

    # 初始化累加器
    w_accum = {k: torch.zeros_like(v, device=device) for k, v in states[0].items()}

    # 等权平均（兼容旧调用）
    if not samples_per_client:
        for st in states:
            for k in w_accum.keys():
                w_accum[k] += st[k]
        for k in w_accum.keys():
            w_accum[k] = w_accum[k] / float(len(states))
        return w_accum

    # 幂次加权 + 下限
    import numpy as np
    counts = np.array([float(samples_per_client[cid]) for cid in ids], dtype=float)
    # 幂次加权：beta=0 等权，beta=1 标准 FedAvg
    weights = counts ** float(weight_exponent)

    # 归一化
    if weights.sum() == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    # 下限：每个客户端至少获得 (min_weight_floor * 均匀权重) 的比例
    if min_weight_floor is not None and min_weight_floor > 0:
        floor = (min_weight_floor / len(weights))
        weights = np.maximum(weights, floor)
        weights = weights / weights.sum()

    # 加权聚合
    for st, w in zip(states, weights):
        for k in w_accum.keys():
                # 只对浮点型参数进行加权平均
            if torch.is_floating_point(st[k]):
                w_accum[k] += st[k] * float(w)
            else:
                # 对整型参数（如 num_batches_tracked）直接取第一个客户端的值
                w_accum[k] = st[k]
    return w_accum



def print_exp_details(args):
    print('\nExperimental details:')
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.num_training_iterations}\n")

    print('    Federated parameters:')
    if args.client_data_distribution == "iid":
        print('    IID')
    else:
        print('    Non-IID')
    print(
        f"    Number of participating users  : {args.num_participating_clients}")
    print(f"    Batch size   : {args.batch_size}")
    print(f"    Local Epochs       : {args.num_local_epochs}\n")

def get_labels_from_dataset(dataset):
    """
    Get labels from a torch Dataset or Subset without iterating through the whole dataset.
    Args:
        dataset: torch.utils.data.Dataset or torch.utils.data.Subset
    Returns:
        labels: list of labels
    """
    if isinstance(dataset, torch.utils.data.Subset):
        # Access the underlying dataset and indices
        subset_indices = dataset.indices
        if hasattr(dataset.dataset, 'targets'):
            labels = [dataset.dataset.targets[i] for i in subset_indices]
            # Convert labels to integers if they are tensors
            return [label.item() if isinstance(label, torch.Tensor) else label for label in labels]
        else:
            raise AttributeError(
                "The underlying dataset does not have a 'targets' attribute.")
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
        # Convert labels to integers if they are tensors
        return [label.item() if isinstance(label, torch.Tensor) else label for label in labels]
    else:
        raise AttributeError(
            "The dataset does not have a 'targets' attribute.")


@typechecked
def print_clientwise_class_distribution(clientwise_dataset: Dict[int, torch.utils.data.Dataset],
                                        num_classes: int, num_workers: int = 0):
    def create_labels(classes):
        labels = dict()
        for i in range(classes):
            labels[i] = 0
        return labels

    for client_id, client_dataset in clientwise_dataset.items():
        labels = create_labels(num_classes)
        data_labels = get_labels_from_dataset(client_dataset)
        for label in data_labels:
            labels[label] += 1
        print(f"Data distribution for client : {client_id} :::: { labels}")


# =====================
# ==== 新增：指标工具 ====
# =====================
def eval_ce_loss(model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 device: str = "cpu") -> float:
    """
    在给定 dataloader 上计算平均交叉熵（用于“目标客户端预测损失”指标）。
    仅前向推理，不修改梯度。
    """
    import torch
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    losses, cnt = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            losses += criterion(out, y).item() * y.size(0)
            cnt += y.size(0)
    return (losses / cnt) if cnt > 0 else 0.0

def eval_retain_acc(model: torch.nn.Module,
                    client_loaders: dict,
                    forget_clients: list,
                    device: str = "cpu") -> float:
    """
    计算所有'保留客户端'（非遗忘目标）训练集上的整体准确率。
    用于评估保留集精度 (Retain Accuracy / Fidelity)。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for cid, loader in client_loaders.items():
            # 跳过遗忘客户端
            if cid in forget_clients:
                continue
            # 累积保留客户端的预测结果
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
    return (correct / total) if total > 0 else 0.0


def _flatten_model_params(model: torch.nn.Module) -> torch.Tensor:
    """
    将模型所有可训练参数展平成单一向量（用于参数夹角）。
    """
    vecs = []
    for p in model.parameters():
        if p.requires_grad:
            vecs.append(p.detach().flatten().float().cpu())
    if len(vecs) == 0:
        import torch
        return torch.zeros(1)
    import torch
    return torch.cat(vecs, dim=0)


def cosine_angle_between_models(a: torch.nn.Module, b: torch.nn.Module) -> float:
    """
    计算两个模型在“全参数向量”上的夹角（单位：度）。
    若存在 0 向量，返回 0.0 兜底。
    """
    import torch, math
    va = _flatten_model_params(a)
    vb = _flatten_model_params(b)
    na = torch.norm(va)
    nb = torch.norm(vb)
    if na.item() == 0.0 or nb.item() == 0.0:
        return 0.0
    cos = torch.clamp(torch.dot(va, vb) / (na * nb), -1.0, 1.0).item()
    return math.degrees(math.acos(cos))


def print_forgetting_metrics(method_name: str,
                             test_acc: float | None,
                             retain_acc: float | None,
                             target_acc: float | None,
                             target_loss: float | None,
                             speedup_x: float | None,
                             angle_deg: float | None,
                             mia_result: dict | None):
    """
    统一格式化打印六项指标；None 会打印为 'NA'。
    """
    def fmt(x, p=4):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "NA"

    # ---- 统一解析 MIA 结果（兼容 dict / tuple / list / np / torch / str）----
    def _parse_mia(res):
        """尽量从多种返回类型中抽取 (precision, recall, f1) 三个标量."""
        prec = rec = f1 = None
        import re as _re
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            import torch as _torch
        except Exception:
            _torch = None

        def _to_float(x):
            # 标量化：tensor/ndarray -> float；列表取均值（兜底）；否则尝试 float()
            try:
                if _torch is not None and isinstance(x, _torch.Tensor):
                    return float(x.mean().item())
                if _np is not None and isinstance(x, _np.ndarray):
                    return float(x.mean())
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    return float(sum(map(_to_float, x)) / len(x))
                return float(x)
            except Exception:
                return None

        # 1) dict：先精确键名，再模糊包含（兼容 mia_attacker_precision 等）
        if isinstance(res, dict):
            lower_map = {str(k).lower().replace("-", "_"): k for k in res.keys()}
            def _get_exact(*cands):
                for c in cands:
                    lk = c.lower().replace("-", "_")
                    if lk in lower_map:
                        return res[lower_map[lk]]
                return None
            # 精确尝试
            prec = _get_exact("precision", "precision_score", "p")
            rec  = _get_exact("recall",    "recall_score",    "r")
            f1   = _get_exact("f1", "f1_score", "f1score")
            # 模糊包含（如 mia_attacker_precision / avg_precision 也可命中）
            if prec is None:
                for lk, ok in lower_map.items():
                    if "precision" in lk:
                        prec = res[ok]; break
            if rec is None:
                for lk, ok in lower_map.items():
                    if "recall" in lk:
                        rec = res[ok]; break
            if f1 is None:
                # 优先匹配 *_f1 / (^|_)f1($|_) / f1_score；若都没有，再兜底任意含 'f1'
                preferred = None
                for lk, ok in lower_map.items():
                    if lk.endswith("_f1") or lk == "f1" or lk.startswith("f1") or "f1_score" in lk:
                        preferred = res[ok]; break
                if preferred is None:
                    for lk, ok in lower_map.items():
                        if "f1" in lk:   # 兜底：包括 mia_attacker_f1 等
                            preferred = res[ok]; break
                f1 = preferred
            return _to_float(prec), _to_float(rec), _to_float(f1)

        # 2) tuple / list / numpy / torch: 按 (P,R,F1) 读取
        if isinstance(res, (list, tuple)):
            arr = list(res)
            if len(arr) >= 1: prec = arr[0]
            if len(arr) >= 2: rec  = arr[1]
            if len(arr) >= 3: f1   = arr[2]
            return _to_float(prec), _to_float(rec), _to_float(f1)
        try:
            import numpy as _np
            if isinstance(res, _np.ndarray):
                arr = res.flatten().tolist()
                return (_to_float(arr[0]) if len(arr)>0 else None,
                        _to_float(arr[1]) if len(arr)>1 else None,
                        _to_float(arr[2]) if len(arr)>2 else None)
        except Exception:
            pass
        try:
            import torch as _torch
            if isinstance(res, _torch.Tensor):
                arr = res.flatten().tolist()
                return (_to_float(arr[0]) if len(arr)>0 else None,
                        _to_float(arr[1]) if len(arr)>1 else None,
                        _to_float(arr[2]) if len(arr)>2 else None)
        except Exception:
            pass

        # 3) 字符串：尝试抓取数字
        if isinstance(res, str):
            m = _re.findall(r"([0-9]*\.?[0-9]+)", res)
            if m:
                vals = [float(x) for x in m[:3]]
                while len(vals) < 3: vals.append(None)
                return vals[0], vals[1], vals[2]

        return None, None, None

    p_val, r_val, f1_val = _parse_mia(mia_result)
    mia_str = f"P={fmt(p_val,3)}, R={fmt(r_val,3)}, F1={fmt(f1_val,3)}" if any(v is not None for v in (p_val, r_val, f1_val)) else "NA"


    print(f"[指标/{method_name}] "
        f"测试集准确率={fmt(test_acc,4)}  "
        f"保留集准确率={fmt(retain_acc,4)}  "
        f"遗忘客户端准确率={fmt(target_acc,4)}  "
        f"遗忘客户端平均交叉熵={fmt(target_loss,4)}  "
        f"加速比={fmt(speedup_x,2)}×  "
        f"参数夹角={fmt(angle_deg,2)}°  "
        f"成员推断({mia_str})")