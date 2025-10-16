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
