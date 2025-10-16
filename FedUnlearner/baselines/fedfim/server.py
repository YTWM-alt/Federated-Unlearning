from collections import OrderedDict
from typing import List, Tuple, Dict


def aggregate_fim(
    fims: List["OrderedDict[str, torch.Tensor]"],
    counts: List[int],
    eps: float = 1e-8
) -> "OrderedDict[str, torch.Tensor]":
    """
    聚合多个客户端的 FIM，对齐 state_dict 键，按样本数加权平均。

    参数:
      fims:   客户端返回的 FIM 字典列表
      counts: 每个客户端的样本数 (n_k)
      eps:    防止除零

    返回:
      一个 OrderedDict, 与模型 state_dict 可训练参数对齐
    """
    assert len(fims) == len(counts), "fims 与 counts 长度不一致"

    total = float(sum(counts)) + eps
    keys = list(fims[0].keys())

    # 初始化为 0
    agg = OrderedDict((k, fims[0][k].new_zeros(fims[0][k].shape)) for k in keys)

    # 逐客户端累加
    for fim, n in zip(fims, counts):
        w = float(n) / total
        for k in keys:
            agg[k] += w * fim[k]

    return agg


def partition_and_aggregate(
    all_fims: Dict[int, Tuple["OrderedDict[str, torch.Tensor]", int]],
    forget_clients: List[int]
) -> Tuple["OrderedDict[str, torch.Tensor]", "OrderedDict[str, torch.Tensor]"]:
    """
    根据客户端编号划分为保留集合 R 和遗忘集合 U，然后分别聚合。

    参数:
      all_fims: {client_id: (fim_dict, n_k)} 的字典
      forget_clients: 要遗忘的客户端编号列表

    返回:
      (F_r, F_f)
    """
    retain_fims, retain_counts = [], []
    forget_fims, forget_counts = [], []

    for cid, (fim, n_k) in all_fims.items():
        if cid in forget_clients:
            forget_fims.append(fim)
            forget_counts.append(n_k)
        else:
            retain_fims.append(fim)
            retain_counts.append(n_k)

    F_r = aggregate_fim(retain_fims, retain_counts) if retain_fims else None
    F_f = aggregate_fim(forget_fims, forget_counts) if forget_fims else None
    return F_r, F_f
