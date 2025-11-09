import os
from collections import OrderedDict
from typing import List

import torch
from typeguard import typechecked


@typechecked
def _get_client_contribution(start_model: OrderedDict, weights_path: str):
    """
    与旧版 unlearn.py 中 get_client_contribution 等价：
    读取 iteration 目录下的 client_{id}.pth，与 start_model 做逐参数绝对差。
    """
    checkpoint_list = [
        f for f in os.listdir(weights_path)
        if f.startswith("client_") and f.endswith(".pth")
    ]
    checkpoint_ids = sorted(int(f[7:-4]) for f in checkpoint_list)

    client_wise_differences = {}
    for client_id in checkpoint_ids:
        client_weights = torch.load(
            os.path.join(weights_path, f"client_{client_id}.pth"),
            map_location="cpu", weights_only=True
        )
        difference = {}
        for param in start_model.keys():
            difference[param] = torch.abs(start_model[param] - client_weights[param])
        client_wise_differences[client_id] = difference
    return client_wise_differences


def _get_group_contribution(contributions: List[dict]):
    """
    与旧版等价：对一组客户端的贡献做逐参数平均。
    """
    if not contributions:
        return {}
    avg = {k: torch.zeros_like(contributions[0][k]) for k in contributions[0].keys()}
    for one in contributions:
        for k in avg.keys():
            avg[k] += one[k]
    for k in avg.keys():
        avg[k] = avg[k] / float(len(contributions))
    return avg


def _apply_dampening(global_state: OrderedDict,
                     forget_client_contributions: dict,
                     retain_clients_contributions: dict,
                     dampening_constant: float,
                     dampening_upper_bound: float,
                     ratio_cutoff: float,
                     dampening_lower_bound: float = 0.6,
                     eps: float = 1e-6):
    """
    与旧版 apply_dampening 等价（保持参数顺序与行为一致）。
    注意：保持与旧版 main.py -> unlearn() 的调用顺序一致，以确保结果完全一致。
    """
    with torch.no_grad():
        for (name_f, forget_grads), (_, retain_grads) in zip(
            forget_client_contributions.items(),
            retain_clients_contributions.items()
        ):
            if len(forget_grads.shape) > 0:
                weight = global_state[name_f]
                retain_contribution = torch.abs(retain_grads)
                forget_contribution = torch.abs(forget_grads)
                # 避免 0 除；并把乘子钳在 [lower, upper]，防止“清空”
                ratio = retain_contribution / (forget_contribution + eps)
                update_locations = (ratio < ratio_cutoff)
                dampening_factor = ratio * dampening_constant
                update = dampening_factor[update_locations].clone()
                # 钳制到区间 [lower_bound, upper_bound]
                if dampening_lower_bound is not None:
                    update.clamp_(min=float(dampening_lower_bound), max=float(dampening_upper_bound))
                else:
                    update.clamp_(max=float(dampening_upper_bound))
                weight[update_locations] = weight[update_locations].mul(update)
    return global_state


@typechecked
def run_conda(global_model: torch.nn.Module,
              weights_path: str,
              forget_clients: List[int],
              total_num_clients: int,
              dampening_constant: float = 0.5,
              dampening_upper_bound: float = 0.5,
              ratio_cutoff: float = 0.5,
              dampening_lower_bound: float = 0.6,
              eps: float = 1e-6,
              device: str = "cpu") -> torch.nn.Module:
    """
    将“Contribution Dampening（旧 LEGACY_UNLEARN）”封装为 baseline：
    - 期望 weights_path 为实验根目录（其下有 full_training）。
    - 读取 full_training/initial_model.pth 以及 iteration_*/client_*.pth。
    - 对 global_model.state_dict() 做原地 dampening 并返回模型。
    """
    training_weights_path = os.path.join(weights_path, "full_training")
    if not os.path.isdir(training_weights_path):
        raise RuntimeError(f"[conda] Not found: {training_weights_path}")

    # 初始模型（作为“起点权重”，与旧版一致）
    start_model: OrderedDict = torch.load(
        os.path.join(training_weights_path, "initial_model.pth"),
        map_location="cpu", weights_only=True
    )

    # 枚举所有 iteration_* 目录（与旧版一致）
    items = [
        d for d in os.listdir(training_weights_path)
        if os.path.isdir(os.path.join(training_weights_path, d)) and d.startswith("iteration_")
    ]
    iter_ids = sorted(int(d.split("_")[1]) for d in items)

    # 累积每个 client 的贡献（与旧版一致）
    avg_contributions = {}
    client_counts = {}
    for it in iter_ids:
        it_dir = os.path.join(training_weights_path, f"iteration_{it}")
        client_contrib = _get_client_contribution(start_model, it_dir)
        for client_id, contrib in client_contrib.items():
            if client_id not in avg_contributions:
                avg_contributions[client_id] = {}
                client_counts[client_id] = 0
            for param in start_model.keys():
                if param not in avg_contributions[client_id]:
                    avg_contributions[client_id][param] = torch.zeros_like(contrib[param])
                avg_contributions[client_id][param] += contrib[param]
            client_counts[client_id] += 1

    # —— 按“该客户端出现的轮数”做平均，避免少参加轮的客户端被高估
    for cid, contrib in avg_contributions.items():
        cnt = max(1, client_counts.get(cid, 1))
        for k in contrib.keys():
            contrib[k] = contrib[k] / float(cnt)

    # —— 仅用“真正出现过”的客户端，避免 KeyError
    present_clients = sorted(avg_contributions.keys())
    # 过滤出确实存在于训练产物里的 forget 集
    forget_group = [c for c in forget_clients if c in avg_contributions]
    missing_fg = [c for c in forget_clients if c not in avg_contributions]
    if missing_fg:
        raise RuntimeError(
            f"[conda] 找不到以下忘却客户端的贡献：{missing_fg}；"
            f"full_training 中实际出现的客户端为：{present_clients}"
        )
    # 保留集 = 出现过的客户端 - 忘却集
    retain_ids = [c for c in present_clients if c not in forget_group]
    if not retain_ids:
        raise RuntimeError(
            f"[conda] 过滤后没有可用于聚合的保留客户端；forget={forget_group}，present={present_clients}"
        )

    # 组内/组外平均贡献（保持旧版顺序与行为）
    forget_contrib = _get_group_contribution([avg_contributions[c] for c in forget_group])
    retain_contrib = _get_group_contribution([avg_contributions[c] for c in retain_ids])

    # 注意：沿用旧版 main.py 的实参顺序（avg_client -> 第二参、avg_forget -> 第三参）
    # 以确保与历史结果完全对齐（即使命名上略显违和）。
    state = global_model.cpu().state_dict()
    state = _apply_dampening(
        state,
        retain_contrib,                # 与旧版传参顺序保持一致
        forget_contrib,
        dampening_constant=dampening_constant,
        dampening_upper_bound=dampening_upper_bound,
        ratio_cutoff=ratio_cutoff,
        dampening_lower_bound=dampening_lower_bound,
        eps=eps,
    )
    global_model.load_state_dict(state)
    if device:
        global_model.to(device)
    return global_model