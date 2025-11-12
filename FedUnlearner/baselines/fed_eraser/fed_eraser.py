# -*- coding: utf-8 -*-
"""
FedEraser baseline (reliable & memory-safe)
-------------------------------------------
- Load checkpoints on CPU, move tensors to device only when needed
- Fix: optimizer must bind the actual client_model used in calibration
- Fix: keep all tensors on the same device during vector math
- Stable normalization with epsilon
- Safer calibration lr (cap at 1e-2 by default)

Expected external deps in your repo:
- FedUnlearner.utils.average_weights(dir, device) -> Dict[str, Tensor]
- FedUnlearner.fed_learn.train_local_model(model, dataloader, loss_fn, optimizer, num_epochs, device) -> state_dict
- FedUnlearner.fed_learn.fed_avg(list_of_state_dicts) -> Dict[str, Tensor]
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import re
from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm

from FedUnlearner.utils import average_weights
from FedUnlearner.fed_learn import train_local_model


# ---- utils -----------------------------------------------------------------

def _safe_load(path: str):
    """
    torch.load with CPU map_location. If PyTorch supports weights_only, use it.
    This is backward compatible with older torch versions.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # older torch: no 'weights_only'
        obj = torch.load(path, map_location="cpu")
    # 兼容 {'state_dict': ...} / DataParallel 包裹
    if isinstance(obj, dict):
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            obj = obj['state_dict']
        # 去掉 'module.' 前缀（若有）
        if all(isinstance(k, str) for k in obj.keys()):
            needs_strip = any(k.startswith("module.") for k in obj.keys())
            if needs_strip:
                obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return obj


def _sd_to_device(sd: Dict[str, torch.Tensor], device: torch.device | str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in sd.items()}

def _simple_fed_avg(list_params: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    In-memory FedAvg over a list of state_dicts (uniform weights).
    Assume tensors already on the same device/dtype.
    """
    assert len(list_params) > 0, "list_params is empty"
    out: Dict[str, torch.Tensor] = {}
    keys = list_params[0].keys()
    for k in keys:
        tensors = [sd[k] for sd in list_params if k in sd]
        # 只对浮点/复数做均值；整型/Bool 等保持第一个（典型如 BN 的 num_batches_tracked）
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            stacked = torch.stack(tensors, dim=0)
            out[k] = stacked.mean(dim=0)
        else:
            # 保持 dtype 与语义（计数/掩码等不平均）
            out[k] = tensors[0].clone()
    return out


# ---- core one-step geometry -------------------------------------------------

def fed_eraser_one_step(
    old_client_models: List[Dict[str, torch.Tensor]],
    new_client_models: List[Dict[str, torch.Tensor]],
    global_model_before_forget: Dict[str, torch.Tensor],
    global_model_after_forget: Dict[str, torch.Tensor],
    device: torch.device | str,
    eps: float = 1e-12,
    # ==== 可调遗忘强度开关 ====
    strength: float = 1.0,                 # 步长强度倍乘
    scale_from: str = "old",               # {"old","new","none"}
    normalize: bool = True,                # 是否除以方向范数
    max_step_ratio: Optional[float] = None,# 步长范数 ≤ ratio * ||newGM[layer]||
    apply_regex: Optional[str] = None,     # 只对匹配层生效（如 "fc|classifier"）
) -> Dict[str, torch.Tensor]:
    """
    Implements (stable direction):
        out_t = newGM_t  +  || Σ_i (oldCM_i^{t-1} - oldGM_{t-1}) || *
                           ( Σ_i (newCM_i^{t}   - oldCM_i^{t-1}) ) / ( || Σ_i (newCM_i^{t} - oldCM_i^{t-1}) || + eps )

    All inputs are moved to `device` inside.
    """

    # ensure same device
    oldGM = _sd_to_device(global_model_before_forget, device)
    newGM = _sd_to_device(global_model_after_forget, device)
    oldCMs = [_sd_to_device(sd, device) for sd in old_client_models]
    newCMs = [_sd_to_device(sd, device) for sd in new_client_models]

    out: Dict[str, torch.Tensor] = {}
    pat = re.compile(apply_regex) if apply_regex else None

    # init accumulators
    for layer in newGM.keys():
        # 层选择：不匹配则保持不动
        if pat and (pat.search(layer) is None):
            out[layer] = newGM[layer].clone()
            continue
        out[layer] = newGM[layer].clone()

        # Σ(旧增量) & 跨轮方向
        delta_old_sum   = torch.zeros_like(newGM[layer])
        cross_round_dir = torch.zeros_like(newGM[layer])  # Σ_i (newCM - oldCM)
        for i in range(len(oldCMs)):
            delta_old_sum   = delta_old_sum   + (oldCMs[i][layer] - oldGM[layer])
            cross_round_dir = cross_round_dir + (newCMs[i][layer] - oldCMs[i][layer])

        # 选择尺度来源
        if scale_from == "old":
            scale = torch.norm(delta_old_sum)
        elif scale_from == "new":
            scale = torch.norm(cross_round_dir)
        else:  # "none"
            scale = torch.tensor(1.0, device=newGM[layer].device, dtype=newGM[layer].dtype)

        # 是否对方向做 L2 归一化
        denom = (torch.norm(cross_round_dir) + eps) if normalize else torch.tensor(1.0, device=newGM[layer].device, dtype=newGM[layer].dtype)
        step = strength * (scale / denom) * cross_round_dir

        # 可选：按层裁剪步长范数，抑制过冲
        if (max_step_ratio is not None) and (max_step_ratio > 0):
            ref = torch.norm(newGM[layer]) + eps
            max_norm = float(max_step_ratio) * ref
            s = torch.norm(step)
            if s > max_norm:
                step = step * (max_norm / (s + 1e-12))

        out[layer] = out[layer] + step

    return out


# ---- main entry -------------------------------------------------------------

def run_fed_eraser(
    global_model: nn.Module,
    weights_path: str,
    forget_clients: List[int],
    clientwise_dataloaders: Dict[int, torch.utils.data.DataLoader],
    device: str,
    optimizer_name: str,
    num_clients: int,
    num_rounds: int,
    lr: float,
    num_unlearn_rounds: int = 1,
    local_cali_round: int = 1,
    num_post_training_rounds: int = 1,
    # ==== FedEraser 强度控制参数（传给 one_step）====
    fe_strength: float = 1.0,
    fe_scale_from: str = "old",
    fe_normalize: bool = True,
    fe_max_step_ratio: Optional[float] = None,
    fe_apply_regex: Optional[str] = None,
    fe_eps: float = 1e-12,
) -> nn.Module:
    """
    Args:
        weights_path: path to full_training dir with iteration_{r}/global_model.pth & client_{i}.pth
        forget_clients: list of client ids to forget (this impl assumes single forget client in main)
        num_rounds: should match your training rounds
    """
    # 同时保留两种形式：字符串给外部API，torch.device给张量计算
    device_str = device if isinstance(device, str) else device.type
    device_torch = torch.device(device_str)

    # 1) preload all old global models (CPU tensors)
    old_global_models: List[Dict[str, torch.Tensor]] = []
    for rnd in range(num_rounds):
        gpath = os.path.join(weights_path, f"iteration_{rnd}", "global_model.pth")
        old_global_models.append(_safe_load(gpath))

    # chosen clients = retain set
    chosen_clients = [i for i in range(num_clients) if i not in forget_clients]

    # 2) one-step geometry per round, keep the *last* as the final unlearned state
    unlearned_global_model = deepcopy(global_model)
    for rnd in range(num_rounds):
        iter_dir = os.path.join(weights_path, f"iteration_{rnd}")

        # t-1 不存在就跳过（第0轮无几何步）
        if rnd == 0:
            continue
        prev_dir = os.path.join(weights_path, f"iteration_{rnd-1}")

        old_prev_global = old_global_models[rnd-1]  # CPU dict (t-1)

        # load old/new client parameters (CPU)
        old_client_parameters: List[Dict[str, torch.Tensor]] = []
        new_client_parameters: List[Dict[str, torch.Tensor]] = []

        # 旧：上一轮（prev_dir）；新：当前轮（iter_dir）
        for cid in chosen_clients:
            old_client_parameters.append(_safe_load(os.path.join(prev_dir, f"client_{cid}.pth")))

        # new_prev_global = FedAvg of this round (on device, retain clients only for consistency)
        # 如果 average_weights(dir) 默认均值全体客户端，这里用 retain 客户端手动均值，避免把被遗忘客户端的权重混进 newGM。
        tmp_new = []
        for cid in chosen_clients:
            tmp_new.append(_safe_load(os.path.join(iter_dir, f"client_{cid}.pth")))
        # tensors on device for averaging
        tmp_new_dev = [_sd_to_device(sd, device_torch) for sd in tmp_new]
        # 简单均值
        keys = tmp_new_dev[0].keys()
        new_prev_global = {k: torch.stack([sd[k] for sd in tmp_new_dev], dim=0).mean(dim=0) for k in keys}

        for cid in chosen_clients:
            new_client_parameters.append(_safe_load(os.path.join(iter_dir, f"client_{cid}.pth")))

        # one-step erase on device
        unlearned_sd = fed_eraser_one_step(
            old_client_models=old_client_parameters,
            new_client_models=new_client_parameters,
            global_model_before_forget=old_prev_global,
            global_model_after_forget=new_prev_global,  # already on device in average_weights
            device=device_torch,
            eps=fe_eps,
            strength=fe_strength,
            scale_from=fe_scale_from,
            normalize=fe_normalize,
            max_step_ratio=fe_max_step_ratio,
            apply_regex=fe_apply_regex,
        )

        # load into model (keep on device for the next stage)
        unlearned_global_model = deepcopy(global_model).to(device_torch)
        unlearned_global_model.load_state_dict(unlearned_sd)

        # free cpu tensors early
        del old_client_parameters, new_client_parameters
        torch.cuda.empty_cache()

    # 3) optional post-training calibration on retain clients
    if num_post_training_rounds > 0 and len(chosen_clients) > 0:
        for r in range(num_post_training_rounds):
            print(f"Finetuning FedEraser unlearned Model: {r}")
            list_params = []

            # safer calibration lr (avoid washing away geometry step)
            cali_lr = lr if lr <= 1e-2 else 1e-2

            for cid in tqdm(chosen_clients):
                client_model = deepcopy(unlearned_global_model).to(device_torch)

                if optimizer_name.lower() == "adam":
                    optimizer = torch.optim.Adam(client_model.parameters(), lr=cali_lr)
                elif optimizer_name.lower() == "sgd":
                    optimizer = torch.optim.SGD(client_model.parameters(), lr=cali_lr, momentum=0.9, weight_decay=5e-4)
                else:
                    raise ValueError(f"Optimizer {optimizer_name} not supported")

                loss_fn = nn.CrossEntropyLoss()
                print(f"-----------client {cid} starts training----------")
                tem_param = train_local_model(
                    model=client_model,
                    dataloader=clientwise_dataloaders[cid],
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    num_epochs=local_cali_round,
                    device=device_str,
                )
                list_params.append(tem_param)

            # server aggregation
            global_param = _simple_fed_avg(list_params)
            unlearned_global_model.load_state_dict(global_param)
            torch.cuda.empty_cache()

    return unlearned_global_model
