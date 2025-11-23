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

@torch.no_grad()
def _eval_forget_client_basic(
    model: nn.Module,
    forget_loader,
    device: torch.device | str,
    tag: str = "",
) -> Dict[str, float]:
    """Quick evaluation on the *forget* client's dataloader.

    打印遗忘客户端在当前模型下的精度和平均交叉熵，
    用来区分：几何步 / post-training 分别对遗忘强度和整体性能的影响。
    """
    device_str = device if isinstance(device, str) else device.type
    dev = torch.device(device_str)
    tmp_model = deepcopy(model).to(dev)
    tmp_model.eval()

    total = 0
    correct = 0
    ce_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in forget_loader:
            # 兼容 (x, y) 或 (x, y, ...) 形式
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                continue

            x = x.to(dev)
            y = y.to(dev)
            logits = tmp_model(x)
            loss = loss_fn(logits, y)
            ce_sum += float(loss.item())

            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

    acc = correct / total if total > 0 else 0.0
    ce = ce_sum / total if total > 0 else 0.0
    print(
        f"[FedEraser-Eval] {tag}: "
        f"forget_client_acc={acc:.4f}, forget_client_ce={ce:.4f}, n={total}"
    )
    return {"acc": acc, "ce": ce, "n": total}


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

    # ---- debug accumulators (per one_step call) ----
    n_total_layers = 0
    n_moved_layers = 0
    n_clipped_layers = 0
    total_norm_old_sq = 0.0
    total_norm_dir_sq = 0.0
    total_norm_step_sq = 0.0

    # init accumulators
    for layer in newGM.keys():
        # 层选择：不匹配则保持不动
        if pat and (pat.search(layer) is None):
            out[layer] = newGM[layer].clone()
            continue

        # 非浮点 / 非复数参数（一般是计数或掩码，如 num_batches_tracked），
        # 不做几何步，直接保持不变，避免在 LongTensor 上做 norm/mean 等操作。
        if (not newGM[layer].is_floating_point()) and (not newGM[layer].is_complex()):
            out[layer] = newGM[layer].clone()
            continue

        out[layer] = newGM[layer].clone()

        # Σ(旧增量) & 跨轮方向
        delta_old_sum   = torch.zeros_like(newGM[layer])
        cross_round_dir = torch.zeros_like(newGM[layer])  # Σ_i (newCM - oldCM)
        for i in range(len(oldCMs)):
            delta_old_sum   = delta_old_sum   + (oldCMs[i][layer] - oldGM[layer])
            cross_round_dir = cross_round_dir + (newCMs[i][layer] - oldCMs[i][layer])

        # 当前层范数
        norm_old = torch.norm(delta_old_sum)
        norm_dir = torch.norm(cross_round_dir)

        # 选择尺度来源
        if scale_from == "old":
            scale = norm_old
        elif scale_from == "new":
            scale = norm_dir
        else:  # "none"
            scale = torch.tensor(1.0, device=newGM[layer].device, dtype=newGM[layer].dtype)

        # 是否对方向做 L2 归一化
        denom = (torch.norm(cross_round_dir) + eps) if normalize else torch.tensor(
            1.0, device=newGM[layer].device, dtype=newGM[layer].dtype
        )
        step = strength * (scale / denom) * cross_round_dir
        step_norm_before = torch.norm(step)

        # 可选：按层裁剪步长范数，抑制过冲
        clipped = False
        if (max_step_ratio is not None) and (max_step_ratio > 0):
            ref = torch.norm(newGM[layer]) + eps
            max_norm = float(max_step_ratio) * ref
            if step_norm_before > max_norm:
                step = step * (max_norm / (step_norm_before + 1e-12))
                clipped = True

        step_norm_after = torch.norm(step)

        out[layer] = out[layer] + step

        # ---- debug: per-layer stats ----
        n_total_layers += 1
        if step_norm_after.item() > 0:
            n_moved_layers += 1
        if clipped:
            n_clipped_layers += 1

        total_norm_old_sq += float(norm_old.item() ** 2)
        total_norm_dir_sq += float(norm_dir.item() ** 2)
        total_norm_step_sq += float(step_norm_after.item() ** 2)

        print(
            "[FedEraser-OneStep] "
            f"layer={layer}, "
            f"||delta_old_sum||={norm_old.item():.4e}, "
            f"||cross_dir||={norm_dir.item():.4e}, "
            f"scale={scale.item():.4e}, "
            f"step_norm_before={step_norm_before.item():.4e}, "
            f"step_norm_after={step_norm_after.item():.4e}, "
            f"clipped={clipped}"
        )

    # ---- debug: global summary for this one_step call ----
    if n_total_layers > 0:
        global_norm_old = total_norm_old_sq ** 0.5
        global_norm_dir = total_norm_dir_sq ** 0.5
        global_norm_step = total_norm_step_sq ** 0.5
        print(
            "[FedEraser-OneStep] summary: "
            f"layers_total={n_total_layers}, "
            f"layers_moved={n_moved_layers}, "
            f"layers_clipped={n_clipped_layers}, "
            f"global_norm_old={global_norm_old:.4e}, "
            f"global_norm_dir={global_norm_dir:.4e}, "
            f"global_norm_step={global_norm_step:.4e}"
        )

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

    # 单个被遗忘客户端的 dataloader（这里只支持 1 个 forget client）
    forget_loader = None
    forget_client = None
    if len(forget_clients) > 0:
        forget_client = forget_clients[0]
        if forget_client in clientwise_dataloaders:
            forget_loader = clientwise_dataloaders[forget_client]

    # 打印 FedEraser 关键配置，方便在 log 里快速定位问题
    print(
        "[FedEraser] config: "
        f"num_rounds={num_rounds}, "
        f"forget_clients={forget_clients}, "
        f"retain_clients={chosen_clients}, "
        f"lr={lr}, "
        f"num_unlearn_rounds={num_unlearn_rounds}, "
        f"local_cali_round={local_cali_round}, "
        f"num_post_training_rounds={num_post_training_rounds}, "
        f"fe_strength={fe_strength}, "
        f"fe_scale_from='{fe_scale_from}', "
        f"fe_normalize={fe_normalize}, "
        f"fe_max_step_ratio={fe_max_step_ratio}, "
        f"fe_apply_regex={fe_apply_regex}, "
        f"fe_eps={fe_eps}"
    )

    # 在任何 FedEraser 操作之前，先看一眼原始模型在遗忘客户端上的表现
    if forget_loader is not None:
        _eval_forget_client_basic(
            model=global_model,
            forget_loader=forget_loader,
            device=device_torch,
            tag="before_federaser_forget",
        )

    # ---- debug: 全局 FedEraser 配置 ----
    print(
        "[FedEraser] config: "
        f"num_rounds={num_rounds}, "
        f"forget_clients={forget_clients}, "
        f"retain_clients={chosen_clients}, "
        f"lr={lr}, "
        f"num_unlearn_rounds={num_unlearn_rounds}, "
        f"local_cali_round={local_cali_round}, "
        f"num_post_training_rounds={num_post_training_rounds}, "
        f"fe_strength={fe_strength}, "
        f"fe_scale_from='{fe_scale_from}', "
        f"fe_normalize={fe_normalize}, "
        f"fe_max_step_ratio={fe_max_step_ratio}, "
        f"fe_apply_regex={fe_apply_regex}, "
        f"fe_eps={fe_eps}"
    )

    # 2) one-step geometry per round, keep the *last* as the final unlearned state
    unlearned_global_model = deepcopy(global_model)
    for rnd in range(num_rounds):
        iter_dir = os.path.join(weights_path, f"iteration_{rnd}")



        # t-1 不存在就跳过（第0轮无几何步）
        if rnd == 0:
            print(f"[FedEraser] Round {rnd}: skip geometry step (no previous round).")
            continue
        prev_dir = os.path.join(weights_path, f"iteration_{rnd-1}")

        print(
            "[FedEraser] Round "
            f"{rnd}: applying geometry step, prev_dir={prev_dir}, iter_dir={iter_dir}, "
            f"#retain_clients={len(chosen_clients)}"
        )

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
        # 使用安全版 FedAvg：仅对浮点 / 复数做均值，整型 buffer 直接拷贝第一个
        new_prev_global = _simple_fed_avg(tmp_new_dev)

        # ---- debug: global pre-geometry norms (old vs new) ----
        old_prev_global_dev = _sd_to_device(old_prev_global, device_torch)
        total_old_sq = 0.0
        total_new_sq = 0.0
        total_diff_sq = 0.0
        n_g_layers = 0
        for k, v_new in new_prev_global.items():
            if (not v_new.is_floating_point()) and (not v_new.is_complex()):
                continue
            v_old = old_prev_global_dev[k]
            diff = v_new - v_old
            total_old_sq += float(torch.norm(v_old).item() ** 2)
            total_new_sq += float(torch.norm(v_new).item() ** 2)
            total_diff_sq += float(torch.norm(diff).item() ** 2)
            n_g_layers += 1
        if n_g_layers > 0:
            print(
                "[FedEraser] Round "
                f"{rnd}: pre-geom global norms: "
                f"||G_old||={(total_old_sq ** 0.5):.4e}, "
                f"||G_new||={(total_new_sq ** 0.5):.4e}, "
                f"||G_new-G_old||={(total_diff_sq ** 0.5):.4e}, "
                f"layers={n_g_layers}"
            )
        del old_prev_global_dev

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

        # ---- debug: global geometry step norms (new vs unlearned) ----
        total_new_sq = 0.0
        total_unl_sq = 0.0
        total_step_sq = 0.0
        n_step_layers = 0
        for k, v_new in new_prev_global.items():
            if (not v_new.is_floating_point()) and (not v_new.is_complex()):
                continue
            v_unl = unlearned_sd[k]
            diff = v_unl - v_new
            total_new_sq += float(torch.norm(v_new).item() ** 2)
            total_unl_sq += float(torch.norm(v_unl).item() ** 2)
            total_step_sq += float(torch.norm(diff).item() ** 2)
            n_step_layers += 1
        if n_step_layers > 0:
            global_new = total_new_sq ** 0.5
            global_unl = total_unl_sq ** 0.5
            global_step = total_step_sq ** 0.5
            rel_step = global_step / (global_new + 1e-12)
            print(
                "[FedEraser] Round "
                f"{rnd}: geometry step norms: "
                f"||G_new||={global_new:.4e}, "
                f"||G_unlearned||={global_unl:.4e}, "
                f"||step||={global_step:.4e}, "
                f"rel_step={rel_step:.4e}, "
                f"layers={n_step_layers}"
            )

        # load into model (keep on device for the next stage)
        unlearned_global_model = deepcopy(global_model).to(device_torch)
        unlearned_global_model.load_state_dict(unlearned_sd)

        # free cpu tensors early
        del old_client_parameters, new_client_parameters
        torch.cuda.empty_cache()


    # 在做任何 post-training 之前，记录一次“几何步之后”的遗忘客户端表现
    if forget_loader is not None:
        _eval_forget_client_basic(
            model=unlearned_global_model,
            forget_loader=forget_loader,
            device=device_torch,
            tag="after_geometry_forget",
        )

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


    # post-training 结束后，再看一次遗忘客户端的表现
    if forget_loader is not None:
        _eval_forget_client_basic(
            model=unlearned_global_model,
            forget_loader=forget_loader,
            device=device_torch,
            tag="after_posttraining_forget",
        )

    return unlearned_global_model
