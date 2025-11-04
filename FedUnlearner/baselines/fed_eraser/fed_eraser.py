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
from typing import Dict, List
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
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # older torch: no 'weights_only'
        return torch.load(path, map_location="cpu")


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
        # stack -> mean 更稳；避免 python 累加产生的 dtype/device 不一致
        stacked = torch.stack([sd[k] for sd in list_params], dim=0)
        out[k] = stacked.mean(dim=0)
    return out


# ---- core one-step geometry -------------------------------------------------

def fed_eraser_one_step(
    old_client_models: List[Dict[str, torch.Tensor]],
    new_client_models: List[Dict[str, torch.Tensor]],
    global_model_before_forget: Dict[str, torch.Tensor],
    global_model_after_forget: Dict[str, torch.Tensor],
    device: torch.device | str,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Implements:  newGM_t  +  ||sum_i (oldCM_i - oldGM_t)|| *  (sum_i (newCM_i - newGM_t)) / ||sum_i (newCM_i - newGM_t)||

    All inputs are moved to `device` inside.
    """

    # ensure same device
    oldGM = _sd_to_device(global_model_before_forget, device)
    newGM = _sd_to_device(global_model_after_forget, device)
    oldCMs = [_sd_to_device(sd, device) for sd in old_client_models]
    newCMs = [_sd_to_device(sd, device) for sd in new_client_models]

    out: Dict[str, torch.Tensor] = {}

    # init accumulators
    for layer in newGM.keys():
        out[layer] = newGM[layer].clone()
        # sum of deltas
        delta_old_sum = torch.zeros_like(newGM[layer])
        delta_new_sum = torch.zeros_like(newGM[layer])

        for i in range(len(oldCMs)):
            delta_old_sum = delta_old_sum + (oldCMs[i][layer] - oldGM[layer])
            delta_new_sum = delta_new_sum + (newCMs[i][layer] - newGM[layer])

        scale = torch.norm(delta_old_sum)
        denom = torch.norm(delta_new_sum) + eps
        step = (scale / denom) * delta_new_sum
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
) -> nn.Module:
    """
    Args:
        weights_path: path to full_training dir with iteration_{r}/global_model.pth & client_{i}.pth
        forget_clients: list of client ids to forget (this impl assumes single forget client in main)
        num_rounds: should match your training rounds
    """
    device = torch.device(device)

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

        old_prev_global = old_global_models[rnd]  # CPU dict

        # load old/new client parameters (CPU)
        old_client_parameters: List[Dict[str, torch.Tensor]] = []
        new_client_parameters: List[Dict[str, torch.Tensor]] = []

        for cid in chosen_clients:
            old_client_parameters.append(_safe_load(os.path.join(iter_dir, f"client_{cid}.pth")))

        # new_prev_global_model = FedAvg of this round (on device)
        new_prev_global = average_weights(iter_dir, device=device)

        for cid in chosen_clients:
            new_client_parameters.append(_safe_load(os.path.join(iter_dir, f"client_{cid}.pth")))

        # one-step erase on device
        unlearned_sd = fed_eraser_one_step(
            old_client_models=old_client_parameters,
            new_client_models=new_client_parameters,
            global_model_before_forget=old_prev_global,
            global_model_after_forget=new_prev_global,  # already on device in average_weights
            device=device,
        )

        # load into model (keep on device for the next stage)
        unlearned_global_model = deepcopy(global_model).to(device)
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
                client_model = deepcopy(unlearned_global_model).to(device)

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
                    device=device,
                )
                list_params.append(tem_param)

            # server aggregation
            global_param = _simple_fed_avg(list_params)
            unlearned_global_model.load_state_dict(global_param)
            torch.cuda.empty_cache()

    return unlearned_global_model
