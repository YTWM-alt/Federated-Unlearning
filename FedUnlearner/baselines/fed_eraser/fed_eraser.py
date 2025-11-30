# -*- coding: utf-8 -*-
"""
FedEraser / Projected Unlearning baseline (Robust Version)
----------------------------------------------------------
Fixes the "BN Collapse" issue in Non-IID settings by using Algebraic Removal
instead of naive FedAvg of retain clients.

Logic:
  Unlearned_Model ~ Global_Model + Strength * (Global_Model - Forget_Client_Model)
  (Moving the global model away from the forget client's direction)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm
import random

from FedUnlearner.fed_learn import train_local_model

# ---- utils -----------------------------------------------------------------

def _safe_load(path: str):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if 'state_dict' in obj:
            obj = obj['state_dict']
        if all(isinstance(k, str) for k in obj.keys()):
            if any(k.startswith("module.") for k in obj.keys()):
                obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return obj

def _sd_to_device(sd: Dict[str, torch.Tensor], device: torch.device | str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in sd.items()}

def _simple_fed_avg(list_params: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    assert len(list_params) > 0
    out: Dict[str, torch.Tensor] = {}
    keys = list_params[0].keys()
    for k in keys:
        tensors = [sd[k] for sd in list_params if k in sd]
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            out[k] = torch.stack(tensors).mean(dim=0)
        else:
            out[k] = tensors[0].clone()
    return out

@torch.no_grad()
def _eval_forget_client_basic(model, forget_loader, device, tag=""):
    dev = torch.device(device if isinstance(device, str) else device.type)
    tmp = deepcopy(model).to(dev).eval()
    total, correct = 0, 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    
    for batch in forget_loader:
        if isinstance(batch, (list, tuple)): x, y = batch[0], batch[1]
        else: continue
        x, y = x.to(dev), y.to(dev)
        out = tmp(x)
        loss_sum += float(loss_fn(out, y).item())
        correct += int((out.argmax(1) == y).sum().item())
        total += y.numel()
        
    acc = correct / total if total > 0 else 0.0
    ce = loss_sum / total if total > 0 else 0.0
    print(f"[FedEraser-Eval] {tag}: Acc={acc:.4f}, Loss={ce:.4f}, N={total}")
    del tmp

# ---- Algebraic Eraser Step --------------------------------------------------

def algebraic_eraser_step(
    global_model_state: Dict[str, torch.Tensor],
    forget_client_state: Dict[str, torch.Tensor],
    device: torch.device | str,
    strength: float = 1.0,  # Recommended: 1.0 corresponds to exact algebraic removal
    clip_threshold: float = 0.2, # Safety clip: max change relative to param norm
) -> Dict[str, torch.Tensor]:
    """
    Computes: W_new = W_global + strength * (W_global - W_forget)
    This algebraically approximates the model trained only on (N-1) clients.
    """
    g_sd = _sd_to_device(global_model_state, device)
    f_sd = _sd_to_device(forget_client_state, device)
    out: Dict[str, torch.Tensor] = {}

    total_diff_norm = 0.0
    
    for k in g_sd.keys():
        # Only modify float parameters (weights/biases/running_stats)
        if not g_sd[k].is_floating_point():
            out[k] = g_sd[k].clone()
            continue
            
        # Direction: Global - Forget (Moving AWAY from forget)
        diff = g_sd[k] - f_sd[k]
        
        # Apply strength
        step = strength * diff
        
        # Safety Clipping: Don't let weights explode
        param_norm = torch.norm(g_sd[k])
        step_norm = torch.norm(step)
        if step_norm > clip_threshold * (param_norm + 1e-6):
            step = step * (clip_threshold * (param_norm + 1e-6) / (step_norm + 1e-12))
            
        out[k] = g_sd[k] + step
        total_diff_norm += float(step_norm.item()**2)

    print(f"[FedEraser-Step] Total Update Norm: {total_diff_norm**0.5:.4f} (Strength={strength})")
    return out

# ---- Main Runner ------------------------------------------------------------

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
    # Eraser Params
    fe_strength: float = 0.05, # Default to 1/N approx
    fe_scale_from: str = "new", 
    fe_normalize: bool = True,
    fe_max_step_ratio: float = 0.1,
    fe_apply_regex: str = None,
    fe_eps: float = 1e-12,
) -> nn.Module:
    
    device_torch = torch.device(device)
    forget_cid = forget_clients[0]
    forget_loader = clientwise_dataloaders.get(forget_cid)

    print(f"[FedEraser] Robust Algebraic Mode. Target Client: {forget_cid}")
    print(f"[FedEraser] Params: Strength={fe_strength}, Post_Rounds={num_post_training_rounds}")

    if forget_loader:
        _eval_forget_client_basic(global_model, forget_loader, device_torch, "Start")

    # 1. Load the FINAL models from the full training
    # We only need the last round to perform the removal
    last_round_idx = num_rounds - 1
    
    # Path to Final Global Model
    global_ckpt = os.path.join(weights_path, "final_model.pth")
    if not os.path.exists(global_ckpt):
        global_ckpt = os.path.join(weights_path, f"iteration_{last_round_idx}", "global_model.pth")
    
    # Path to Final Forget Client Model
    forget_ckpt = os.path.join(weights_path, f"iteration_{last_round_idx}", f"client_{forget_cid}.pth")
    
    if not os.path.exists(global_ckpt) or not os.path.exists(forget_ckpt):
        raise FileNotFoundError(f"Missing models for FedEraser:\nGlobal: {global_ckpt}\nClient: {forget_ckpt}")

    print(f"[FedEraser] Loading Global: {global_ckpt}")
    print(f"[FedEraser] Loading Forget: {forget_ckpt}")

    global_sd = _safe_load(global_ckpt)
    forget_sd = _safe_load(forget_ckpt)

    # 2. Perform Algebraic Erasure
    # "Removing" one client from the average is equivalent to:
    # W_new = W_global + alpha * (W_global - W_forget)
    # Ideally alpha = 1 / (Total_Clients - 1). For 20 clients, alpha ~ 0.053
    # Use fe_strength to override this if needed.
    
    unlearned_sd = algebraic_eraser_step(
        global_model_state=global_sd,
        forget_client_state=forget_sd,
        device=device_torch,
        strength=fe_strength,
        clip_threshold=fe_max_step_ratio or 0.2
    )

    unlearned_model = deepcopy(global_model).to(device_torch)
    unlearned_model.load_state_dict(unlearned_sd)

    if forget_loader:
        _eval_forget_client_basic(unlearned_model, forget_loader, device_torch, "After_Erasure")

    # 3. Post-training Calibration (Essential for BN repair)
    # Using Retain Clients to fix BN stats and fine-tune
    retain_cids = [i for i in range(num_clients) if i not in forget_clients]
    
    if num_post_training_rounds > 0:
        # Use a smaller LR for repair to avoid undoing the unlearning
        repair_lr = lr * 0.1 if lr > 0.01 else 0.001
        print(f"[FedEraser] Starting {num_post_training_rounds} rounds of repair (LR={repair_lr})...")
        
        for r in range(num_post_training_rounds):
            local_weights = []
            # Subsample retain clients to speed up if many
            round_clients = retain_cids if len(retain_cids) < 10 else random.sample(retain_cids, 10)
            
            for cid in round_clients:
                # print(f"Repairing on Client {cid}...")
                c_model = deepcopy(unlearned_model)
                if optimizer_name == 'adam':
                    opt = torch.optim.Adam(c_model.parameters(), lr=repair_lr)
                else:
                    opt = torch.optim.SGD(c_model.parameters(), lr=repair_lr, momentum=0.9)
                
                # Train only 1 epoch per round for repair
                updated_sd = train_local_model(
                    model=c_model,
                    dataloader=clientwise_dataloaders[cid],
                    loss_fn=nn.CrossEntropyLoss(),
                    optimizer=opt,
                    num_epochs=local_cali_round,
                    device=device
                )
                local_weights.append(updated_sd)
            
            # Aggregation
            # Note: We average state_dicts which includes BN stats. 
            # Since we started from a "Global-ish" model, this average is safer now.
            avg_sd = _simple_fed_avg(local_weights)
            unlearned_model.load_state_dict(avg_sd)
            
    if forget_loader:
        _eval_forget_client_basic(unlearned_model, forget_loader, device_torch, "Final")

    return unlearned_model