# baselines/fedfim/fedfim.py
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .client import client_compute_fim_diag
from .server import partition_and_aggregate
from .dampening import apply_fedfIM_dampening

@torch.no_grad()
def _clone_state_dict(sd: "OrderedDict[str, torch.Tensor]"):
    return OrderedDict((k, v.clone()) for k, v in sd.items())

def run_fedfIM(
    model: nn.Module,
    client_loaders: Dict[int, DataLoader],        # {client_id: DataLoader}
    forget_clients: List[int],                    # 要遗忘的 client 列表
    device: torch.device,
    *,
    num_classes: int,
    fim_max_passes: int = 1,
    fim_max_batches: Optional[int] = None,
    fim_mode: str = "prob",
    fim_topk: Optional[int] = None,
    dampening_constant: float = 0.2,
    ratio_cutoff: float = 1.0,
    upper_bound: float = 1.0,
    finetune_epochs: int = 0,                     # >0 则在保留侧微调
    finetune_lr: float = 1e-3,
    finetune_weight_decay: float = 0.0,
) -> Tuple[nn.Module, "OrderedDict[str, torch.Tensor]", "OrderedDict[str, torch.Tensor]"]:
    """
    以“后处理式”完成一次联邦遗忘：
      1) 各客户端基于当前全局模型计算 FIM；
      2) 服务器聚合得到 F_r / F_f；
      3) 对参数执行乘性抑制（不放大）；
      4) 可选：在保留客户端上微调若干 epoch。
    """
    model.to(device)
    model.eval()
    base_sd = copy.deepcopy(model.state_dict())   # 记录“遗忘前”参数

    # 1) 客户端计算 FIM
    all_fims = {}
    for cid, loader in client_loaders.items():
        fim_k, n_k = client_compute_fim_diag(
            model, loader, device,
            num_classes=num_classes,
            max_passes=fim_max_passes,
            max_batches=fim_max_batches,
            mode=fim_mode,
            topk=fim_topk
        )
        all_fims[cid] = (fim_k, n_k)

    # 2) 服务器聚合
    F_r, F_f = partition_and_aggregate(all_fims, forget_clients=forget_clients)

    # 3) 参数抑制
    new_sd = apply_fedfIM_dampening(
        state_dict=model.state_dict(), F_r=F_r, F_f=F_f,
        dampening_constant=dampening_constant, ratio_cutoff=ratio_cutoff, upper_bound=upper_bound
    )
    model.load_state_dict(new_sd, strict=True)

    # 4) 可选：仅在保留客户端微调，回补精度
    if finetune_epochs > 0:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=finetune_lr, weight_decay=finetune_weight_decay, momentum=0.9)
        ce = nn.CrossEntropyLoss()
        retain_ids = [cid for cid in client_loaders.keys() if cid not in forget_clients]
        for _ in range(finetune_epochs):
            for cid in retain_ids:
                for (x, y) in client_loaders[cid]:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    logits = model(x)
                    loss = ce(logits, y)
                    loss.backward()
                    opt.step()
        model.eval()

    return model, F_r, F_f
