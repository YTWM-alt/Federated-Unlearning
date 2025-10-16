# fedfim_client.py
from collections import OrderedDict
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def _zeros_like_state_dict(model: nn.Module, device: Optional[torch.device] = None) -> "OrderedDict[str, torch.Tensor]":
    """
    [改] 为了对齐 state_dict 的键顺序，先创建同形状的 0 张量字典。
    只包含 requires_grad=True 的参数，避免把 BN 缓存等 buffer 加进去。
    """
    zeros = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            zeros[name] = torch.zeros_like(p, device=device if device is not None else p.device)
    return zeros


def _add_inplace(dst: "OrderedDict[str, torch.Tensor]", src: "OrderedDict[str, torch.Tensor]") -> None:
    """
    [改] 逐键相加，原地更新，避免多余内存。
    """
    for k in dst.keys():
        dst[k].add_(src[k])


def _square_grads_to_dict(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """
    [移植] 把当前 param.grad 的逐元素平方收集成与 state_dict 对齐的字典。
    """
    out = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                raise RuntimeError(f"Parameter {name} has no grad. Did you call backward(retain_graph=True)?")
            out[name] = p.grad.pow(2).detach()
    return out


def client_compute_fim_diag(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    num_classes: Optional[int] = None,
    criterion: Optional[nn.Module] = None,
    max_passes: int = 1,
    max_batches: Optional[int] = None,
    mode: str = "prob",            # "prob" = 按类别概率做期望 [移植]；"label" = 仅用真实标签近似 [改]
    topk: Optional[int] = None,    # 只对概率最大的 k 个类别做期望，减少计算 [改]
    normalize: bool = True,        # 是否按样本数做归一化 [改]
    eps: float = 1e-8              # 数值稳健 [改]
) -> Tuple["OrderedDict[str, torch.Tensor]", int]:
    """
    计算经验 Fisher 的对角近似，并返回 (fim_diag, n_k)。

    返回：
      fim_diag: OrderedDict，与 model.state_dict() 的可训练参数一一对应
      n_k:      本客户端用于样本量加权的样本数

    说明：
      mode="prob"  —— [移植] 与 FisherForgetting 的核心一致：对所有（或 top-k）类别做
                      CE(out, cls) 的梯度平方，并以 softmax 概率加权求和，得到 E_y[∇θℓ(x,y)^2]
      mode="label" —— [改] 只用真实标签做经验 Fisher（更快），常见于 EWC/FedCurv 的实现
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")

    model = model.to(device)
    model.eval()  # [改] 冻结 BN/Dropout 的行为，但仍然会计算梯度

    fim_sum = _zeros_like_state_dict(model, device=device)  # [改]
    seen = 0

    # 训练图需要梯度
    torch.set_grad_enabled(True)

    for _ in range(max_passes):
        for bidx, (x, y) in enumerate(dataloader):
            if max_batches is not None and bidx >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            seen += x.size(0)

            out = model(x)

            if mode == "label":
                # [改] 真实标签近似：empirical Fisher ≈ (∇θ ℓ(x, y_true))^2
                loss = criterion(out, y)
                model.zero_grad(set_to_none=True)
                loss.backward()  # 不需要保图
                grad2 = _square_grads_to_dict(model)
                _add_inplace(fim_sum, grad2)

            elif mode == "prob":
                # [移植] 概率期望写法：对每个候选类别计算 CE(out, cls) 的梯度平方，再按 softmax 概率加权
                with torch.no_grad():
                    prob = F.softmax(out, dim=-1)  # [移植]
                C = out.size(1) if num_classes is None else num_classes

                # 选 top-k 类别减少计算 [改]
                if topk is not None and 1 <= topk < C:
                    # 取每个样本的 top-k 类别集合的并集，近似整体 top-k
                    topk_idx = torch.topk(prob, k=topk, dim=1).indices
                    cls_set = torch.unique(topk_idx).tolist()
                else:
                    cls_set = list(range(C))

                # 为了把 batch 权重弄正确，按“类别权重的批均值”进行加权 [改：工程近似]
                for cls in cls_set:
                    target = torch.full_like(y, fill_value=cls)
                    loss = criterion(out, target)
                    model.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=True)  # [移植] 需要保留计算图以遍历多个 cls
                    grad2 = _square_grads_to_dict(model)

                    # 批内对该类别的平均概率，作为权重 [移植/改：与原理一致的常用近似]
                    w = prob[:, cls].mean()

                    # 逐键加权累计
                    for k in fim_sum.keys():
                        fim_sum[k].add_(w * grad2[k])
            else:
                raise ValueError(f"Unknown mode={mode}. Choose from ['prob','label'].")

    # 归一化
    if normalize and seen > 0:
        inv = 1.0 / max(float(seen), eps)
        for k in fim_sum.keys():
            fim_sum[k].mul_(inv)

    torch.set_grad_enabled(False)
    return fim_sum, seen
