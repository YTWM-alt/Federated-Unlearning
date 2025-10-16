import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional

def empirical_fisher_diagonal(model: torch.nn.Module,
                              dataloader: DataLoader,
                              device: str = "cpu",
                              max_batches: Optional[int] = 10) -> Dict[str, torch.Tensor]:
    """
    经验 Fisher 对角近似：在 eval 模式下保留梯度，逐 batch 反传一次，
    将各参数梯度平方做平均作为 Fisher 对角估计。
    """
    model.to(device)
    model.eval()

    # 确保参数允许求导（有些流水线可能把 requires_grad 关了）
    for p in model.parameters():
        p.requires_grad_(True)

    fisher_diag: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(p, device=device)
        for name, p in model.named_parameters() if p.requires_grad
    }

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    n = 0

    # 显式开启梯度跟踪
    with torch.enable_grad():
        for b, batch in enumerate(dataloader):
            if max_batches is not None and b >= max_batches:
                break

            # 兼容 (x,y) 或 (x,y,...) 形式
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch

            x = x.to(device)
            y = y.to(device)

            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            # 反传得到每个参数的梯度
            loss.backward()

            # 累加梯度平方
            for (name, p) in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_diag[name] += (p.grad.detach() ** 2)

            n += 1

    if n == 0:
        return fisher_diag

    for k in fisher_diag:
        fisher_diag[k] /= float(n)

    return fisher_diag
