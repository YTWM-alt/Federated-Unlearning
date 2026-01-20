import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Union

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


def empirical_fisher_full(model: torch.nn.Module,
                          dataloader: DataLoader,
                          param_keys: List[str],
                          device: str = "cpu",
                          max_batches: Optional[int] = 10) -> torch.Tensor:
    """
    [新增] 计算全量 Fisher 矩阵 (Full Fisher Matrix)。
    返回一个 D x D 的稠密矩阵，其中 D 是 param_keys 指定参数的总维度。
    注意：D 较大时极易 OOM。
    """
    model.to(device)
    model.eval()

    # 确保 requires_grad
    for p in model.parameters():
        p.requires_grad_(True)

    # 预计算总维度 D
    state_dict = model.state_dict()
    D = sum(state_dict[k].numel() for k in param_keys)
    print(f"[Full Fisher] Computing D x D matrix where D={D}. Matrix size ~ {D*D*4/1024**3:.2f} GB")

    # 初始化大矩阵
    # 建议放在 CPU，除非显存非常大
    F = torch.zeros((D, D), device=device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    n = 0

    with torch.enable_grad():
        for b, batch in enumerate(dataloader):
            if max_batches is not None and b >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)

            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # 收集展平的梯度向量 g (仅包含 param_keys)
            grads = []
            for k in param_keys:
                # 有些参数可能没有梯度（如 freeze 了），视为 0
                # 通过 state_dict 获取 param 对象比较麻烦，这里直接遍历 named_parameters 匹配名字
                # 优化：建立 name->param 映射
                pass 
            
            # 更快的方法：遍历 named_parameters
            grad_vecs = []
            for name, p in model.named_parameters():
                if name in param_keys:
                    if p.grad is not None:
                        grad_vecs.append(p.grad.detach().view(-1))
                    else:
                        grad_vecs.append(torch.zeros_like(p).view(-1))
            
            if not grad_vecs:
                continue

            g = torch.cat(grad_vecs) # shape (D,)
            
            # 外积 g * g^T
            # F += g.unsqueeze(1) @ g.unsqueeze(0)
            # 优化：使用 addmm 或者 rank-1 update
            F.addr_(g, g)
            
            n += 1

    if n > 0:
        F /= float(n)
    
    return F
