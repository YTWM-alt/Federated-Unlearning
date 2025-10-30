import torch
from typing import List, Dict, Tuple

def flatten_state_dict(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.view(-1) for v in sd.values()])

def state_dict_like(vec: torch.Tensor, ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    offset = 0
    for k, t in ref.items():
        numel = t.numel()
        out[k] = vec[offset:offset+numel].view_as(t)
        offset += numel
    return out

def weighted_matrix_from_deltas(
    deltas: List[Dict[str, torch.Tensor]], 
    fisher: Dict[str, torch.Tensor], 
    device: str = "cuda"
) -> torch.Tensor:
    flat_f = flatten_state_dict(fisher).to(device)
    w = torch.sqrt(flat_f + 1e-12)
    X = torch.stack([flatten_state_dict(d).to(device) for d in deltas], dim=0)
    return X * w.unsqueeze(0)

def topk_right_singular_vectors(X: torch.Tensor, k: int) -> torch.Tensor:
    # 保留但不再在大维度上调用；大维度场景请使用下方 Gram 版本
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh.T[:, :k]

def rho_values(V: torch.Tensor, other_deltas: List[Dict[str, torch.Tensor]]) -> List[float]:
    if len(other_deltas) == 0:
        return [0.0 for _ in range(V.shape[1])]
    device = V.device
    D = torch.stack([flatten_state_dict(d).to(V.device) for d in other_deltas], dim=0)
    return [float(torch.abs(D @ V[:, i]).mean().item()) for i in range(V.shape[1])]

def split_subspaces(V: torch.Tensor, rho: List[float], tau: float) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    spec_idx = [i for i, r in enumerate(rho) if r < tau]
    comm_idx = [i for i, r in enumerate(rho) if r >= tau]
    V_spec = V[:, spec_idx] if len(spec_idx) > 0 else V[:, :0]
    V_comm = V[:, comm_idx] if len(comm_idx) > 0 else V[:, :0]
    return V_spec, V_comm, spec_idx, comm_idx

# === 新增：仅对指定 keys (parameters) 展开/还原 ===
def flatten_by_keys(sd: Dict[str, torch.Tensor], keys: list, device: str = None) -> torch.Tensor:
    vecs = []
    for k in keys:
        t = sd[k]
        if device is not None:
            t = t.to(device)
        vecs.append(t.view(-1))
    return torch.cat(vecs)

def state_dict_like_by_keys(vec: torch.Tensor, ref: Dict[str, torch.Tensor], keys: list) -> Dict[str, torch.Tensor]:
    out = {}
    offset = 0
    for k in keys:
        t = ref[k]
        n = t.numel()
        out[k] = vec[offset:offset+n].view_as(t)
        offset += n
    return out

def weighted_matrix_from_deltas_keys(deltas: list, fisher: Dict[str, torch.Tensor], keys: list, device: str = "cpu") -> torch.Tensor:
    # 对齐与防御：keys 不能为空，且必须同时存在于 fisher 与每个 delta 中
    if not keys:
        raise RuntimeError("weighted_matrix_from_deltas_keys: received empty keys after filtering.")
    valid_keys = [k for k in keys if (k in fisher) and all(k in d for d in deltas)]
    if len(valid_keys) == 0:
        sample = list(fisher.keys())[:8]
        raise RuntimeError(
            f"weighted_matrix_from_deltas_keys: no overlapping keys among requested={keys[:6]} and fisher/deltas. "
            f"fisher_sample={sample}"
        )
    w = torch.sqrt(torch.cat([fisher[k].to(device).view(-1) for k in valid_keys]) + 1e-12)
    X = torch.stack([flatten_by_keys(d, valid_keys, device=device) for d in deltas], dim=0)
    return X * w.unsqueeze(0)

@torch.no_grad()
def topk_right_singular_vectors_gram(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    Tall-Skinny 场景（T ≪ D）稳定获取右奇异向量：
      1) Xc = X - mean
      2) 先对 G = Xc Xc^T (T×T) 做特征分解
      3) V = Xc^T U S^{-1}
    内存占用仅与 T×T 成正比，避免直接对 T×D 做 SVD 的巨大工作空间。
    """
    Xc = X - X.mean(dim=0, keepdim=True)         # T × D
    G  = Xc @ Xc.T                               # T × T
    evals, U = torch.linalg.eigh(G)              # 升序
    idx = torch.argsort(evals, descending=True)  # 取前 k
    idx = idx[:min(k, U.shape[1])]
    U_k = U[:, idx]                              # T × k
    S_k = torch.sqrt(torch.clamp(evals[idx], min=1e-12))  # k
    # V = Xc^T U S^{-1}
    V = Xc.T @ (U_k / S_k.unsqueeze(0))          # D × k
    return V


@torch.no_grad()
def rho_values_keys(
    V: torch.Tensor,
    other_deltas: list,
    keys: list,
    max_samples: int = 512,
) -> list:
    """
    流式计算 ρ，避免把所有 other_deltas 堆到 (M×D) 大矩阵里。
    同时对子样本数做上限（默认 512），控制时间/内存。
    ρ_i = mean_{d∈others} |v_i^T d|
    """
    k = int(V.shape[1])
    if len(other_deltas) == 0 or k == 0:
        return [0.0 for _ in range(k)]
    device = V.device
    # —— 子采样，防止 M 极大导致长时间/大内存
    if max_samples is not None and len(other_deltas) > max_samples:
        step = max(1, len(other_deltas) // max_samples)
        others = other_deltas[::step][:max_samples]
    else:
        others = other_deltas
    totals = torch.zeros(k, device=device, dtype=torch.float32)
    VT = V.T.contiguous()  # k × D
    cnt = 0
    for d in others:
        v = flatten_by_keys(d, keys, device=device).float()  # D
        s = torch.abs(VT @ v)  # k
        totals += s
        cnt += 1
    if cnt == 0:
        return [0.0 for _ in range(k)]
    means = (totals / float(cnt)).tolist()
    return [float(x) for x in means]
