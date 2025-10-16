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
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh.T
    return V[:, :k]

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
    w = torch.sqrt(torch.cat([fisher[k].to(device).view(-1) for k in keys]) + 1e-12)
    X = torch.stack([flatten_by_keys(d, keys, device=device) for d in deltas], dim=0)
    return X * w.unsqueeze(0)

def rho_values_keys(V: torch.Tensor, other_deltas: list, keys: list) -> list:
    if len(other_deltas) == 0:
        return [0.0 for _ in range(V.shape[1])]
    D = torch.stack([flatten_by_keys(d, keys, device=V.device) for d in other_deltas], dim=0)
    return [float(torch.abs(D @ V[:, i]).mean().item()) for i in range(V.shape[1])]
