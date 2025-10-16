import torch
from typing import Dict
from .subspace import flatten_state_dict, state_dict_like

def projection_matrix(V_spec: torch.Tensor) -> torch.Tensor:
    if V_spec.numel() == 0:
        return None
    Q, _ = torch.linalg.qr(V_spec, mode='reduced')
    return torch.eye(Q.shape[0], device=Q.device) - Q @ Q.T

def apply_projection_to_update(delta: Dict[str, torch.Tensor],
                               P: torch.Tensor,
                               ref_like: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if P is None:
        return delta
    flat = flatten_state_dict(delta)
    flat_new = P @ flat
    return state_dict_like(flat_new, ref_like)
