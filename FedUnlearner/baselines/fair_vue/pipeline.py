import torch, os, glob
from typing import List, Dict
from .fisher import empirical_fisher_diagonal
from .subspace import weighted_matrix_from_deltas, topk_right_singular_vectors, rho_values, split_subspaces, flatten_state_dict, state_dict_like
from .projection import projection_matrix, apply_projection_to_update

def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt

def delta_from_start(start: Dict[str, torch.Tensor], end: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: end[k] - start[k] for k in start.keys() if k in end}

def collect_client_deltas(weights_path: str, start_model: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    out = {}
    for p in glob.glob(os.path.join(weights_path, "client_*.pth")):
        cid = int(os.path.basename(p).split("_")[1].split(".")[0])
        sd = load_state_dict(p)
        out[cid] = delta_from_start(start_model, sd)
    return out

def fair_vue_unlearn(global_model: torch.nn.Module,
                     weights_path: str,
                     clientwise_dataloaders: Dict[int, torch.utils.data.DataLoader],
                     forget_clients: List[int],
                     rank_k: int = 8,
                     tau_mode: str = "median",
                     fisher_batches: int = 10,
                     device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    FAIR-VUE 无分簇版：Fisher加权+SVD特征方向+ρ划分+全局投影遗忘
    """
    global_model = global_model.to(device)
    start_sd = {k: v.detach().clone().cpu() for k, v in global_model.state_dict().items()}
    client_deltas = collect_client_deltas(weights_path, start_sd)
    target_id = forget_clients[0]
    others = [cid for cid in client_deltas if cid != target_id]
    target_deltas = [client_deltas[target_id]]
    # Fisher (固定随机性，确保完整流程与简化流程结果一致)
    import random, numpy as np
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if target_id in clientwise_dataloaders:
        fisher = empirical_fisher_diagonal(
            global_model,
            clientwise_dataloaders[target_id],
            device=device,
            max_batches=fisher_batches
        )
    else:
        fisher = {k: torch.ones_like(v) for k, v in start_sd.items()}
    # 子空间
    Xw = weighted_matrix_from_deltas(target_deltas, fisher)
    V = topk_right_singular_vectors(Xw, k=rank_k)
    rhos = rho_values(V, [client_deltas[i] for i in others])
    tau = sorted(rhos)[len(rhos)//2] if tau_mode == "median" else sum(rhos)/len(rhos)
    V_spec, V_comm, _, _ = split_subspaces(V, rhos, tau)
    # 投影全局更新
    flat_updates = torch.stack([flatten_state_dict(d) for d in client_deltas.values()], dim=0).mean(dim=0)
    global_delta = state_dict_like(flat_updates, start_sd)
    P = projection_matrix(V_spec)
    new_delta = apply_projection_to_update(global_delta, P, start_sd)
    new_sd = {k: start_sd[k] + new_delta[k] for k in start_sd}
    return new_sd
