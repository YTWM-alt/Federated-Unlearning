import torch, os, glob
from typing import List, Dict
from .fisher import empirical_fisher_diagonal
from .subspace import weighted_matrix_from_deltas, topk_right_singular_vectors, rho_values, split_subspaces, flatten_state_dict, state_dict_like

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
                     device: str = "cpu",
                     *,
                     # 新增：客户端 Fisher 端点或回调（任选其一）。两者都缺省时，将不触碰原始数据，回退为单位权重。
                     client_endpoints: Dict[int, object] = None,
                     compute_fisher_fn=None) -> Dict[str, torch.Tensor]:
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

    # === 改动点：禁止服务端直接用 dataloader 计算 Fisher ===
    # 优先通过“客户端端点/回调”在本地计算 Fisher，对服务端仅回传对角统计量
    fisher = None
    if client_endpoints is not None and target_id in client_endpoints:
        # 端点协议：.compute_fisher(model_state_dict=..., device=..., max_batches=...)
        fisher = client_endpoints[target_id].compute_fisher(
            model_state_dict=global_model.state_dict(),
            device=device,
            max_batches=fisher_batches
        )
    elif callable(compute_fisher_fn):
        # 自定义回调协议：compute_fisher_fn(target_id=..., model_state_dict=..., device=..., max_batches=...)
        fisher = compute_fisher_fn(
            target_id=target_id,
            model_state_dict=global_model.state_dict(),
            device=device,
            max_batches=fisher_batches
        )
    # 若没有端点/回调，则回退为单位权重（不触碰原始样本）
    if fisher is None:
        fisher = {k: torch.ones_like(v) for k, v in start_sd.items()}
    # 子空间
    Xw = weighted_matrix_from_deltas(target_deltas, fisher)
    V = topk_right_singular_vectors(Xw, k=rank_k)
    rhos = rho_values(V, [client_deltas[i] for i in others])
    tau = sorted(rhos)[len(rhos)//2] if tau_mode == "median" else sum(rhos)/len(rhos)
    V_spec, V_comm, _, _ = split_subspaces(V, rhos, tau)
    
# === [Theoretical Fix A2] 严谨的黎曼流形投影: 变换 -> 投影 -> 逆变换 ===
    # 1. 准备黎曼度量权重
    flat_f = flatten_state_dict(fisher).to(device)
    
    # [修复 1] 增大 epsilon，防止 Fisher 极小值导致的数值爆炸
    # 经验值：1e-5 到 1e-4 通常比较稳健
    epsilon = 1e-5 
    w = torch.sqrt(flat_f + epsilon)
    
    # [修复 2] 逆变换权重的计算与截断
    # 理论上是 1/w，但为了防止 w 极小导致 w_inv 极大，我们需要 clamp
    w_inv = 1.0 / w
    
    # [关键技巧] 限制放大的倍数。如果某参数 Fisher 很小，我们不应将其更新放大 10000 倍
    # 设定一个上限，比如 max_scale=100.0 或 1000.0
    w_inv = torch.clamp(w_inv, max=1000.0)

    # 2. 获取原始全局更新
    flat_updates_raw = torch.stack([flatten_state_dict(d).to(device) for d in client_deltas.values()], dim=0).mean(dim=0)

    # 3. 变换到黎曼流形
    flat_updates_weighted = flat_updates_raw * w

    # 4. 执行流形上的正交投影
    if V_spec.numel() > 0:
        Q, _ = torch.linalg.qr(V_spec, mode='reduced')
        spec_component = Q @ (Q.T @ flat_updates_weighted)
        
        # [可选] 检查一下 spec_component 的范数，如果太大说明可能有问题
        # print(f"Spec Norm: {torch.norm(spec_component)}")
        
        flat_projected_weighted = flat_updates_weighted - spec_component
    else:
        flat_projected_weighted = flat_updates_weighted

    # 5. 逆变换回参数空间
    flat_final = flat_projected_weighted * w_inv

    new_delta = state_dict_like(flat_final, start_sd)
    new_sd = {k: start_sd[k] + new_delta[k] for k in start_sd}
    return new_sd
