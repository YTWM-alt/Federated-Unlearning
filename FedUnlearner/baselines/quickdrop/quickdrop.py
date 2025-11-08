
import os
import math
from copy import deepcopy
from typing import Dict, Tuple, List

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from FedUnlearner.models import AllCNN, ResNet18, SmallCNN
from FedUnlearner.utils import get_labels_from_dataset  # 按类统计/索引  :contentReference[oaicite:3]{index=3}

def _build_model(args, num_classes: int) -> nn.Module:
    """与 main.py 中保持同构：根据 args.model/dataset 构建模型"""
    if args.model == 'allcnn':
        if args.dataset == 'mnist':
            return AllCNN(num_classes=num_classes, num_channels=1)
        else:
            return AllCNN(num_classes=num_classes)
    elif args.model == 'resnet18':
        if args.dataset == 'mnist':
            return ResNet18(num_classes=num_classes, pretrained=args.pretrained, num_channels=1)
        else:
            return ResNet18(num_classes=num_classes, pretrained=args.pretrained)
    elif args.model == 'smallcnn':
        in_ch = 1 if args.dataset == 'mnist' else 3
        return SmallCNN(num_channels=in_ch, num_classes=num_classes)
    else:
        raise ValueError("Invalid model name for QuickDrop")


@torch.no_grad()
def _class_index_map(dataset, num_classes: int) -> List[List[int]]:
    """返回每个类别对应的样本索引列表"""
    labels = get_labels_from_dataset(dataset)
    buckets: List[List[int]] = [[] for _ in range(num_classes)]
    for idx, y in enumerate(labels):
        buckets[int(y)].append(idx)
    return buckets


def _grad_list(
    model: nn.Module,
    loss: torch.Tensor,
    *,
    create_graph: bool = False,
    allow_unused: bool = True,
) -> List[torch.Tensor]:
    """计算对所有可训练参数的梯度向量列表。
    用于“合成批”时必须 create_graph=True，使梯度对输入图像可反传。"""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss, params,
        # 关键修复：当需要二阶图（create_graph=True）时，也要保留图，
        # 否则后续对 L.backward() 会用到已被释放的中间张量
        retain_graph=create_graph,
        create_graph=create_graph, allow_unused=allow_unused)
    out: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p))
        else:
            out.append(g if create_graph else g.detach())
    return out


def _normalize_flat(grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """按张量范数归一化（避免不同层量纲差异）。"""
    out = []
    eps = 1e-12
    for g in grads:
        out.append(g / (g.norm() + eps))
    return out


def _match_loss(ga: List[torch.Tensor], gb: List[torch.Tensor]) -> torch.Tensor:
    """MSE( normalize(grad_real), normalize(grad_syn) )"""
    assert len(ga) == len(gb)
    loss = torch.zeros(1, device=ga[0].device)
    for a, b in zip(_normalize_flat(ga), _normalize_flat(gb)):
        loss = loss + F.mse_loss(a, b)
    return loss


def _distill_dc(
    args,
    client_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    global_model: nn.Module,
    cid: int = -1,               # 仅用于日志打印
) -> Tuple[TensorDataset, torch.Tensor, torch.Tensor]:
    """
    DC/gradient-matching 蒸馏：
      - 为每类分配 n_syn = floor(n_real * args.qd_scale)，至少 1
      - 以真实 batch 的梯度作为 target，优化合成图像使其梯度匹配
      - 返回 (TensorDataset(images_syn, labels_syn), images_syn, labels_syn)
    """
    model = deepcopy(global_model).to(device)
    model.train(False)

    # 1) 按类分桶 + 计算每类配额
    base_ds = client_loader.dataset
    buckets = _class_index_map(base_ds, num_classes)
    per_class = []
    total_syn = 0
    for c in range(num_classes):
        n_real = len(buckets[c])
        n_syn = max(1, int(math.floor(n_real * float(args.qd_scale)))) if n_real > 0 else 0
        per_class.append(n_syn)
        total_syn += n_syn

    if total_syn == 0:
        # 该客户端没有样本（极端划分），退化：直接返回空数据集
        empty = TensorDataset(torch.empty(0, *next(iter(client_loader))[0].shape[1:], device='cpu'),
                              torch.empty(0, dtype=torch.long))
        return empty, None, None

    # 2) 初始化合成图像：从真实样本中抽取 n_syn 个拷贝（更稳）
    #    保持 requires_grad=True，以便优化“图像像素”
    x0, y0 = next(iter(client_loader))
    C, H, W = x0.shape[1:]
    images_syn = []
    labels_syn = []
    for c in range(num_classes):
        idxs = buckets[c]
        n_syn = per_class[c]
        if n_syn <= 0:
            continue
        # 随机抽取 n_syn 个真实样本作为起点
        choice = torch.randint(low=0, high=len(idxs), size=(n_syn,)) if len(idxs) > 0 else torch.empty(0,dtype=torch.long)
        for k in range(n_syn):
            ridx = idxs[ int(choice[k]) ]
            # 通过 DataLoader 的底层 Dataset 取原图
            xi, yi = base_ds[ridx]
            images_syn.append(xi)
            labels_syn.append(torch.tensor(int(yi), dtype=torch.long))
    images_syn = torch.stack(images_syn, dim=0).to(device).clone().detach().requires_grad_(True)
    labels_syn = torch.stack(labels_syn, dim=0).to(device)

    # 3) 蒸馏外循环：梯度匹配（real vs syn）
    opt_img = torch.optim.SGD([images_syn], lr=float(args.qd_lr_img), momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 为了稳定，真实 batch 来自该客户端原 Loader
    real_iter = iter(client_loader)
    t0 = time.time()
    for it in range(int(args.qd_syn_steps)):
        try:
            xr, yr = next(real_iter)
        except StopIteration:
            real_iter = iter(client_loader)
            xr, yr = next(real_iter)
        xr, yr = xr.to(device, non_blocking=True), yr.to(device, non_blocking=True)
        # 子采样到 qd_batch_real
        if xr.size(0) > args.qd_batch_real:
            xr = xr[:args.qd_batch_real]
            yr = yr[:args.qd_batch_real]

        # 真实梯度（不需要二阶图）
        model.zero_grad(set_to_none=True)
        loss_real = criterion(model(xr), yr)
        g_real = _grad_list(model, loss_real, create_graph=False)

        # 合成梯度（需要对“图像变量”反传 → 必须 create_graph=True）
        idx_syn = torch.randperm(images_syn.size(0), device=device)[:min(images_syn.size(0), args.qd_batch_syn)]
        xs, ys = images_syn[idx_syn], labels_syn[idx_syn]
        # 前向 → 损失 → 对“网络参数”的梯度（梯度需带图，才能对 xs 反传）
        loss_syn = criterion(model(xs), ys)
        g_syn = _grad_list(model, loss_syn, create_graph=True)

        # 匹配损失 w.r.t. 网络参数梯度；对“图像变量”反传
        L = _match_loss(g_real, g_syn)
        opt_img.zero_grad(set_to_none=True)
        L.backward()
        opt_img.step()


        # ---- 进度与内存保洁（每 qd_log_interval 步）----
        if (it + 1) % max(1, getattr(args, "qd_log_interval", 50)) == 0:
            elapsed = time.time() - t0
            try:
                loss_val = float(L.item())
            except Exception:
                loss_val = float('nan')
            print(f"[QuickDrop][cid={cid}] it={it+1}/{args.qd_syn_steps}  match_loss={loss_val:.4f}  Δt={elapsed:.1f}s")
            t0 = time.time()
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # 4) 得到合成集（返回 TensorDataset；可选落盘以复现）
    ds_syn = TensorDataset(images_syn.detach().cpu(), labels_syn.detach().cpu())
    return ds_syn, images_syn.detach().cpu(), labels_syn.detach().cpu()


def _local_train_on_syn(
    args,
    global_model: nn.Module,
    syn_ds: TensorDataset,
    device: torch.device,
) -> nn.Module:
    """仅用合成数据进行本地训练；返回本轮客户端更新后的模型副本。"""
    model = deepcopy(global_model).to(device)
    if len(syn_ds) == 0:
        return model
    epochs = int(args.qd_local_epochs) if args.qd_local_epochs else int(args.num_local_epochs)
    loader = DataLoader(syn_ds, batch_size=min(len(syn_ds), max(1, args.qd_batch_syn)), shuffle=True)
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
    return model


def _state_dict_average(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    """等权平均（兼容你项目里已有的 FedAvg 聚合风格）。"""
    assert len(models) > 0
    with torch.no_grad():
        avg = {k: torch.zeros_like(v) for k, v in models[0].state_dict().items()}
        for m in models:
            sd = m.state_dict()
            for k in avg.keys():
                if torch.is_floating_point(sd[k]):
                    avg[k].add_(sd[k])
                else:
                    # buffer/整型，直接用第一个
                    if avg[k].abs().sum() == 0:
                        avg[k].copy_(sd[k])
        for k in avg.keys():
            if torch.is_floating_point(avg[k]):
                avg[k].div_(float(len(models)))
        return avg


def run_quickdrop(
    *,
    args,
    global_model: nn.Module,
    clientwise_dataloaders: Dict[int, DataLoader],
    test_dataloader: DataLoader,
    num_classes: int,
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """
    作为 baseline 的对外入口（与其它 baselines 保持一致风格）：
      - 输入：当前全局模型 + 客户端 DataLoader + 测试集
      - 输出：完成 num_training_iterations 轮后的全局模型 + 侧信息（比如保存合成集的路径）
    """
    device = torch.device(device)
    G = deepcopy(global_model).to(device)
    G.train()

    # 输出目录（仿照 retraining 的组织方式）
    exp_root = os.path.abspath(os.path.join(args.exp_path, args.exp_name))
    out_root = os.path.join(exp_root, "quickdrop")
    os.makedirs(out_root, exist_ok=True)
    if args.qd_save_affine:
        affine_root = os.path.join(exp_root, args.qd_affine_dir)
        os.makedirs(affine_root, exist_ok=True)

    for round_idx in range(int(args.num_training_iterations)):
        # 所有（或前 num_participating_clients 个）客户端参与
        client_ids = list(clientwise_dataloaders.keys())
        if args.num_participating_clients and args.num_participating_clients > 0:
            client_ids = client_ids[: args.num_participating_clients]

        local_models = []
        # === Client loop ===
        for cid in client_ids:
            loader = clientwise_dataloaders[cid]
            # 先蒸馏得到合成集
            syn_ds, syn_img, syn_lab = _distill_dc(
                args=args,
                client_loader=loader,
                num_classes=num_classes,
                device=device,
                global_model=G,
                cid=cid,
            )
            # 可选：落盘合成集，便于复现实验（“affine dataset”）
            if args.qd_save_affine and syn_img is not None:
                torch.save(
                    {"images": syn_img, "labels": syn_lab},
                    os.path.join(affine_root, f"round_{round_idx}_client_{cid}.pt"),
                )
            # 只用合成集进行本地微调
            m_loc = _local_train_on_syn(args=args, global_model=G, syn_ds=syn_ds, device=device)
            local_models.append(m_loc)

        # === Server aggregation ===
        avg_state = _state_dict_average(local_models)
        G.load_state_dict(avg_state)

        # 可选：保存该轮全局
        iter_dir = os.path.join(out_root, f"iteration_{round_idx}")
        os.makedirs(iter_dir, exist_ok=True)
        torch.save(G.state_dict(), os.path.join(iter_dir, "global_model.pth"))

    # final
    torch.save(G.state_dict(), os.path.join(out_root, "final_model.pth"))
    info = {"weights_dir": out_root}
    return G.to("cpu"), info
