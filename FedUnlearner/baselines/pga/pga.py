import torch
from typeguard import typechecked
from typing import Dict, List, Union
import time
from copy import deepcopy
import os
from tqdm import tqdm

from .pga_utils import get_model_ref, get_threshold, unlearn, get_distance
from FedUnlearner.fed_learn import train_local_model
from FedUnlearner.utils import average_weights

"""
PGA
Source:
https://arxiv.org/pdf/2207.05521.pdf &
https://proceedings.mlr.press/v222/nguyen24a/nguyen24a.pdf
"""


def fed_avg(w):
    """
    Returns the average of the weights.
    """
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


@typechecked
def run_pga(global_model: torch.nn.Module,
            weights_path: str,
            clientwise_dataloaders: Dict[int, torch.utils.data.DataLoader],
            num_clients: int,
            forget_client: List[int],
            optimizer_name: str,
            lr: float,
            model: str,
            dataset: str,
            num_classes: int,
            pretrained: bool,
            num_training_iterations: int,
            num_local_epochs: int,
            device: str,
            num_unlearn_rounds: int = 1,
            num_post_training_rounds: int = 1,
            alpha: float = 1.0,) -> torch.nn.Module:

    forget_client_model_path = os.path.join(
        weights_path, f"iteration_{num_training_iterations - 1}", f"client_{forget_client[0]}.pth")
    forget_client_model = deepcopy(global_model)
    forget_client_model.load_state_dict(torch.load(forget_client_model_path))
    start_time = time.time()
    # Reference Model
    model_ref = get_model_ref(global_model=global_model,
                              forget_client_model=forget_client_model,
                              num_clients=num_clients,
                              model=model,
                              dataset=dataset,
                              num_classes=num_classes,
                              pretrained=pretrained,
                              device=device)
    # [FIX] 不要使用 get_threshold (基于随机模型距离太大)，而是基于当前模型间的距离
    # base_distance 是 global_model 和 forget_client_model 之间的距离
    # 我们希望模型在 unlearn 过程中偏离 model_ref 的距离不要超过这个尺度太多
    base_distance = float(get_distance(global_model, forget_client_model).item())
    
    # 设定投影半径。通常取 base_distance 的一部分或倍数。
    # 经验值：设置为 base_distance 的 0.5 到 1.0 倍比较安全，能保证模型不崩坏。
    # 如果设得太小，模型变动不了；设得太大，模型会崩。
    threshold = base_distance * 0.5 
    
    print(f"[PGA-FIX] Replaced random-init threshold with dynamic threshold: {threshold:.4f} (base_dist={base_distance:.4f})")

    # —— 自适应的距离阈值（替代之前写死的 2.2）——
    # alpha 用来控制“离开遗忘客户端”的尺度与强度
    # 注意：停止条件和阈值现在都用同一对模型(global, forget) 的距离，
    # 避免原来那种“二极管”式跳变。
    ALPHA = float(alpha)
    base_distance = float(get_distance(global_model, forget_client_model).item())
    # alpha = 0 时：distance_threshold = base_distance（几乎不走）
    # alpha 变大：允许相对于当前距离再走 alpha 倍，行为更平滑
    distance_threshold = (1.0 + ALPHA) * base_distance
    
    # [RESTORED] 将控制权交还给 main.py 参数
    # 之前成功的经验是：Threshold 约为 base_distance 的 0.35 倍
    # 所以现在 threshold = base_distance * alpha
    threshold = base_distance * ALPHA
    
    print(f"[PGA-CONFIG] Base Dist={base_distance:.4f}, Alpha={ALPHA}, Threshold={threshold:.4f}")
    print(f"[PGA-CONFIG] Unlearn LR={lr}")

    unlearned_global_model = unlearn(
        global_model=global_model,
        forget_client_model=forget_client_model,
        model_ref=model_ref,
        distance_threshold=distance_threshold,
        loader=clientwise_dataloaders[forget_client[0]],
        optimizer_name=optimizer_name,
        device=device,
        threshold=threshold,
        clip_grad=5,
        epochs=num_unlearn_rounds,
        lr=lr,
        alpha=ALPHA,
    )

    total_time = time.time() - start_time

    ######################## post train ############################

    finetuned_model = deepcopy(unlearned_global_model)


    # —— finetune 学习率与训练 lr 解耦，避免把遗忘效果洗回去 ——
    # 若用户训练 lr 很大（如 0.1），这里收敛会过猛；默认收敛到不超过 1e-2。
    # 如需自定义，可在后续扩展成入参；当前保持向后兼容。
    finetune_lr = lr if lr <= 0.01 else 0.01


    for round in range(num_post_training_rounds):
        print(f"Finetuning PGA unlearned Model: {round}")

        new_local_weights = []
        chosen_clients = [i for i in range(0, num_clients)
                          if i not in forget_client]
        for client_idx in tqdm(chosen_clients):

            client_dataloader = clientwise_dataloaders[client_idx]
            client_model = deepcopy(finetuned_model)

            if client_idx == forget_client[0]:
                continue
            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(client_model.parameters(), lr=finetune_lr)
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(client_model.parameters(), lr=finetune_lr)
            else:
                raise ValueError(f"Optimizer {optimizer_name} not supported")

            loss_fn = torch.nn.CrossEntropyLoss()
            train_local_model(model=client_model, dataloader=client_dataloader,
                              loss_fn=loss_fn, optimizer=optimizer, num_epochs=num_local_epochs,
                              device=device)
            new_local_weights.append(client_model.state_dict())

        # server aggregation
        updated_global_weights = fed_avg(new_local_weights)
        finetuned_model.load_state_dict(updated_global_weights)

    return finetuned_model
