from copy import copy, deepcopy

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_, parameters_to_vector, vector_to_parameters
from typeguard import typechecked
from typing import Dict, List, Union
from FedUnlearner.models import AllCNN, ResNet18
# from utils import meter
from .pga_model import get_model


@typechecked
def get_distance(model1: torch.nn.Module,
                 model2: torch.nn.Module):
    """
    returns L2 distance (norm) between two models
    """
    with torch.no_grad():
        model1_flattened = parameters_to_vector(model1.parameters())
        model2_flattened = parameters_to_vector(model2.parameters())
        # 使用真正的 L2 范数，后续所有阈值和步长都在同一量纲上
        distance = torch.norm(model1_flattened - model2_flattened)
    return distance


@typechecked
def get_ref_vec(global_model: torch.nn.Module,
                forget_client_model: torch.nn.Module,
                num_clients: int):
    """
    global_model : Global Model trained on all clients
    unlearn_client_model : Last Model returned by forget client to global server
    num_client: Number of clients participating in Federated Learningfederated-unlearning/FedUnlearner/baselines/pga/pga_utils.py
    """

    # global_param = copy.deepcopy(global_model)
    global_param = global_model.parameters()
    # forget_client_param = copy.deepcopy(forget_client_model)
    forget_client_param = forget_client_model.parameters()

    model_ref_vec = num_clients / (num_clients - 1) * parameters_to_vector(
        global_param) - 1 / (num_clients - 1) * parameters_to_vector(forget_client_param)

    return model_ref_vec


@typechecked
def get_model_ref(global_model: torch.nn.Module,
                  forget_client_model: torch.nn.Module,
                  num_clients: int,
                  model: str,
                  dataset: str,
                  num_classes: int,
                  pretrained: bool,
                  device: str):
    model_ref_vec = get_ref_vec(global_model=global_model,
                                forget_client_model=forget_client_model,
                                num_clients=num_clients,)

    model_ref = get_model(
        model=model, num_classes=num_classes, pretrained=pretrained,
        device=device, dataset=dataset)
    vector_to_parameters(model_ref_vec, model_ref.parameters())
    return model_ref


def get_threshold(model_ref: torch.nn.Module,
                  model: str,
                  dataset: str,
                  num_classes: int,
                  pretrained: bool,
                  device: str):
    dist_ref_random_lst = []
    for _ in range(10):
        random_model = get_model(model=model,
                                 num_classes=num_classes,
                                 pretrained=pretrained,
                                 device=device, dataset=dataset)
        dist_ref_random_lst.append(get_distance(model_ref, random_model).cpu())

    # threshold 现在就是 L2 半径，而不是“距离平方”
    threshold = float(np.mean(dist_ref_random_lst) / 3.0)
    return threshold


def unlearn(
    global_model: torch.nn.Module,
    forget_client_model: torch.nn.Module,
    model_ref: torch.nn.Module,
    distance_threshold: float,
    loader: torch.utils.data.DataLoader,
    threshold: float,
    optimizer_name: str,
    device: str,
    clip_grad: float = 1,
    epochs: int = 1,
    lr: float = 0.01,
    alpha: float = 1.0,
):
    """

    """

    print(f"Performing PGA unlearning for {epochs} rounds (alpha={alpha}, lr={lr})")
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(),
                                    lr=lr,
                                    momentum=0.9)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

    loss_fn = torch.nn.CrossEntropyLoss()
    global_model.train()

    flag = False
    for epoch in range(epochs):
        if flag:
            break
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = global_model(data)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            # 这里只做“标准”的梯度上升，alpha 不再直接缩放梯度，
            # 而是完全通过 distance_threshold 控制遗忘强度。
            # 这样可以避免因为梯度裁剪导致的“二极管”现象：
            # alpha 跨过某个值之后，梯度总被裁剪成同一大小，效果几乎不变。
            loss = -loss  # negate the loss for gradient ascent
            loss.backward()
            if clip_grad > 0:
                clip_grad_norm_(global_model.parameters(), clip_grad)
            optimizer.step()

            with torch.no_grad():
                # 投影到以 model_ref 为中心、半径 = threshold 的 L2 球内
                distance = get_distance(global_model, model_ref)
                if distance > threshold:
                    dist_vec = parameters_to_vector(
                        global_model.parameters()
                    ) - parameters_to_vector(model_ref.parameters())
                    dist_vec = dist_vec / torch.norm(dist_vec) * threshold
                    proj_vec = parameters_to_vector(
                        model_ref.parameters()) + dist_vec
                    vector_to_parameters(proj_vec, global_model.parameters())
                    distance = get_distance(global_model, model_ref)

            distance_ref_forget_client = get_distance(
                global_model, forget_client_model)

            if distance_ref_forget_client >= distance_threshold:
                flag = True
                break

    return global_model
