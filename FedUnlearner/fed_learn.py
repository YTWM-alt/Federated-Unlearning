import torch
from typeguard import typechecked
from typing import Tuple, Dict

from FedUnlearner.utils import average_weights
from copy import deepcopy
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
import os

import sys

@typechecked
def fed_train(num_training_iterations: int, test_dataloader: torch.utils.data.DataLoader, 
              clientwise_dataloaders: dict[int, torch.utils.data.DataLoader], weights_path : str,
              global_model: torch.nn.Module, num_local_epochs: int, lr: float, optimizer_name: str, device: str = 'cpu'):
    """
    """
    
    if os.path.exists(weights_path):
        shutil.rmtree(weights_path)
    
    os.makedirs(weights_path)

    # savve the initial model
    torch.save(global_model.state_dict(), os.path.join(weights_path, "initial_model.pth"))
    global_model.to(device)
    global_model.train()
    
    clients = clientwise_dataloaders.keys()

    client_contributions = {}
    for client_idx in clients:
        client_contributions[client_idx] = []

    best_acc = 0
    patience = 10
    counter = 0

    for iteration in range(num_training_iterations):
        print(f"Global Iteration: {iteration}")
        iteration_weights_path = os.path.join(weights_path, f"iteration_{iteration}")
        os.makedirs(iteration_weights_path, exist_ok = True)
        for client_idx in clients:
            print(f"Client: {client_idx}")
            client_dataloader = clientwise_dataloaders[client_idx]
            client_model = deepcopy(global_model)
            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)  # create optimizer
            elif optimizer_name == 'sgd':
                    optimizer = torch.optim.SGD(
                        client_model.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=5e-4,
                        nesterov=True
                    )
            else:
                raise ValueError(f"Optimizer {optimizer_name} not supported")
            
            # === Cosine 学习率调度（按 batch 衰减）===
            # T_max = 本地总步数；eta_min 可按需改成 lr*0.01 或 0.0，这里给个温和下限
            _steps_per_epoch = max(1, len(client_dataloader))
            _tmax = max(1, num_local_epochs * _steps_per_epoch)
            scheduler = CosineAnnealingLR(optimizer, T_max=_tmax, eta_min=lr*0.01)

            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

            train_local_model(
                model=client_model,
                dataloader=client_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                num_epochs=num_local_epochs,
                device=device,
                scheduler=scheduler,                 # 新增
            )
            torch.save(client_model.state_dict(), os.path.join(iteration_weights_path, f"client_{client_idx}.pth"))

            # test_acc_client, test_loss_client = test_local_model(client_model, test_dataloader, loss_fn, device)
            # print(f"Test Accuracy for client {client_idx} : {test_acc_client*100}, Loss : {test_loss_client}")
            

        # update gloal model
        # 统计本轮参与客户端的样本数并做加权聚合
        # 统计本轮参与客户端的样本数
        samples_per_client = {}
        for client_idx in clients:
            weight_file = os.path.join(iteration_weights_path, f"client_{client_idx}.pth")
            if os.path.exists(weight_file):
                samples_per_client[client_idx] = len(clientwise_dataloaders[client_idx].dataset)

        # 幂次加权 + 下限（beta=0.5, floor=5%）
        updated_global_weights = average_weights(
            iteration_weights_path, device=device,
            samples_per_client=samples_per_client,
            weight_exponent=0.5,         # 0=等权, 1=标准FedAvg, 0.5=折中
            min_weight_floor=0.05        # 每个客户端至少获得均匀权重的5%
        )
        global_model.load_state_dict(updated_global_weights)

        torch.save(global_model.state_dict(), os.path.join(iteration_weights_path, "global_model.pth"))

        

        # evaluate global model
        test_acc_global, test_loss_global = test_local_model(global_model, test_dataloader, loss_fn, device)
        print(f"Test Accuracy for global model : {test_acc_global*100}, Loss : {test_loss_global}")

        # 早停机制（双条件：达到目标或长时间未提升）
        target_acc = 1  # 提前停止的目标精度
        if test_acc_global >= target_acc:
            print(f"Early stopping: accuracy {test_acc_global*100:.2f}% reached target {target_acc*100:.0f}%")
            break

        if test_acc_global > best_acc:
            best_acc = test_acc_global
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered due to no improvement")
                break

    torch.save(global_model.state_dict(), os.path.join(weights_path, "final_model.pth"))

    return global_model

            

@typechecked
def train_local_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn,
                      optimizer: torch.optim.Optimizer, num_epochs: int, device: str = 'cpu',
                      scheduler=None):
    model = model.to(device)
    model.train()


    for iter in range(num_epochs):
        tqdm_iterator = tqdm(dataloader, desc = f"Epoch: {iter}")
        for images, labels in tqdm_iterator:
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            log_probs = model(images)
            loss = loss_fn(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            # 余弦调度按 batch 步进，适合本地 epoch 较小的联邦场景
            if scheduler is not None:
                scheduler.step()
            # 可选：把当前 LR 打到进度条里便于观察
            try:
                cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                tqdm_iterator.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.5f}"})
            except Exception:
                tqdm_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    return model.state_dict()

@torch.no_grad()
@typechecked
def test_local_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                     loss_fn = None, device: str = 'cpu'):
    model = model.to(device)
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            if loss_fn is not None:
                test_loss += loss_fn(log_probs, labels).item() * labels.size(0)
            pred = log_probs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    return test_acc, test_loss

@torch.no_grad()
@typechecked
def get_classwise_accuracy(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                           num_classes: int, device: str = 'cpu') -> Dict[int, float]:
    model = model.to(device)
    model.eval()
    classwise_correct = {}
    classwise_nums = {}
    for i in range(num_classes):
        classwise_correct[i] = 0
        classwise_nums[i] = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            pred = log_probs.argmax(dim=1, keepdim=True)

            for i in range(num_classes):
                mask = (labels == i)
                cls_preds = pred.view_as(labels)[mask]
                cls_labels = labels[mask]
                classwise_correct[i] += cls_preds.eq(cls_labels).sum().item()
                classwise_nums[i] += mask.sum().item()
    classwise_acc = {}
    for i in range(num_classes):
        classwise_acc[i] = classwise_correct[i] / classwise_nums[i]
    return classwise_acc


@typechecked
def get_clientwise_accuracy(model: torch.nn.Module, clientwise_dataloaders: dict[int, torch.utils.data.DataLoader],
                            device: str = 'cpu') -> dict[int, float]:
    """
    Get the clientwise accuracy.
    """
    results = {}
    for client_id, dataloader in clientwise_dataloaders.items():
        test_acc, _ = test_local_model(model = model, dataloader = dataloader, device = device)
        results[client_id] = test_acc*100
    return results


def get_performance(model, test_dataloader, clientwise_dataloader, num_classes, device):
    results = {}
    test_acc, test_loss = test_local_model(model = model, dataloader = test_dataloader, device=device)
    results['test_acc'] = test_acc

    results['classwise_acc'] = get_classwise_accuracy(model = model, dataloader = test_dataloader, 
                                                      num_classes = num_classes, device = device)
    results['clientwise_acc'] = get_clientwise_accuracy(model = model, clientwise_dataloaders = clientwise_dataloader,
                                                        device = device)

    return results
