import torch
import torchvision
import argparse
import json
import time
import os
from copy import deepcopy
import random
import numpy as np
import sys
from datetime import datetime
import re
import io


from FedUnlearner.utils import print_exp_details, print_clientwise_class_distribution
from FedUnlearner.data_utils import get_dataset, create_dirichlet_data_distribution, create_iid_data_distribution
from FedUnlearner.fed_learn import fed_train, get_performance
from FedUnlearner.unlearn import unlearn as unlearn_ours
from FedUnlearner.models import AllCNN, ResNet18, SmallCNN
from FedUnlearner.attacks.backdoor import create_backdoor_dataset, evaluate_backdoor_attack
from FedUnlearner.baselines import run_pga, run_fed_eraser
from FedUnlearner.attacks.mia import train_attack_model, evaluate_mia_attack
from FedUnlearner.attacks.poisoning import create_poisoning_dataset, evaluate_poisoning_attack

# 模型治疗
from FedUnlearner.baselines.fair_vue.healing import selective_kd_heal

# >>> FAIR-VUE: imports
from FedUnlearner.baselines.fair_vue.fisher import empirical_fisher_diagonal
from FedUnlearner.baselines.fair_vue.subspace import (
    weighted_matrix_from_deltas, topk_right_singular_vectors,
    rho_values, split_subspaces, flatten_state_dict, state_dict_like
)
from FedUnlearner.baselines.fair_vue.projection import projection_matrix

# === 日志目录与文件 ===
os.makedirs("./logs", exist_ok=True)
log_path = f"./logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class ProxyLog:
    """
    代理 stdout/stderr：
    1) 过滤 tqdm 进度条与 'Client: x' 行到日志文件
    2) 控制台照常显示（保留 '█'）
    3) 对外暴露 isatty/encoding/fileno 等，让 tqdm 识别为 TTY 支持 Unicode
    """
    def __init__(self, stream):
        self.stream = stream
        # 匹配进度条（百分比+两侧竖线）或常见吞吐字段；兼容 Unicode/ASCII 进度条
        self.re_bar = re.compile(
            r"(?:\r)?(?:(?=.*\d{1,3}%\|.+\|).*|.*\|[#█░▉▊▋▌▍▎▏]+\|.*|.*\b(?:it/s|s/it|ETA|elapsed|remaining)\b.*)"
        )
        # 匹配 "Client: 数字" 整行
        self.re_client = re.compile(r"^\s*Client:\s*\d+\s*$")
        self._buf = ""
        # 透传编码属性，避免 tqdm 误判
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def _should_skip(self, text: str) -> bool:
        # tqdm 更新常带 '\r' 且不一定含换行；遇到 '\r' 直接视为进度刷新，不落日志
        if "\r" in text:
            return True
        if self.re_bar.search(text):
            return True
        if self.re_client.search(text.strip()):
            return True
        return False

    def write(self, message: str):
        # 累积到换行，确保按“行”判断是否写日志，避免 tqdm 的碎片更新误判
        self._buf += message
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line_out = line + "\n"
            if not self._should_skip(line_out):
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line_out)
            # 控制台始终原样输出
            self.stream.write(line_out)

    def flush(self):
        # 刷出残留（通常是非进度的最后一行）
        if self._buf:
            if not self._should_skip(self._buf):
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(self._buf)
            self.stream.write(self._buf)
            self._buf = ""
        if hasattr(self.stream, "flush"):
            self.stream.flush()

    # —— 关键透传，让 tqdm 识别为“真 TTY”，从而使用 Unicode '█' —— #
    def isatty(self):
        try:
            return self.stream.isatty()
        except Exception:
            return True  # 宁可当作 TTY

    def fileno(self):
        try:
            return self.stream.fileno()
        except Exception:
            raise io.UnsupportedOperation("fileno")

    def writable(self):
        return True

    def __getattr__(self, name):
        # 其他一切属性/方法都代理给原始流
        return getattr(self.stream, name)

# 应用到 stdout / stderr
sys.stdout = ProxyLog(sys.stdout)
sys.stderr = ProxyLog(sys.stderr)

def get_accuracy_only(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    import torch
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

# >>> FAIR-VUE: round-wise helpers
import re

def _list_iteration_dirs(train_path: str):
    """返回按轮排序的 [(轮次,int_dir_path), ...]"""
    items = []
    if not os.path.isdir(train_path):
        return items
    for name in os.listdir(train_path):
        m = re.match(r'^iteration_(\d+)$', name)
        if m:
            items.append((int(m.group(1)), os.path.join(train_path, name)))
    items.sort(key=lambda x: x[0])
    return items

def _load_client_sd(iter_dir: str, cid: int):
    """加载某一轮的 client_{cid}.pth（返回state_dict或None）"""
    p = os.path.join(iter_dir, f"client_{cid}.pth")
    if not os.path.isfile(p):
        return None
    ckpt = torch.load(p, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt  # 默认就是state_dict

def _build_round_deltas(train_path: str, total_clients: int):
    """
    返回:
      round_client_deltas: dict[round_idx] -> dict[cid] -> Δ_i^t
    其中 Δ_i^t = client_{i,t} - mean_clients_{t-1}  (用上一轮客户端权重的均值近似 global_{t-1})
    round 从 1 开始有意义；0 轮没有上一轮。
    """
    rounds = _list_iteration_dirs(train_path)
    round_client_deltas = {}
    for r_idx in range(1, len(rounds)):
        r_prev, dir_prev = rounds[r_idx-1]
        r_cur,  dir_cur  = rounds[r_idx]

        # 先收集上一轮所有客户端权重，求均值 g_prev
        prev_states = []
        for cid in range(total_clients):
            sd_prev = _load_client_sd(dir_prev, cid)
            if sd_prev is not None:
                prev_states.append(sd_prev)
        if len(prev_states) == 0:
            continue
        # 逐 key 求均值
        keys = prev_states[0].keys()
        g_prev = {k: sum(sd[k] for sd in prev_states) / float(len(prev_states)) for k in keys}

        # 用 g_prev 做基线，构造本轮每个客户端的 Δ_i^t
        deltas_this_round = {}
        for cid in range(total_clients):
            sd_cur = _load_client_sd(dir_cur, cid)
            if sd_cur is None:
                continue
            delta = {k: sd_cur[k] - g_prev[k] for k in g_prev.keys() if k in sd_cur}
            deltas_this_round[cid] = delta

        if len(deltas_this_round) > 0:
            round_client_deltas[r_cur] = deltas_this_round
    return round_client_deltas



# create argument parser
parser = argparse.ArgumentParser(description='FedUnlearner')

# add arguments
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--exp_path", default="./experiments/", type=str)
parser.add_argument('--model', type=str, default='allcnn', choices=["allcnn", 'resnet18', 'smallcnn'],
                    help='model name')
parser.add_argument('--pretrained', type=bool,
                    default=False, help='use pretrained model')

parser.add_argument('--dataset', type=str, default='cifar10', choices=["mnist", "cifar10"],
                    help='dataset name')
parser.add_argument('--optimizer', type=str, default='adam', choices=["sgd", "adam"],
                    help='optimizer name')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_local_epochs', type=int,
                    default=1, help='number of local epochs')

parser.add_argument('--num_training_iterations', type=int, default=1,
                    help='number of training iterations for global model')
parser.add_argument('--num_participating_clients', type=int, default=-1, help='number of users participating in trainig, \
                                                                                    -1 if all are required to participate')

# baslines
parser.add_argument('--baselines', type=str, nargs="*", default=[], 
    choices=['pga', 'fed_eraser', 'fedfim', 'fair_vue'],
    help='baseline methods for unlearning')

# FAIR-VUE 参数
parser.add_argument('--fair_rank_k', type=int, default=16, help='SVD 主成分数')
parser.add_argument('--fair_tau_mode', type=str, default='median', choices=['median','mean'], help='ρ阈值模式')
parser.add_argument('--fair_fisher_batches', type=int, default=5, help='Fisher估计的批次数')
parser.add_argument('--fair_vue_debug', action='store_true',
                    help='打印 FAIR-VUE 关键中间量，便于排查是否真的执行')
parser.add_argument('--fair_erase_scale', type=float, default=0.25,
                    help='特异分量擦除强度 (0,1]，默认0.25，建议先小后大')
parser.add_argument('--skip_retraining', action='store_true',
                    help='跳过重训练基线（不执行 retain-only retraining 及其评估）')

# backdoor attack related arguments
parser.add_argument('--apply_backdoor', action='store_true',
                    help='apply backdoor attack')
parser.add_argument('--backdoor_position', type=str, default='corner', choices=["corner", "center"],
                    help='backdoor position')
parser.add_argument('--num_backdoor_samples_per_forget_client', type=int, default=10,
                    help='number of backdoor samples per forget client')
parser.add_argument('--backdoor_label', type=int,
                    default=0, help='backdoor label')

# membership inference attack related arguments
parser.add_argument('--apply_membership_inference', type=bool, default=False,
                    help='apply membership inference attack')
parser.add_argument('--attack_type', type=str, default='blackbox', choices=["blackbox", "whitebox"],
                    help='attack type')

# label posioning attack related arguments
parser.add_argument('--apply_label_poisoning', type=bool, default=False,
                    help='apply label poisoning attack')
parser.add_argument('--num_label_poison_samples', type=int, default=10,
                    help='number of label poisoning samples')

# provide indexes of clients which are to be forgotten, allow multiple clients to be forgotten
parser.add_argument('--forget_clients', type=int, nargs='+',
                    default=[0], help='forget clients')
parser.add_argument('--total_num_clients', type=int,
                    default=10, help='total number of clients')
parser.add_argument('--client_data_distribution', type=str, default='dirichlet',
                    choices=["dirichlet", "iid"], help='client data distribution')
parser.add_argument('--dampening_constant', type=float,
                    default=0.5, help='dampening constant')
parser.add_argument('--dampening_upper_bound', type=float,
                    default=0.5, help='dampening upper bound')
parser.add_argument('--ratio_cutoff', type=float,
                    default=0.5, help='ratio cutoff')
parser.add_argument('--device', type=str, default='cpu',
                    choices=["cpu", "cuda"], help='device name')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--verbose', type=bool, default=True, help='verbose')
parser.add_argument("--num_workers", type=int, default=32,
                    help="number of workers for data loading")

# create argument parser ...
parser.add_argument('--unlearn_only', action='store_true',
                    help='跳过联邦训练，直接在已有 full_training 上执行遗忘/治疗/评测')
parser.add_argument('--full_training_dir', type=str, default='',
                    help='已有的 full_training 目录（含 iteration_*/client_*.pth 和 final_model.pth）')
parser.add_argument('--global_ckpt', type=str, default='',
                    help='可选：显式指定要加载的全局模型权重路径（.pth）')

# FedFIM 参数
parser.add_argument("--fim_max_batches", type=int, default=2)
parser.add_argument("--fim_max_passes", type=int, default=1)
parser.add_argument("--fim_mode", type=str, default="prob", choices=["prob", "label"])
parser.add_argument("--fim_topk", type=int, default=None)
parser.add_argument("--fim_ratio_cutoff", type=float, default=1.0)
parser.add_argument("--fim_gamma", type=float, default=0.2)
parser.add_argument("--fim_upper_bound", type=float, default=1.0)
parser.add_argument("--finetune_epochs", type=int, default=0)
parser.add_argument("--finetune_lr", type=float, default=1e-3)
parser.add_argument("--finetune_wd", type=float, default=0.0)
parser.add_argument("--global_weight", type=str, default="")
parser.add_argument("--output_weight_path", type=str, default="")


# ==== HEAL / 模型治疗参数 ====
parser.add_argument('--heal', action='store_true',
                    help='开启治疗阶段（FAIR-VUE 擦除后运行 healing）')
parser.add_argument('--heal_steps', type=int, default=80,
                    help='治疗步数（迭代批次数）')
parser.add_argument('--heal_lr', type=float, default=1e-4,
                    help='治疗学习率')
parser.add_argument('--heal_T', type=float, default=2.0,
                    help='KD 蒸馏温度')
parser.add_argument('--heal_lambda_kd', type=float, default=0.3,
                    help='KD loss 权重')
parser.add_argument('--heal_lambda_ortho', type=float, default=1e-3,
                    help='正交惩罚项权重（抑制回流到被遗忘子空间）')
parser.add_argument('--heal_teacher', type=str, default='post',
                    choices=['pre','post'],
                    help='选择 teacher：pre=遗忘前全局模型，post=遗忘后模型（默认，避免把目标知识拉回）')
parser.add_argument('--no_heal_grad_proj', action='store_true',
                    help='关闭梯度正交投影（默认开启）')
parser.add_argument('--heal_shuffle', action='store_true',
                    help='治疗 DataLoader 是否 shuffle（默认 False 更可复现）')
parser.add_argument('--heal_batch_size', type=int, default=None,
                    help='治疗 batch_size；不填则用训练时 batch_size')



if __name__ == "__main__":

    args = parser.parse_args()
    # ---- 旧版 Legacy-Unlearn 总开关（默认关）----
    RUN_LEGACY_UNLEARN = False
    weights_path = os.path.abspath(os.path.join(args.exp_path, args.exp_name))

    print_exp_details(args)
    summary = {}
    # get the dataset
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset)

    # create client groups
    client_groups = None

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.client_data_distribution == 'dirichlet':
        clientwise_dataset = create_dirichlet_data_distribution(train_dataset,
                                                                num_clients=args.total_num_clients, num_classes=num_classes, alpha=0.5)
    elif args.client_data_distribution == 'iid':
        clientwise_dataset = create_iid_data_distribution(train_dataset, num_clients=args.total_num_clients,
                                                          num_classes=num_classes)
    else:
        raise "Invalid client data distribution"

    # print the clientwise class distribution
    print_clientwise_class_distribution(clientwise_dataset, num_classes)


    if args.num_participating_clients > 1:
        print(
            f"Cutting of num participating client to: {args.num_participating_clients}")
        clientwise_dataset = {i: clientwise_dataset[i] for i in range(
            args.num_participating_clients)}
        print("Clientwise distribution after cutting: ")
        print_clientwise_class_distribution(clientwise_dataset, num_classes)
    # get the forget client

    if len(args.forget_clients) > 1:
        raise "Only one client forgetting supported at the moment."
    forget_client = args.forget_clients[0]

    if args.apply_backdoor:
        backdoor_dataset = None
        backdoor_pixels = None
        image_size = 224
        patch_size = 30
        if args.backdoor_position == 'corner':
            # [top left corner of patch, bottom right corner of patch]
            backdoor_pixels = [(0, 0), (patch_size, patch_size)]
        elif args.backdoor_position == 'center':
            backdoor_pixels = [(image_size//2 - patch_size//2, image_size//2 - patch_size//2),
                               (image_size//2 + patch_size//2, image_size//2 + patch_size//2)]
        else:
            raise "Invalid backdoor position"

        print(
            f"Size of client dataset before backdoor ingestion: {len(clientwise_dataset[args.forget_clients[0]])}")
        clientwise_dataset, backdoor_context = create_backdoor_dataset(clientwise_dataset=clientwise_dataset,
                                                                       forget_clients=args.forget_clients,
                                                                       backdoor_pixels=backdoor_pixels,
                                                                       backdoor_label=args.backdoor_label,
                                                                       num_samples=args.num_backdoor_samples_per_forget_client
                                                                       )

        print(
            f"Size of client dataset after backdoor ingestion: {len(clientwise_dataset[args.forget_clients[0]])}")

    if args.apply_label_poisoning:
        clientwise_dataset, poisoning_context = create_poisoning_dataset(clientwise_dataset=clientwise_dataset,
                                                                         forget_clients=args.forget_clients,
                                                                         test_split=0.2,
                                                                         num_poisoning_samples=args.num_label_poison_samples)

    # create dataloaders for the clients
    clientwise_dataloaders = {}
    for client_id, client_dataset in clientwise_dataset.items():
        print(f"Creating data loader for client: {client_id}")
        client_dataloader = torch.utils.data.DataLoader(
            client_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        clientwise_dataloaders[client_id] = client_dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)

    # train the model
    global_model = None
    retrained_global_model = None
    if args.model == 'allcnn': 
        if args.dataset == 'mnist':
            global_model = AllCNN(num_classes=num_classes, num_channels=1)
        else:
            global_model = AllCNN(num_classes=num_classes)

    elif args.model == 'resnet18':
        if args.dataset == 'mnist':
            global_model = ResNet18(num_classes=num_classes,
                                    pretrained=args.pretrained, num_channels=1)
        else:
            global_model = ResNet18(num_classes=num_classes,
                                    pretrained=args.pretrained)

    elif args.model == 'smallcnn':
        in_ch = 1 if args.dataset == 'mnist' else 3
        global_model = SmallCNN(num_channels=in_ch, num_classes=num_classes)

    else:
        raise ValueError("Invalid model name")

    retrained_global_model = deepcopy(global_model)
    print(f"Model: {global_model}")
    # 原来：
    # train_path = os.path.abspath(os.path.join(weights_path, "full_training"))
    # global_model = fed_train(...)

    # 改为：
    if args.unlearn_only:
        # 复用已有训练产物
        train_path = os.path.abspath(
            args.full_training_dir if args.full_training_dir
            else os.path.join(weights_path, "full_training")
        )
        if not os.path.isdir(train_path):
            raise RuntimeError(f"[Unlearn-Only] 找不到 full_training 目录：{train_path}")

        # 选择要加载的全局模型权重
        import os, torch
        candidates = [
            os.path.join(train_path, "final_model.pth"),
            os.path.join(train_path, f"iteration_{max([int(d.split('_')[-1]) for d in os.listdir(train_path) if d.startswith('iteration_')], default=-1)}", "global_model.pth")
        ]
        ckpt_path = args.global_ckpt if args.global_ckpt else next((p for p in candidates if os.path.isfile(p)), None)
        if not ckpt_path:
            raise RuntimeError("[Unlearn-Only] 未找到可用的全局模型权重（final_model.pth 或最后一轮的 global_model.pth）")

        global_model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        print(f"[Unlearn-Only] 复用训练结果：{ckpt_path}")
    else:
        # 照常训练（注意 fed_train 会清空 weights_path）
        train_path = os.path.abspath(os.path.join(weights_path, "full_training"))
        global_model = fed_train(num_training_iterations=args.num_training_iterations,
                                test_dataloader=test_dataloader,
                                clientwise_dataloaders=clientwise_dataloaders,
                                weights_path=train_path,
                                global_model=global_model,
                                num_local_epochs=args.num_local_epochs,
                                lr=args.lr,
                                optimizer_name=args.optimizer,
                                device=args.device)


    perf = get_performance(model=global_model, test_dataloader=test_dataloader, num_classes=num_classes,
                           clientwise_dataloader=clientwise_dataloaders, device=args.device)
    summary['performance'] = {}
    summary['performance']['after_training'] = perf
    if args.verbose:
        print(f"Performance after training : {perf}")

        forget_loader = clientwise_dataloaders[forget_client]
        acc = get_accuracy_only(global_model, forget_loader, args.device)
        print(f"[Training模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")
    # -------------------------------------------------------
    # train mia attack model
    if args.apply_membership_inference:
        shadow_model = deepcopy(global_model)
        # attack_model = XGBClassifier()
        attack_model = train_attack_model(shadow_global_model=shadow_model,
                                          shadow_client_loaders=clientwise_dataloaders,
                                          shadow_test_loader=test_dataloader,
                                          dataset=args.dataset,
                                          device=args.device)
    # ---------------------------------------------------------
    # evaluate attack accuracy
    if args.apply_backdoor:

        # ------------------------------------------
        # TO-DO: Implement data poisoning and eval from https://arxiv.org/abs/2402.14015 (give credit in code!)

        ''' 
        -- Create data loaders with poison/clean data --
        To be implemented in here (till ca. line 44): https://github.com/drimpossible/corrective-unlearning-bench/blob/main/src/main.py
        FYI kep corrupt size at 3 for the pixel attack patch, manip_set_size=opt.forget_set_size determines how many poisoned samples
        From line 80 onwards keep opt.deletion_size == opt.forget_set_size to have all poison samples known. Unlearning unkown samples is
        a different hard problem beyond the scope of this FL-UL paper
        Helper functions in here: https://github.com/drimpossible/corrective-unlearning-bench/blob/main/src/datasets.py

        -- Eval --
        For evaluation, report the accuracies on the poisoned data with the clean labels 
        (i.e., what they should be, not what the manipulated poisoned sample says it is) and
        the accuracy on the remaining clean data. See figures 2 & 3
        '''
        # ------------------------------------------

        backdoor_results = evaluate_backdoor_attack(model=global_model, backdoor_context=backdoor_context,
                                                    device=args.device)
        summary['backdoor_results'] = {}
        summary['backdoor_results']['global_model_after_training'] = backdoor_results

        backdoor_client = deepcopy(global_model)
        backdoor_client_path = os.path.abspath(os.path.join(
            train_path, f"iteration_{args.num_training_iterations - 1}", f"client_{args.forget_clients[0]}.pth"))
        backdoor_client.load_state_dict(torch.load(backdoor_client_path))

        backdoor_results_client = evaluate_backdoor_attack(model=backdoor_client, backdoor_context=backdoor_context,
                                                           device=args.device)
        summary['backdoor_results']['backdoor_client_after_training'] = backdoor_results_client

        if args.verbose:
            print(
                f"Backdoor results after training : {summary['backdoor_results']}")

    # evaluate poisoning accuracy
    if args.apply_label_poisoning:
        poisoning_results = evaluate_poisoning_attack(model=global_model,
                                                      poisoning_context=poisoning_context,
                                                      device=args.device)
        summary['poisoning_results'] = {}
        summary['poisoning_results']['global_model_after_training'] = poisoning_results

        poisoning_client = deepcopy(global_model)
        poisoning_client_path = os.path.abspath(os.path.join(
            train_path, f"iteration_{args.num_training_iterations - 1}", f"client_{args.forget_clients[0]}.pth"))
        poisoning_client.load_state_dict(torch.load(poisoning_client_path))

        poisoning_results_client = evaluate_poisoning_attack(model=poisoning_client,
                                                             poisoning_context=poisoning_context,
                                                             device=args.device)
        summary['poisoning_results']['poisoning_client_after_training'] = poisoning_results_client

        if args.verbose:
            print(
                f"Poisoning results after training : {summary['poisoning_results']}")

    retrain_path = os.path.join(weights_path, "retraining")
    # train the model on retain data
    retain_clientwise_dataloaders = {key: value for key, value in clientwise_dataloaders.items()
                                     if key not in args.forget_clients}
    print(f"Retain Client wise Loaders: {retain_clientwise_dataloaders}")

    if not args.skip_retraining:

        retrained_global_model = fed_train(num_training_iterations=args.num_training_iterations, test_dataloader=test_dataloader,
                                        clientwise_dataloaders=retain_clientwise_dataloaders,
                                        global_model=retrained_global_model, num_local_epochs=args.num_local_epochs,
                                        device=args.device, weights_path=retrain_path, lr=args.lr, optimizer_name=args.optimizer)

        perf = get_performance(model=retrained_global_model, test_dataloader=test_dataloader,
                            clientwise_dataloader=clientwise_dataloaders,
                            num_classes=num_classes, device=args.device)
        summary['performance']['after_retraining'] = perf
        if args.verbose:
            print(f"Performance after retraining : {perf}")
        # evaluate attack accuracy on retrained model

        # ---- 专门测忘却客户端的精度 ----
            forget_loader = clientwise_dataloaders[forget_client]
            acc = get_accuracy_only(retrained_global_model, forget_loader, args.device)
            print(f"[Retrain模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")

    else:
        if args.verbose:
            print("[Skip] 跳过重训练基线（--skip_retraining）")        




    
    if args.apply_backdoor and not args.skip_retraining:
        retrained_backdoor_results = evaluate_backdoor_attack(model=retrained_global_model,
                                                              backdoor_context=backdoor_context, device=args.device)
        summary['backdoor_results']['after_retraining'] = retrained_backdoor_results
        if args.verbose:
            print(
                f"Backdoor results after retraining : {retrained_backdoor_results}")

    if args.apply_label_poisoning and not args.skip_retraining:
        retrained_poisoning_results = evaluate_poisoning_attack(model=retrained_global_model,
                                                                poisoning_context=poisoning_context,
                                                                device=args.device)
        summary['poisoning_results']['after_retraining'] = retrained_poisoning_results

        if args.verbose:
            print(
                f"Poisoning results after retraining : {retrained_poisoning_results}")

    # Run Baseline methods and check the performance on them
    baselines_methods = args.baselines
    for baseline in baselines_methods:
        if baseline == 'pga':
            global_model_pga = deepcopy(global_model)
            unlearned_pga_model = run_pga(global_model=global_model_pga,
                                          weights_path=train_path,
                                          clientwise_dataloaders=clientwise_dataloaders,
                                          forget_client=args.forget_clients,
                                          model=args.model,
                                          dataset=args.dataset,
                                          num_clients=args.total_num_clients,
                                          num_classes=num_classes,
                                          pretrained=args.pretrained,
                                          num_training_iterations=args.num_training_iterations,
                                          device=args.device,
                                          lr=args.lr,
                                          optimizer_name=args.optimizer,
                                          num_local_epochs=args.num_local_epochs,
                                          num_unlearn_rounds=1,
                                          num_post_training_rounds=1)
            perf = get_performance(model=unlearned_pga_model, test_dataloader=test_dataloader,
                                   clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                   device=args.device)
            summary['performance']['after_pga'] = perf
            if args.verbose:
                print(f"Performance after pga : {perf}")
            # check backdoor on PGA model
            if args.apply_backdoor:
                forget_backdoor_pga = evaluate_backdoor_attack(model=unlearned_pga_model, backdoor_context=backdoor_context,
                                                               device=args.device)
                summary['backdoor_results']['after_pga'] = forget_backdoor_pga

                if args.verbose:
                    print(
                        f"Backdoor results after pga : {forget_backdoor_pga}")
            if args.apply_label_poisoning:
                forget_poisoning_pga = evaluate_poisoning_attack(model=unlearned_pga_model,
                                                                 poisoning_context=poisoning_context,
                                                                 device=args.device)
                summary['poisoning_results']['after_pga'] = forget_poisoning_pga

                if args.verbose:
                    print(
                        f"Poisoning results after pga : {forget_poisoning_pga}")
        elif baseline == 'fed_eraser':
            global_model_federaser = deepcopy(global_model)
            unlearned_federaser_model = run_fed_eraser(global_model=global_model_federaser,
                                                       weights_path=train_path,
                                                       clientwise_dataloaders=clientwise_dataloaders,
                                                       forget_clients=args.forget_clients,
                                                       num_clients=args.total_num_clients,
                                                       num_rounds=args.num_training_iterations,
                                                       device=args.device,
                                                       lr=args.lr,
                                                       optimizer_name=args.optimizer,
                                                       local_cali_round=1,
                                                       num_unlearn_rounds=1,
                                                       num_post_training_rounds=1)
            perf = get_performance(model=unlearned_federaser_model, test_dataloader=test_dataloader,
                                   clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                   device=args.device)
            summary['performance']['after_federaser'] = perf
            if args.verbose:
                print(f"Performance after federaser : {perf}")
            # check backdoor on Federaser model
            if args.apply_backdoor:
                forget_backdoor_federaser = evaluate_backdoor_attack(model=unlearned_federaser_model, backdoor_context=backdoor_context,
                                                                     device=args.device)
                summary['backdoor_results']['after_federaser'] = forget_backdoor_federaser

                if args.verbose:
                    print(
                        f"Backdoor results after federaser : {forget_backdoor_federaser}")
            if args.apply_label_poisoning:
                forget_poisoning_federaser = evaluate_poisoning_attack(model=unlearned_federaser_model,
                                                                       poisoning_context=poisoning_context,
                                                                       device=args.device)
                summary['poisoning_results']['after_federaser'] = forget_poisoning_federaser

                if args.verbose:
                    print(
                        f"Poisoning results after federaser : {forget_poisoning_federaser}")
        elif baseline == 'fair_vue':
            
            # ---- FAIR-VUE（按轮）----
            print(">>> Running FAIR-VUE (round-wise)...")
            fair_model = deepcopy(global_model).to(args.device)
            fair_model.eval()
            param_keys = [name for name, p in fair_model.named_parameters() if p.requires_grad]
            from FedUnlearner.baselines.fair_vue.subspace import (
                weighted_matrix_from_deltas_keys, rho_values_keys,
                flatten_by_keys, state_dict_like_by_keys
            )

            # 1) 按轮读取所有客户端逐轮增量 Δ_{cid}^{(r)}
            train_path = os.path.abspath(os.path.join(weights_path, "full_training"))
            round_client_deltas = _build_round_deltas(train_path, args.total_num_clients)

            # === 插入点 A：刚刚构造完 round_client_deltas 之后 ===
            if args.fair_vue_debug:
                rounds = sorted(list(round_client_deltas.keys()))
                print(f"[FV-DBG] rounds_with_deltas={len(rounds)} -> {rounds[:8]}{'...' if len(rounds)>8 else ''}")
                total_pairs = sum(len(v) for v in round_client_deltas.values())
                print(f"[FV-DBG] total (round,client) delta pairs={total_pairs}")
                # 打印前3个轮的 Δ 范数均值
                from statistics import mean
                def _flat_norm(d): 
                    import torch
                    from FedUnlearner.baselines.fair_vue.subspace import flatten_state_dict
                    return float(torch.norm(flatten_state_dict(d)).item())
                for r in rounds[:3]:
                    ns = [ _flat_norm(d) for d in round_client_deltas[r].values() ]
                    print(f"[FV-DBG] round={r} mean||Δ||={mean(ns):.3e} (n={len(ns)})")

            if len(round_client_deltas) == 0:
                raise RuntimeError(f"[FAIR-VUE] 没有从 {train_path} 解析到逐轮的 client 权重，无法构造增量。")

            # 2) 目标客户端历史：拼成列表用于SVD；其它客户端的增量合起来用于ρ
            target_id = args.forget_clients[0]
            target_deltas_list = []
            other_deltas_list  = []

            for r, deltas in round_client_deltas.items():
                if target_id in deltas:
                    target_deltas_list.append(deltas[target_id])
                # 其它客户端
                for cid, d in deltas.items():
                    if cid != target_id:
                        other_deltas_list.append(d)

            if len(target_deltas_list) == 0:
                raise RuntimeError("[FAIR-VUE] 没有找到目标客户端的逐轮增量，无法进行SVD。")
            # === 原有：target/others 划分完成后 ===
            if args.fair_vue_debug:
                print(f"[FV-DBG] target_id={target_id}, T=len(target_deltas_list)={len(target_deltas_list)}, "
                    f"M=len(other_deltas_list)={len(other_deltas_list)}")

            # 3) Fisher（在目标客户端数据上估计对角Fisher）
            if target_id in clientwise_dataloaders:
                # 确保模型参数可求导
                for p in fair_model.parameters():
                    p.requires_grad_(True)
                fisher = empirical_fisher_diagonal(fair_model, clientwise_dataloaders[target_id],
                                                device=args.device, max_batches=args.fair_fisher_batches)
            else:
                # fallback：如果没有目标客户端loader，就用全1权重
                fisher = {k: torch.ones_like(v) for k, v in fair_model.state_dict().items()}

            # === Fisher 计算完毕后，插入点 B ===
            if args.fair_vue_debug:
                import torch
                from FedUnlearner.baselines.fair_vue.subspace import flatten_state_dict
                fvec = flatten_state_dict(fisher)
                print(f"[FV-DBG] fisher: device={fvec.device}, D={fvec.numel()}, "
                    f"min={float(torch.min(fvec)): .3e}, max={float(torch.max(fvec)): .3e}, "
                    f"mean={float(torch.mean(fvec)): .3e}")

            # 4) Fisher加权矩阵 & 低秩SVD拿到主方向V
            Xw = weighted_matrix_from_deltas_keys(target_deltas_list, fisher, param_keys, device=args.device)  # 已在GPU
            # === Xw / SVD 完成后，插入点 C ===
            if args.fair_vue_debug:
                # Xw: T x D; V: D x k; SVD 的奇异值能反映主方向能量
                print(f"[FV-DBG] Xw.shape={tuple(Xw.shape)}, device={Xw.device}")
            V  = topk_right_singular_vectors(Xw, k=args.fair_rank_k)
            if args.fair_vue_debug:
                # 重新SVD一次拿 S，不影响 V（只做诊断）
                U_, S_, Vh_ = torch.linalg.svd(Xw - Xw.mean(dim=0, keepdim=True), full_matrices=False)
                top_s = [float(s) for s in S_[:min(8, S_.numel())]]
                print(f"[FV-DBG] top singular values={top_s}, k={V.shape[1]}, V.device={V.device}")                 # 跟随Xw的device
            dev = V.device  # ★ 统一使用这个设备（通常是cuda:0）



            dev = V.device

            # 5) 计算ρ并按阈值切分为 V_spec / V_comm
            rhos = rho_values_keys(V, other_deltas_list, param_keys)  # 内部已用 V.device 对齐
            if len(rhos) == 0:
                tau = float('inf')
            else:
                tau = (sorted(rhos)[len(rhos)//2] if args.fair_tau_mode == 'median'
                    else sum(rhos)/len(rhos))
            # ★ 这里要接住索引（别用下划线丢掉）
            V_spec, V_comm, spec_idx, comm_idx = split_subspaces(V, rhos, tau)

            # ★ 兜底：如果 V_spec 还是空，强制取 rho 最小的 1~2 个方向，放在 debug 外侧更稳
            if V_spec is None or V_spec.numel() == 0:
                idx_sorted = sorted(range(len(rhos)), key=lambda i: rhos[i])
                take = min(2, len(idx_sorted))
                if take > 0:
                    V_spec = V[:, idx_sorted[:take]]

            if args.fair_vue_debug:
                import numpy as np
                r = np.array(rhos) if len(rhos)>0 else np.array([0.0])
                def q(a,p): 
                    import numpy as np
                    return float(np.quantile(a,p))
                print(f"[FV-DBG] rho stats: n={len(rhos)}, min={r.min():.3e}, q25={q(r,0.25):.3e}, "
                    f"median={q(r,0.5):.3e}, q75={q(r,0.75):.3e}, max={r.max():.3e}, tau={tau:.3e}")
                print(f"[FV-DBG] |V_spec|={V_spec.shape[1]}, idx={spec_idx[:10]}{'...' if len(spec_idx)>10 else ''}")
                print(f"[FV-DBG] |V_comm|={V_comm.shape[1]}")

                if V_spec.numel() == 0:
                    print("[FV-DBG][WARN] V_spec is EMPTY -> 将强制选择 rho 最小的1~2个方向作为特异方向（兜底）")
                    idx_sorted = sorted(range(len(rhos)), key=lambda i: rhos[i])
                    take = min(2, len(idx_sorted))
                    V_spec = V[:, idx_sorted[:take]]

            # 6) 基于 V_spec 构造投影矩阵（会跟随 V_spec.device）
            # 低秩：只构造 Q，不构造 P

            Q = None
            if V_spec is not None and V_spec.numel() > 0:
                Q, _ = torch.linalg.qr(V_spec, mode='reduced')  # D_param x r
            if args.fair_vue_debug:
                print(f"[FV-DBG] Q is None? {Q is None}, "
                    f"Q.shape={(None if Q is None else tuple(Q.shape))}, device={(None if Q is None else Q.device)}")



            # 7) 逐轮累计“特异分量”
            # 只在 parameters 空间累积“目标客户端”的特异分量
            start_sd = {k: v.detach().clone().cpu() for k, v in fair_model.state_dict().items()}
            Dparam   = flatten_by_keys(start_sd, param_keys).numel()
            spec_total = torch.zeros(Dparam, device=dev)

            used_rounds = 0
            for r, deltas in round_client_deltas.items():
                if target_id not in deltas:
                    continue
                # 仅取目标客户端
                d_tar = flatten_by_keys(deltas[target_id], param_keys, device=dev)  # Δ_target^t
                if Q is not None:
                    spec = Q @ (Q.T @ d_tar)  # 只拿“目标”的特异分量
                else:
                    spec = torch.zeros_like(d_tar, device=dev)
                spec_total += spec
                used_rounds += 1

            if args.fair_vue_debug:
                print(f"[FV-DBG] used_rounds_for_target={used_rounds}, ||spec_total||_2={float(torch.norm(spec_total)):.3e}")

            # 应用擦除（仅 parameters）
            erase_scale = getattr(args, "fair_erase_scale", 0.25)
            param_now   = flatten_by_keys(start_sd, param_keys, device=dev)
            param_new   = param_now - erase_scale * spec_total
            new_params  = state_dict_like_by_keys(param_new.to('cpu'), start_sd, param_keys)
            new_sd = dict(start_sd)
            for k in param_keys:
                new_sd[k] = new_params[k]
            fair_model.load_state_dict(new_sd)


           

            # 9) 评测
            perf = get_performance(model=fair_model, test_dataloader=test_dataloader,
                                clientwise_dataloader=clientwise_dataloaders,
                                num_classes=num_classes, device=args.device)
            summary.setdefault('performance', {})
            summary['performance']['after_fair_vue'] = perf
            if args.verbose:
                print(f"Performance after FAIR-VUE : {perf}")

            # ---- 专门测忘却客户端的精度 ----
            forget_loader = clientwise_dataloaders[target_id]
            acc = get_accuracy_only(fair_model, forget_loader, args.device)
            print(f"[FAIR-VUE模型] 忘却客户端{target_id}自有数据精度: {acc*100:.2f}%")



            # ===== HEALING: 让遗忘后的模型在保留客户端上“治疗/恢复” =====
            if args.heal:
                # 1) teacher 选择：post=用擦除后的模型；pre=用遗忘前的全局模型
                if args.heal_teacher == 'post':
                    teacher = deepcopy(fair_model).to(args.device).eval()
                else:
                    teacher = deepcopy(global_model).to(args.device).eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)

                # 2) 组治疗数据：拼 retain 客户端
                from torch.utils.data import ConcatDataset, DataLoader
                retain_datasets = [clientwise_dataset[cid] for cid in clientwise_dataset
                                if cid not in args.forget_clients]
                heal_dataset = ConcatDataset(retain_datasets)
                hb = args.heal_batch_size if args.heal_batch_size else args.batch_size
                
                from collections import Counter
                from torch.utils.data import WeightedRandomSampler

                targets = []
                for ds in retain_datasets:
                    # 假设 ds[i] -> (x, y)，y 是 int
                    targets.extend([ds[i][1] for i in range(len(ds))])

                cls_count = Counter(targets)
                total = sum(cls_count.values())
                # 每类的采样权重 = 1/频次
                weights = [1.0 / cls_count[y] for y in targets]
                sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)

                heal_loader = DataLoader(heal_dataset, batch_size=hb,
                                        sampler=sampler,  # 用 sampler 而不是 shuffle
                                        num_workers=args.num_workers)

                # 3) 触发治疗（参数来自命令行）
                selective_kd_heal(
                    student=fair_model,
                    teacher=teacher,
                    dataloader=heal_loader,
                    v_spec=V_spec,
                    num_steps=args.heal_steps,
                    lr=args.heal_lr,
                    T=args.heal_T,
                    lambda_kd=args.heal_lambda_kd,
                    lambda_ortho=args.heal_lambda_ortho,
                    device=args.device,
                    keys=param_keys,               # 与 V_spec 的参数展开顺序对齐
                    grad_project=not args.no_heal_grad_proj
                )

                # 4) 治疗后评测
                perf_heal = get_performance(model=fair_model, test_dataloader=test_dataloader,
                                            clientwise_dataloader=clientwise_dataloaders,
                                            num_classes=num_classes, device=args.device)
                print(f"[HEAL] Performance after healing : {perf_heal}")
                forget_loader = clientwise_dataloaders[target_id]
                acc_forget_heal = get_accuracy_only(fair_model, forget_loader, args.device)
                print(f"[HEAL] 忘却客户端{target_id}自有数据精度(治疗后): {acc_forget_heal*100:.2f}%")

        # ----------------- FedFIM -----------------
    if "fedfim" in args.baselines:
        from FedUnlearner.baselines import run_fedfIM
        fedfim_model = deepcopy(global_model)

        _result = run_fedfIM(
            model=fedfim_model,
            client_loaders=clientwise_dataloaders,
            forget_clients=args.forget_clients,
            device=args.device,
            num_classes=num_classes,
            fim_max_passes=args.fim_max_passes,
            fim_max_batches=args.fim_max_batches,
            fim_mode=args.fim_mode,
            fim_topk=args.fim_topk,
            dampening_constant=args.fim_gamma,
            ratio_cutoff=args.fim_ratio_cutoff,
            upper_bound=args.fim_upper_bound,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            finetune_weight_decay=args.finetune_wd,
        )
# 兼容：run_fedfIM 可能返回 model 或 (model, F_r, F_f)
        if isinstance(_result, tuple):
            fedfim_model = _result[0]
        else:
            fedfim_model = _result

        # 保底断言，防止类型再出错
        import torch as _torch
        assert isinstance(fedfim_model, _torch.nn.Module), \
            f"run_fedfIM must return an nn.Module as first item, got {type(fedfim_model)}"

        perf = get_performance(model=fedfim_model, test_dataloader=test_dataloader,
                            clientwise_dataloader=clientwise_dataloaders,
                            num_classes=num_classes, device=args.device)
        summary['performance']['after_fedfim'] = perf
        if args.verbose:
            print(f"Performance after FedFIM : {perf}")

        forget_loader = clientwise_dataloaders[forget_client]
        acc = get_accuracy_only(fedfim_model, forget_loader, args.device)
        print(f"[FedFIM模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")


    # ============ Legacy Unlearn Baseline（默认不执行） ============
    if RUN_LEGACY_UNLEARN:
        unlearned_global_weights = unlearn_ours(global_model=global_model.cpu().state_dict(), forget_clients=args.forget_clients,
                                                total_num_clients=args.total_num_clients, weights_path=weights_path,
                                                dampening_constant=args.dampening_constant, dampening_upper_bound=args.dampening_upper_bound,
                                                ratio_cutoff=args.ratio_cutoff)
        unlearned_global_model = deepcopy(global_model)
        unlearned_global_model.load_state_dict(unlearned_global_weights)
        # check classwise accuracy
        if RUN_LEGACY_UNLEARN:
            perf = get_performance(model=unlearned_global_model, test_dataloader=test_dataloader,
                                clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                device=args.device)
            summary['performance']['after_unlearning'] = perf
        if args.verbose:
            print(f"Performance after unlearning : {perf}")

            # ---- 专门测忘却客户端的精度 ----
            forget_loader = clientwise_dataloaders[forget_client]
            acc = get_accuracy_only(unlearned_global_model, forget_loader, args.device)
            print(f"[Unlearn模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")


        # check backdoor on unlearned model
        if args.apply_backdoor:
            if RUN_LEGACY_UNLEARN:
                forget_backdoor_attacks = evaluate_backdoor_attack(model=unlearned_global_model, backdoor_context=backdoor_context,
                                                                device=args.device)
                summary['backdoor_results']['after_unlearning'] = forget_backdoor_attacks

            if args.verbose:
                print(
                    f"Backdoor results after unlearning : {forget_backdoor_attacks}")

        # check poisoning on unlearned model
        if args.apply_label_poisoning:
            if RUN_LEGACY_UNLEARN:
                forget_poisoning_attacks = evaluate_poisoning_attack(model=unlearned_global_model,
                                                                    poisoning_context=poisoning_context,
                                                                    device=args.device)
                summary['poisoning_results']['after_unlearning'] = forget_poisoning_attacks

            if args.verbose:
                print(
                    f"Poisoning results after unlearning : {forget_poisoning_attacks}")
    # check mia precision and recall on all model
    summary['mia_attack'] = {}
    if args.apply_membership_inference:
        if RUN_LEGACY_UNLEARN:
            unlearning_mia_result = evaluate_mia_attack(target_model=deepcopy(unlearned_global_model),
                                                        attack_model=attack_model,
                                                        client_loaders=clientwise_dataloaders,
                                                        test_loader=test_dataloader,
                                                        dataset=args.dataset,
                                                        forget_client_idx=args.forget_clients[0],
                                                        device=args.device)
            summary['mia_attack']['after_unlearning'] = unlearning_mia_result
            if args.verbose:
                print(
                    f"MIA results after unlearning : {unlearning_mia_result}")
        if not args.skip_retraining:
            retrained_mia_result = evaluate_mia_attack(target_model=deepcopy(retrained_global_model),
                                                    attack_model=attack_model,
                                                    client_loaders=clientwise_dataloaders,
                                                    test_loader=test_dataloader,
                                                    dataset=args.dataset,
                                                    forget_client_idx=args.forget_clients[0],
                                                    device=args.device)
            summary['mia_attack']['after_retraining'] = retrained_mia_result
            if args.verbose:
                print(
                    f"MIA results after retraining : {retrained_mia_result}")

        for baseline in baselines_methods:
            if baseline == 'pga':

                pga_mia_result = evaluate_mia_attack(target_model=deepcopy(unlearned_pga_model),
                                                     attack_model=attack_model,
                                                     client_loaders=clientwise_dataloaders,
                                                     test_loader=test_dataloader,
                                                     dataset=args.dataset,
                                                     forget_client_idx=args.forget_clients[0],
                                                     device=args.device)
                summary['mia_attack']['after_pga'] = pga_mia_result
                if args.verbose:
                    print(
                        f"MIA results after pga : {pga_mia_result}")
            elif baseline == 'fed_eraser':
                federaser_mia_result = evaluate_mia_attack(target_model=deepcopy(unlearned_federaser_model),
                                                           attack_model=attack_model,
                                                           client_loaders=clientwise_dataloaders,
                                                           test_loader=test_dataloader,
                                                           dataset=args.dataset,
                                                           forget_client_idx=args.forget_clients[0],
                                                           device=args.device)
                summary['mia_attack']['after_federaser'] = federaser_mia_result
                if args.verbose:
                    print(
                        f"MIA results after federaser : {federaser_mia_result}")

    # Add configurations to the summary
    summary['config'] = vars(args)

    # Create a timestamp for the summary file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Dump the summary into a file with the summary-timestamp name
    with open(os.path.join(weights_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
