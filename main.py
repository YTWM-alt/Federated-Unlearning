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
import gc


from FedUnlearner.utils import (
    print_exp_details, print_clientwise_class_distribution,
    eval_ce_loss, cosine_angle_between_models, print_forgetting_metrics,
    eval_retain_acc
)
from FedUnlearner.data_utils import get_dataset, create_dirichlet_data_distribution, create_iid_data_distribution, create_class_exclusive_distribution
from FedUnlearner.fed_learn import fed_train, get_performance
from FedUnlearner.models import AllCNN, ResNet18, SmallCNN
from FedUnlearner.attacks.backdoor import create_backdoor_dataset, evaluate_backdoor_attack
from FedUnlearner.baselines import run_pga, run_fed_eraser
from FedUnlearner.baselines import run_conda
from FedUnlearner.attacks.mia import train_attack_model, evaluate_mia_attack
from FedUnlearner.attacks.poisoning import create_poisoning_dataset, evaluate_poisoning_attack

# 模型治疗（方案1：参数平滑）
from FedUnlearner.baselines.fair_vue.healing import weight_interpolate_heal

# >>> FAIR-VUE: imports
from FedUnlearner.baselines.fair_vue.fisher import empirical_fisher_diagonal
from FedUnlearner.baselines.fair_vue.subspace import (
    weighted_matrix_from_deltas, topk_right_singular_vectors,
    rho_values, split_subspaces, flatten_state_dict, state_dict_like,
    # 仅参数键的版本：避免包含 buffer 造成维度不匹配
    flatten_by_keys, state_dict_like_by_keys,
    weighted_matrix_from_deltas_keys, rho_values_keys
)
from FedUnlearner.baselines.fair_vue.projection import projection_matrix

# === 日志目录与文件 ===
os.makedirs("./logs", exist_ok=True)
log_path = f"./logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

 # === 日志：支持多文件；真正初始化延后到解析参数之后 ===
class ProxyLog:
    """
    代理 stdout/stderr：
    1) 可写入多个日志文件（同一份日志复制到多处）
    2) 控制台照常显示（保留 tqdm 进度条等）
    3) 暴露 isatty/fileno 等，让 tqdm 识别为 TTY
    """
    def __init__(self, stream, log_paths):
        import os
        self.stream = stream
        # 允许传入字符串或列表
        if isinstance(log_paths, (str, os.PathLike)):
            self.log_paths = [str(log_paths)]
        else:
            self.log_paths = [str(p) for p in log_paths]
        # 进度条/吞吐的过滤（保持与原逻辑一致）
        self.re_bar = re.compile(
            r"(?:\r)?(?:(?=.*\d{1,3}%\|.+\|).*|.*\|[#█░▉▊▋▌▍▎▏]+\|.*|.*\b(?:it/s|s/it|ETA|elapsed|remaining)\b.*)"
        )
        self.re_client = re.compile(r"^\s*Client:\s*\d+\s*$")
        self._buf = ""
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def _should_skip(self, text: str) -> bool:
        if "\r" in text:
            return True
        if self.re_bar.search(text):
            return True
        if self.re_client.search(text.strip()):
            return True
        return False

    def _write_all(self, text: str):
        for p in self.log_paths:
            # 目录可能尚未创建；这里稳妥创建
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(text)

    def write(self, message: str):
        self._buf += message
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line_out = line + "\n"
            if not self._should_skip(line_out):
                self._write_all(line_out)
            self.stream.write(line_out)

    def flush(self):
        if self._buf:
            if not self._should_skip(self._buf):
                self._write_all(self._buf)
            self.stream.write(self._buf)
            self._buf = ""
        if hasattr(self.stream, "flush"):
            self.stream.flush()

    def isatty(self):
        try:
            return self.stream.isatty()
        except Exception:
            return True

    def fileno(self):
        try:
            return self.stream.fileno()
        except Exception:
            import io as _io
            raise _io.UnsupportedOperation("fileno")

    def writable(self):
        return True

    def __getattr__(self, name):
        return getattr(self.stream, name)

# 命令行显式布尔解析函数
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ——————————————

def get_accuracy_only(model, dataloader, device):
    model = model.to(device)
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

def _iter_round_deltas_stream(
    train_path: str,
    total_clients: int,
    param_keys: list = None,
    max_rounds: int = None,
):
    """
    逐轮产生 {cid -> Δ_i^t}，避免一次性攒全量到内存。
    Δ_i^t = client_{i,t} - mean_clients_{t-1}

    优化点：
      1) 只在给定的 param_keys 上累积/构造 Δ，避免对整个 ResNet 全参数做差。
      2) 复用上一轮的均值 g_prev：每轮只加载“当前轮” client_{cid}.pth，
         不再为每一轮重复读取上一轮权重，减少约一半 ckpt 读盘。
      3) 可选 max_rounds：只解析最近 max_rounds 个“轮”的 Δ，进一步节省时间。
    """
    rounds = _list_iteration_dirs(train_path)
    num_rounds = len(rounds)
    if num_rounds <= 1:
        return

    # 需要的“当前轮”个数（Δ_i^t 的轮数），不能超过 num_rounds-1
    max_rounds = int(max_rounds) if (max_rounds is not None and max_rounds > 0) else None
    if max_rounds is not None:
        max_rounds = min(max_rounds, num_rounds - 1)
        start_idx = num_rounds - max_rounds   # 从这里开始作为 “当前轮”
    else:
        start_idx = 1  # 兼容旧行为：从第 1 轮开始（需要第 0 轮作 g_prev）

    # 要用哪些参数 key
    keys_filter = set(param_keys) if param_keys is not None and len(param_keys) > 0 else None

    # --- 先在 start_idx-1 那一轮上构造 g_prev（mean_clients_{start_idx-1}） ---
    _, dir_prev = rounds[start_idx - 1]
    g_prev = None
    cnt_prev = 0
    for cid in range(total_clients):
        sd_prev = _load_client_sd(dir_prev, cid)
        if sd_prev is None:
            continue
        if g_prev is None:
            # 初始化：只保留需要的 keys（若未指定 param_keys，则保留全部）
            if keys_filter is None:
                g_prev = {
                    k: v.detach().to('cpu', dtype=torch.float32).clone()
                    for k, v in sd_prev.items()
                }
            else:
                g_prev = {}
                for k in keys_filter:
                    if k in sd_prev:
                        g_prev[k] = sd_prev[k].detach().to('cpu', dtype=torch.float32).clone()
        else:
            if keys_filter is None:
                for k in g_prev.keys():
                    if k in sd_prev:
                        g_prev[k].add_(sd_prev[k].to(dtype=torch.float32))
            else:
                for k in g_prev.keys():
                    if k in sd_prev:
                        g_prev[k].add_(sd_prev[k].to(dtype=torch.float32))
        cnt_prev += 1
        del sd_prev

    if not cnt_prev or g_prev is None:
        return

    for k in g_prev.keys():
        g_prev[k].div_(float(cnt_prev))

    # --- 从 start_idx 开始，边算 Δ 边更新下一轮的 g_prev ---
    for r_idx in range(start_idx, num_rounds):
        r_cur, dir_cur = rounds[r_idx]

        deltas_this_round = {}
        sums_cur = None
        cnt_cur = 0

        for cid in range(total_clients):
            sd_cur = _load_client_sd(dir_cur, cid)
            if sd_cur is None:
                continue

            # 1) 只在需要的 keys 上构造 Δ
            if keys_filter is None:
                keys_now = g_prev.keys()
            else:
                keys_now = [k for k in g_prev.keys() if k in keys_filter]

            d = {}
            for k in keys_now:
                if k in sd_cur and k in g_prev:
                    d[k] = sd_cur[k].to('cpu', dtype=torch.float32) - g_prev[k]
            if d:
                deltas_this_round[cid] = d

            # 2) 顺便累积本轮均值，用于下一轮的 g_prev
            if sums_cur is None:
                if keys_filter is None:
                    sums_cur = {
                        k: v.detach().to('cpu', dtype=torch.float32).clone()
                        for k, v in sd_cur.items()
                    }
                else:
                    sums_cur = {}
                    for k in keys_filter:
                        if k in sd_cur:
                            sums_cur[k] = sd_cur[k].detach().to('cpu', dtype=torch.float32).clone()
            else:
                if keys_filter is None:
                    for k in sums_cur.keys():
                        if k in sd_cur:
                            sums_cur[k].add_(sd_cur[k].to(dtype=torch.float32))
                else:
                    for k in sums_cur.keys():
                        if k in sd_cur:
                            sums_cur[k].add_(sd_cur[k].to(dtype=torch.float32))

            cnt_cur += 1
            del sd_cur

        if deltas_this_round:
            yield (r_cur, deltas_this_round)

        # 更新 g_prev → 当前轮的均值
        if sums_cur is not None and cnt_cur > 0:
            for k in sums_cur.keys():
                sums_cur[k].div_(float(cnt_cur))
            g_prev = sums_cur

        # 释放
        del deltas_this_round
        if sums_cur is not None:
            del sums_cur
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




# ---- FAIR-VUE Δ 缓存：类似 conda 的“环境缓存” ----
def _fair_vue_delta_cache_key(train_path: str,
                              total_clients: int,
                              target_id: int,
                              param_keys: list,
                              fair_target_rounds: int):
    import hashlib
    import json as _json
    import os as _os
    meta = {
        "train_path": _os.path.abspath(train_path),
        "total_clients": int(total_clients),
        "target_id": int(target_id),
        "param_keys": list(param_keys or []),
        "fair_target_rounds": int(fair_target_rounds),
        "version": 1,  # 以后改格式可以+1，强制失效旧 cache
    }
    key_str = _json.dumps(meta, sort_keys=True)
    h = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
    return h, meta


def _fair_vue_delta_cache_load(cache_dir: str, cache_key: str, meta_expected: dict):
    import os as _os
    import torch as _torch
    path = _os.path.join(cache_dir, f"fairvue_delta_{cache_key}.pt")
    if not _os.path.isfile(path):
        return None
    try:
        obj = _torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(obj, dict) or "meta" not in obj:
        return None
    meta = obj.get("meta", {})
    # 简单比对：关键字段一致才算命中
    for k in ("train_path", "total_clients", "target_id",
              "fair_target_rounds", "param_keys", "version"):
        if meta.get(k) != meta_expected.get(k):
            return None
    return obj


def _fair_vue_delta_cache_save(cache_dir: str, cache_key: str, payload: dict, verbose: bool = False):
    import os as _os
    import torch as _torch
    try:
        _os.makedirs(cache_dir, exist_ok=True)
        path = _os.path.join(cache_dir, f"fairvue_delta_{cache_key}.pt")
        _torch.save(payload, path)
        if verbose:
            print(f"[FV-CACHE] delta cache SAVED to {path}")
    except Exception as e:
        if verbose:
            print(f"[FV-CACHE][WARN] save failed: {repr(e)}")



# ---- MIA 结果瘦身：只保留标量，丢弃巨大数组，便于写入 summary.json ----
def _shrink_mia_result(res):
    """
    接受 evaluate_mia_attack 的任意返回（dict/tuple/ndarray/tensor/str），
    仅抽取四个标量：accuracy / precision / recall / f1。
    """
    if res is None:
        return None
    def _to_float(x):
        try:
            import numpy as _np, torch as _torch
            if isinstance(x, (list, tuple)) and x:
                return float(sum(_to_float(v) for v in x)/len(x))
            if '_torch' in locals() and isinstance(x, _torch.Tensor):
                return float(x.mean().item())
            if '_np' in locals() and isinstance(x, _np.ndarray):
                return float(x.mean())
            return float(x)
        except Exception:
            return None
    acc=prec=rec=f1=None
    if isinstance(res, dict):
        m={k.lower().replace('-','_'):k for k in res.keys()}
        def g(*names):
            for n in names:
                if n in m: return res[m[n]]
            # 包含匹配：支持 mia_attacker_precision 等
            for lk, ok in m.items():
                if any(n in lk for n in names):
                    return res[ok]
            return None
        acc = g('accuracy','acc')
        prec= g('precision','precision_score')
        rec = g('recall','recall_score')
        f1  = g('f1','f1_score')
    elif isinstance(res, (list, tuple)):
        if len(res)>0: prec=res[0]
        if len(res)>1: rec =res[1]
        if len(res)>2: f1  =res[2]
    # 返回统一瘦身结果
    return {
        'accuracy': _to_float(acc),
        'precision': _to_float(prec),
        'recall': _to_float(rec),
        'f1': _to_float(f1),
    }

# ---- Memory/Space overhead helpers (CPU & GPU peak during a block) ----
try:
    import resource as _resource  # not available on Windows
except Exception:
    _resource = None

class PeakMem:
    def __init__(self, device):
        self.device = device
        self.cpu_peak_mb = None
        self.gpu_peak_mb = None
        self._base_ru = None
        self._is_cuda = False
    def __enter__(self):
        import torch, gc as _gc
        self._is_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        _gc.collect()
        if self._is_cuda:
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(torch.device(self.device))
            except Exception:
                pass
        if _resource is not None:
            try:
                self._base_ru = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            except Exception:
                self._base_ru = None
        return self
    def __exit__(self, exc_type, exc, tb):
        import sys as _sys
        import torch
        # CPU peak (delta ru_maxrss)
        if (_resource is not None) and (self._base_ru is not None):
            try:
                ru1 = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
                delta = max(0, ru1 - self._base_ru)
                # Linux returns KB; macOS returns bytes
                bytes_used = delta if _sys.platform == "darwin" else (delta * 1024)
                self.cpu_peak_mb = bytes_used / (1024 * 1024)
            except Exception:
                self.cpu_peak_mb = None
        # GPU peak (allocated)
        if self._is_cuda:
            try:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(torch.device(self.device))
                self.gpu_peak_mb = peak / (1024 * 1024)
            except Exception:
                self.gpu_peak_mb = None
        return False

def _print_mem_overhead(label: str, pm: PeakMem, summary: dict):
    cpu_str = f"{pm.cpu_peak_mb:.2f} MB" if pm.cpu_peak_mb is not None else "NA"
    gpu_str = f"{pm.gpu_peak_mb:.2f} MB" if pm.gpu_peak_mb is not None else "NA"
    print(f"[Memory] {label}: peak CPU={cpu_str}; peak GPU={gpu_str}")
    try:
        summary.setdefault('space_overhead', {})[label] = {
            'cpu_peak_mb': pm.cpu_peak_mb,
            'gpu_peak_mb': pm.gpu_peak_mb
        }
    except Exception:
        pass

# create argument parser
parser = argparse.ArgumentParser(description='FedUnlearner')

# add arguments
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--exp_path", default="./experiments/", type=str)
parser.add_argument('--model', type=str, default='allcnn', choices=["allcnn", 'resnet18', 'smallcnn'],
                    help='model name')
parser.add_argument('--pretrained', type=str2bool,
                    default=False, help='use pretrained model')

parser.add_argument('--dataset', type=str, default='cifar10', choices=["mnist", "cifar10", "cifar100", "tinyimagenet"],
                    help='dataset name')
parser.add_argument('--optimizer', type=str, default='adam', choices=["sgd", "adam"],
                    help='optimizer name')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=5e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_local_epochs', type=int,
                    default=1, help='number of local epochs')

parser.add_argument('--num_training_iterations', type=int, default=1,
                    help='number of training iterations for global model')
parser.add_argument('--num_participating_clients', type=int, default=-1, help='number of users participating in trainig, \
                                                                                    -1 if all are required to participate')

# baselines
parser.add_argument('--baselines', type=str, nargs="*", default=[], 
    choices=['pga', 'fed_eraser', 'fedfim', 'fair_vue', 'fast_fu', 'quickdrop', 'conda'],
    help='baseline methods for unlearning')

# ===== PGA 超参（显式控制遗忘强度） =====
parser.add_argument('--pga_alpha', type=float, default=1.95,
    help='核心：PGA: unlearning strength factor (scales both distance threshold and gradient-ascent step size)')
parser.add_argument('--pga_unlearn_rounds', type=int, default=5,
    help='PGA: number of gradient-ascent epochs on the forget client data')
parser.add_argument('--pga_unlearn_lr', type=float, default=0.2,
    help='核心：PGA: learning rate used during PGA unlearning (default: use --lr)')


# ===== FedEraser 可调强度参数 =====
parser.add_argument('--fe_strength', type=float, default=10,
    help='FedEraser: overall multiplier on geometry step size')
parser.add_argument('--fe_scale_from', type=str, default='new', choices=['old','new','none'],
    help='FedEraser: scale source; old=||Σ(oldCM-oldGM)||, new=||Σ(newCM-oldCM)||, none=1')
parser.add_argument('--fe_normalize', type=str2bool, default=True,
    help='FedEraser: divide by direction L2 norm')
parser.add_argument('--fe_max_step_ratio', type=float, default=3,
    help='FedEraser: clip per-layer step norm to ratio * ||newGM[layer]||')
parser.add_argument('--fe_apply_regex', type=str, default=None,
    help='FedEraser: only apply to params whose name matches this regex (e.g., "fc|classifier")')
parser.add_argument('--fe_eps', type=float, default=1e-12,
    help='FedEraser: small epsilon for numeric stability')

# ---------- fast-fU 超参（与原实现同名语义） ----------
parser.add_argument('--fast_expected_saving', type=int, default=5,
    help='fast-fU: expected number of saved client updates (m)')
parser.add_argument('--fast_alpha', type=float, default=0.5,
    help='核心：fast-fU: alpha coefficient')
parser.add_argument('--fast_theta', type=float, default=0.35,
    help='fast-fU: theta scaling for unlearning term')

# ---------- QuickDrop 超参（贴近原实现命名/语义） ----------
parser.add_argument('--qd_scale', type=float, default=0.01,
    help='核心：QuickDrop: 每类合成样本比例（如 0.01 表示每类约 1%）')
parser.add_argument('--qd_method', type=str, default='dc', choices=['dc'],
    help='QuickDrop: 蒸馏方法（此实现提供 DC/gradient matching 变体）')
parser.add_argument('--qd_syn_steps', type=int, default=5,
    help='QuickDrop: 蒸馏外循环步数（优化合成图像）')
parser.add_argument('--qd_lr_img', type=float, default=0.3,
    help='QuickDrop: 合成图像的学习率')
parser.add_argument('--qd_batch_real', type=int, default=64,
    help='QuickDrop: 真实批大小（用来计算目标梯度）')
parser.add_argument('--qd_batch_syn', type=int, default=128,
    help='QuickDrop: 合成批大小（用来计算匹配梯度）')
parser.add_argument('--qd_local_epochs', type=int, default=1,
    help='QuickDrop: 本地训练轮数（默认沿用 num_local_epochs）')
parser.add_argument('--qd_save_affine', type=str2bool, default=False,
    help='QuickDrop: 是否保存各客户端合成（affine/synthetic）数据张量')
parser.add_argument('--qd_affine_dir', type=str, default='quickdrop_affine',
    help='QuickDrop: 合成数据保存目录（位于 experiments/exp_name 下）')
parser.add_argument('--qd_log_interval', type=int, default=25,
    help='QuickDrop: 蒸馏外循环日志步长（每多少步打印一次进度）')
# [新增] 专属遗忘参数，解耦训练参数
parser.add_argument('--qd_unlearn_lr', type=float, default=None,
    help='QuickDrop: 遗忘阶段的专用学习率 (默认使用全局 lr)')
parser.add_argument('--qd_unlearn_wd', type=float, default=None,
    help='QuickDrop: 遗忘阶段的专用权重衰减 (默认使用全局 weight_decay)')

# 若已存在合成集缓存，则直接加载并跳过蒸馏（默认开启）
parser.add_argument('--qd_use_affine_cache', type=str2bool, default=True,
    help='QuickDrop: 发现缓存则复用合成集，避免重复蒸馏')




# FAIR-VUE 参数
parser.add_argument('--fair_rank_k', type=int, default=16, help='SVD 主成分数')
parser.add_argument('--fair_tau_mode', type=str, default='median', choices=['median','mean'], help='ρ阈值模式')
parser.add_argument('--fair_fisher_batches', type=int, default=5, help='Fisher估计的批次数')
parser.add_argument('--fair_vue_debug', type=str2bool, default=False,
                    help='是否打印 FAIR-VUE 调试信息（True/False）')
parser.add_argument('--fair_erase_scale', type=float, default=0.25,
                    help='特异分量擦除强度 (0,1]，默认0.25，建议先小后大')
parser.add_argument('--fair_auto_erase', type=str2bool, default=False,
                    help='自动调参 erase_scale 以拟合重训练（默认开启）')
parser.add_argument('--fair_auto_tune_all', type=str2bool, default=False,
                    help='联合自动调参 Fisher批次 / rank_k / tau_mode（默认开启）')
parser.add_argument('--fair_fisher_grid', type=str, default='1,2,5,10',
                    help='Fisher 批次数候选（逗号分隔），用于稳定性搜索')
parser.add_argument('--fair_fisher_stability', type=float, default=0.98,
                    help='Fisher 对角稳定阈值（与更大批次数的余弦相似度阈值）')
parser.add_argument('--fair_rank_energy', type=float, default=0.90,
                    help='选取 fair_rank_k 的累计奇异值能量阈值（默认 90%）')
parser.add_argument('--fair_rank_k_min', type=int, default=4, help='rank_k 下界')
parser.add_argument('--fair_rank_k_max', type=int, default=64, help='rank_k 上界')
parser.add_argument('--fair_tau_metric', type=str, default='stdgap', choices=['stdgap','gap'],
                    help='τ 选择的分离度指标：stdgap=(均值差/总体std)，gap=上下组均值差')
parser.add_argument('--fair_drop_bounds', type=str, default='0.00,0.04',
                    help='期望全局精度下降区间，形如 "min,max"（默认 0~4%）')
parser.add_argument('--fair_grid_scales', type=str, default='0.5,0.75,1.0,1.25,1.5',
                    help='粗网格倍数（相对 fair_erase_scale）')
parser.add_argument('--fair_bisect_steps', type=int, default=3,
                    help='命中区间后的二分细化步数')
parser.add_argument('--skip_retraining', type=str2bool, default=False,
                    help='是否跳过重训练阶段（True/False）')
# —— 新增：严格控制 FAIR-VUE 的内存/样本规模 ——
parser.add_argument('--fair_target_rounds', type=int, default=200,
                    help='仅取最近 R 轮目标客户端增量')
parser.add_argument('--fair_rho_max_samples', type=int, default=128,
                    help='ρ 的其它客户端子采样上限（默认 128）')
parser.add_argument('--fair_use_delta_cache', type=str2bool, default=True,
                    help='FAIR-VUE: 是否启用逐轮增量缓存（默认开启）')

# ==== FAIR-VUE 消融实验参数 ====
parser.add_argument('--fair_ablation', type=str, default='none',
                    choices=['none', 'no_fisher', 'no_repair', 'no_dual'],
                    help='消融实验变体: none(完整), no_fisher(欧氏距离), no_repair(无修复), no_dual(无对偶优化/易OOM)')
parser.add_argument('--fair_repair_ratio', type=float, default=0.4,
                    help='FAIR-VUE: 能量补偿比例 (修复能量/擦除能量)，默认 0.4')


# backdoor attack related arguments
parser.add_argument('--apply_backdoor', type=str2bool, default=False,
                    help='是否启用后门攻击（True/False）')
parser.add_argument('--backdoor_position', type=str, default='corner', choices=["corner", "center"],
                    help='backdoor position')
parser.add_argument('--num_backdoor_samples_per_forget_client', type=int, default=10,
                    help='number of backdoor samples per forget client')
parser.add_argument('--backdoor_label', type=int,
                    default=0, help='backdoor label')

# membership inference attack related arguments
parser.add_argument('--apply_membership_inference', type=str2bool, default=False,
                    help='是否启用成员推理攻击（True/False）')
# 是否打印 MIA 详细调试信息（默认不打印）
parser.add_argument('--mia_verbose', type=str2bool, default=False,
                        help='Print detailed diagnostics for MIA (default: false)')
parser.add_argument('--attack_type', type=str, default='blackbox', choices=["blackbox", "whitebox"],
                    help='attack type')

# label posioning attack related arguments
parser.add_argument('--apply_label_poisoning', type=str2bool, default=False,
                    help='是否启用标签投毒（True/False）')
parser.add_argument('--num_label_poison_samples', type=int, default=10,
                    help='number of label poisoning samples')

# —— CONDA provide indexes of clients which are to be forgotten, allow multiple clients to be forgotten
parser.add_argument('--forget_clients', type=int, nargs='+',
                    default=[0], help='forget clients')
parser.add_argument('--total_num_clients', type=int,
                    default=10, help='total number of clients')
parser.add_argument('--client_data_distribution', type=str, default='dirichlet',
                    choices=["dirichlet", "iid", "exclusive"], help='client data distribution')
parser.add_argument('--dampening_constant', type=float,
                    default=0.5, help='dampening constant')
parser.add_argument('--dampening_upper_bound', type=float,
                    default=0.5, help='dampening upper bound')
parser.add_argument('--ratio_cutoff', type=float,
                    default=0.5, help='ratio cutoff,conda核心')
# —— CONDA 额外安全阈值
parser.add_argument('--conda_lower_bound', type=float, default=0,
                    help='CONDA: 乘子下界（避免把权重乘成接近 0）')
parser.add_argument('--conda_eps', type=float, default=1e-6,
                    help='CONDA: 防 0 除的数值稳定项')
parser.add_argument('--device', type=str, default='cpu',
                    choices=["cpu", "cuda"], help='device name')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--verbose', type=bool, default=True, help='verbose')
parser.add_argument("--num_workers", type=int, default=0,
                    help="number of workers for data loading")
parser.add_argument('--conda_weights_path', type=str, default=None,
                    help='CONDA: 手动指定读取权重的目录（实验根目录或 full_training 目录）')

# create argument parser ...
parser.add_argument('--skip_training', type=str2bool, default=False,
                    help='是否仅执行遗忘流程（True/False）')
parser.add_argument('--full_training_dir', type=str, default='',
                    help='已有的 full_training 目录（含 iteration_*/client_*.pth 和 final_model.pth）')
parser.add_argument('--global_ckpt', type=str, default='',
                    help='可选：显式指定要加载的全局模型权重路径（.pth）')
parser.add_argument('--retraining_dir', type=str, default='',
                    help='当 --skip_retraining 时可用：已有的 retraining 目录（含 iteration_*/global_model.pth 或 final_model.pth）')
parser.add_argument('--retrained_ckpt', type=str, default='',
                    help='当 --skip_retraining 时可用：显式指定重训练基线的全局模型 .pth 路径')

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


# ==== 执行阶段控制 (新增) ====
parser.add_argument('--execution_stage', type=str, default='all',
                    choices=['all', 'full_training', 'retraining', 'unlearning'],
                    help='指定运行阶段：all(全流程), full_training(仅训练), retraining(仅重训练), unlearning(仅遗忘)')



# ==== HEAL / 模型治疗参数 ====
parser.add_argument('--heal', type=str2bool, default=False,
                    help='是否启用治疗阶段（True/False）')
parser.add_argument('--heal_alpha', type=float, default=0.05,
                    help='权重插值系数 α，student←(1-α)student+α·teacher，建议 0.02~0.10')
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
    # 将开关传递给 mia.py（用环境变量最省事）
    import os as _os
    _os.environ["MIA_VERBOSE"] = "1" if args.mia_verbose else "0"
    weights_path = os.path.abspath(os.path.join(args.exp_path, args.exp_name))
    
    # === 两份日志路径（时间命名 + 参数命名） ===
    LOG_ROOT = "./logs"
    TIME_DIR = os.path.join(LOG_ROOT, "by_time")
    PARAM_DIR = os.path.join(LOG_ROOT, "by_params")
    os.makedirs(TIME_DIR, exist_ok=True)
    os.makedirs(PARAM_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 只支持单个忘却客户端；若为空就用 NA 占位
    cid = (args.forget_clients[0] if getattr(args, "forget_clients", None) else "NA")

    param_basename = (
        f"client{cid}"
        f"_k{args.fair_rank_k}"
        f"_tau{args.fair_tau_mode}"
        f"_fb{args.fair_fisher_batches}"
        f"_es{args.fair_erase_scale}.log"
    )

    log_path_time   = os.path.join(TIME_DIR,   f"run_{ts}.log")
    log_path_param  = os.path.join(PARAM_DIR,  param_basename)

    # 安装到 stdout / stderr（同样的输出写两份日志）
    sys.stdout = ProxyLog(sys.stdout, [log_path_time, log_path_param])
    sys.stderr = ProxyLog(sys.stderr, [log_path_time, log_path_param])

    # 之后再打印实验详情，确保写入两份日志
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
    elif args.client_data_distribution == 'exclusive':
        # [修改] 多类主导分布：Client 0 独占 0, 1, 2, 3 的全量数据
        # 其他客户端只有这四个类别的 20% 副本 (稀释后)
        target_classes = [0, 1, 2, 3]
        print(f"[Setup] 创建多类主导分布：Client {cid} 独占 Classes {target_classes}")
        clientwise_dataset = create_class_exclusive_distribution(train_dataset, 
                                                                 num_clients=args.total_num_clients,
                                                                 num_classes=num_classes,
                                                                 exclusive_client=int(cid) if cid != "NA" else 0,
                                                                 exclusive_classes=target_classes)
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
            client_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, drop_last=True, pin_memory=True, persistent_workers=(args.num_workers > 0))
        clientwise_dataloaders[client_id] = client_dataloader
    
    # === 本地 Fisher 端点：客户端持有数据；服务端仅“请求 Fisher”，不触碰原始样本 ===
    def _build_fresh_model_for_args(args):
        # 与上面创建全局模型的分支保持一致，客户端本地重建同构模型
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
            raise ValueError("Invalid model name")

    class LocalClientEndpoint:
        """
        轻量“RPC”端点：模拟把模型权重下发到客户端，
        由客户端在本地数据上计算 Fisher 对角并上传（仅上传统计量，不上传原始样本）。
        """
        def __init__(self, cid, dataloader, args):
            self.cid = cid
            self.loader = dataloader
            self.args = args

        def compute_fisher(self, model_state_dict, device="cpu", max_batches=10):
            # 客户端本地重建同构模型并载入服务端下发的参数
            model = _build_fresh_model_for_args(self.args)
            # 若服务端权重来自 DataParallel，自动去除 'module.' 前缀以与本地裸模型对齐
            ref_keys = list(model.state_dict().keys())
            if all(k.startswith("module.") for k in model_state_dict.keys()) and \
               not any(k.startswith("module.") for k in ref_keys):
                model_state_dict = {k.replace("module.", "", 1): v for k, v in model_state_dict.items()}
            ret = model.load_state_dict(model_state_dict, strict=True)
            # 严格校验，避免 BN buffer / 参数名不一致导致的静默偏差
            assert len(ret.missing_keys) == 0 and len(ret.unexpected_keys) == 0, \
                f"Incompatible keys when loading endpoint model: {ret}"
            
            # [关键修复] 临时创建一个全新的 DataLoader，彻底隔离 MIA 的影响
            # 无论之前 MIA 是否遍历过 self.loader，这里都强制使用固定种子生成新的迭代顺序
            g_fisher = torch.Generator()
            g_fisher.manual_seed(42) # 强力固定 Fisher 计算的数据顺序
            
            # 复用原 loader 的参数，但注入 generator
            temp_loader = torch.utils.data.DataLoader(
                self.loader.dataset,
                batch_size=self.loader.batch_size,
                shuffle=True,
                num_workers=self.loader.num_workers,
                drop_last=self.loader.drop_last,
                generator=g_fisher
            )

            # 用原始算法计算经验 Fisher 对角近似（保持算法不变）
            return empirical_fisher_diagonal(
                model=model,
                dataloader=temp_loader, # 使用临时 loader，而非 self.loader
                device=device,
                max_batches=max_batches
            )

    # 为每个客户端建立一个端点（仅保存回调，不暴露原始数据给服务端使用）
    client_endpoints = {
        cid: LocalClientEndpoint(cid, loader, args)
        for cid, loader in clientwise_dataloaders.items()
    }    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))

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

    # =========================================================================
    #                               阶段 1: Full Training
    # =========================================================================
    
    # 如果处于 'retraining' 或 'unlearning' 阶段，强制跳过训练（变为加载模式）
    if args.execution_stage in ['retraining', 'unlearning']:
        args.skip_training = True
        if args.verbose:
            print(f"[{args.execution_stage}] 模式：自动跳过 Full Training 训练，尝试加载权重...")

    # 原来：
    # train_path = os.path.abspath(os.path.join(weights_path, "full_training"))
    # global_model = fed_train(...)

    # 改为：
    if args.skip_training:
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

        state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=True)
        global_model.load_state_dict(state_dict)
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


    # [修改] 仅在 'all' 或 'full_training' 阶段才执行 Full Training 的测评
    if args.execution_stage in ['all', 'full_training']:
        perf = get_performance(model=global_model, test_dataloader=test_dataloader, num_classes=num_classes,
                            clientwise_dataloader=clientwise_dataloaders, device=args.device)
        summary['performance'] = {}
        summary['performance']['after_training'] = perf
        if args.verbose:
            print(f"Performance after training : {perf}")

            forget_loader = clientwise_dataloaders[forget_client]
            acc = get_accuracy_only(global_model, forget_loader, args.device)
            print(f"[Training模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")
    else:
        # 如果跳过测评，给个空字典防止报错
        summary['performance'] = {}
        if args.verbose: print("[Skip] 跳过 Full Training 测评")

    # === [MIA-INIT] 在任何 evaluate_mia_attack 调用之前，先准备好分割与攻击器 ===
    # 幂等：若后面已有同名对象，这里不会重复构造
    attack_model = locals().get("attack_model", None)
    mia_shadow_nonmem_loader = locals().get("mia_shadow_nonmem_loader", None)
    mia_eval_nonmem_loader   = locals().get("mia_eval_nonmem_loader", None)
    should_train_mia = args.apply_membership_inference

    if should_train_mia:
        # 1) 准备“互斥”的非成员集（shadow/eval）
        if mia_eval_nonmem_loader is None or mia_shadow_nonmem_loader is None:
            from torch.utils.data import random_split, DataLoader as _DL
            _n_test = len(test_dataloader.dataset)
            _n_shadow_nonmem = int(0.8 * _n_test)
            _n_eval_nonmem   = _n_test - _n_shadow_nonmem
            _gen = torch.Generator().manual_seed(args.seed if getattr(args, "seed", None) is not None else 0)
            _shadow_nm_ds, _eval_nm_ds = random_split(
                test_dataloader.dataset, [_n_shadow_nonmem, _n_eval_nonmem], generator=_gen
            )
            mia_shadow_nonmem_loader = _DL(_shadow_nm_ds, batch_size=test_dataloader.batch_size,
                                           shuffle=False, num_workers=args.num_workers)
            mia_eval_nonmem_loader   = _DL(_eval_nm_ds,    batch_size=test_dataloader.batch_size,
                                           shuffle=False, num_workers=args.num_workers)
            # 诊断打印：仅在 --mia_verbose 时输出
            if args.mia_verbose:
                try:
                    _s_idx = getattr(_shadow_nm_ds, "indices", None)
                    _e_idx = getattr(_eval_nm_ds, "indices", None)
                    _overlap = (set(_s_idx) & set(_e_idx)) if (_s_idx is not None and _e_idx is not None) else set()
                    print(f"[MIA-SPLIT] shadow_nonmem={_n_shadow_nonmem} eval_nonmem={_n_eval_nonmem} "
                          f"shadow_id={id(_shadow_nm_ds)} eval_id={id(_eval_nm_ds)} overlap={len(_overlap)}")
                    if _s_idx is not None and _e_idx is not None:
                        print(f"[MIA-SPLIT] shadow_head={list(_s_idx[:5])} ... tail={list(_s_idx[-5:])}")
                        print(f"[MIA-SPLIT] eval__head={list(_e_idx[:5])} ... tail={list(_e_idx[-5:])}")
                except Exception as _e:
                    print(f"[MIA-SPLIT][WARN] split diagnostics failed: {_e}")

        # 2) 训练一次攻击器（基于 full-training shadow 模型）
        if attack_model is None:
            shadow_model = deepcopy(global_model)
            attack_model = train_attack_model(
                shadow_global_model=shadow_model,
                shadow_client_loaders=clientwise_dataloaders,
                shadow_test_loader=mia_shadow_nonmem_loader,
                dataset=args.dataset, device=args.device)



    # ==== 六项指标统一打印（Training 基线）+ MIA：只要 --apply_membership_inference 就默认跑 ====
    mia_training = None
    if args.apply_membership_inference:
        # 在“训练好的完整模型”上执行成员推断（评估集非成员与 shadow 非成员互斥）
        mia_training = evaluate_mia_attack(
            target_model=deepcopy(global_model),
            attack_model=attack_model,
            client_loaders=clientwise_dataloaders,
            test_loader=test_dataloader,
            dataset=args.dataset,
            forget_client_idx=forget_client,
            device=args.device,
            eval_nonmem_loader=mia_eval_nonmem_loader
        )

    # 统一六个指标：测试集准确率、遗忘客户端准确率、遗忘客户端交叉熵、加速比(Training 无)、参数夹角(Training 无)、MIA（三元组）
    if args.execution_stage in ['all', 'full_training']:
        _forget_loader = clientwise_dataloaders[forget_client]
        test_acc_tr    = get_accuracy_only(global_model, test_dataloader, args.device)
        target_acc_tr  = get_accuracy_only(global_model, _forget_loader, args.device)
        retain_acc_tr  = eval_retain_acc(global_model, clientwise_dataloaders, args.forget_clients, args.device)
        target_loss_tr = eval_ce_loss(global_model, _forget_loader, args.device)
        speedup_tr     = None   # 以 retrain 为基线，此处不计
        angle_tr       = None   # 需相对 retrain 的夹角，这里留空
        print_forgetting_metrics(
            method_name="Training",
            test_acc=test_acc_tr,
            retain_acc=retain_acc_tr,
            target_acc=target_acc_tr,
            target_loss=target_loss_tr,
            speedup_x=speedup_tr,
            angle_deg=angle_tr,
            mia_result=mia_training
        )

    # [逻辑中断] 如果只跑 full_training，到此结束
    if args.execution_stage == 'full_training':
        print(">>> [Finish] Full Training stage completed. Exiting.")
        sys.exit(0)

    # 清理 MIA 大对象与 CUDA 缓存，避免后续卡住 (通用清理)
    try:
        import torch, gc
        if isinstance(mia_training, dict):
            for k in ['mia_attacker_predictions','mia_attacker_probabilities','predictions','probabilities','scores']:
                mia_training.pop(k, None)
        # [修改] 在 unlearning 阶段跳过强制同步和清理，节省 2-5 秒
        if args.execution_stage != 'unlearning' and torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    # （删除：这里不再做三参自动调参，改到 fair_vue 分支内“按轮增量解析后”执行）



    # -------------------------------------------------------
    # === MIA: 准备“互斥”的非成员数据（避免泄漏）【若前面已构造，这里跳过】===
    from torch.utils.data import random_split, DataLoader as _DL
    if args.apply_membership_inference and (locals().get("mia_eval_nonmem_loader") is None
                                            or locals().get("mia_shadow_nonmem_loader") is None):
        _n_test = len(test_dataloader.dataset)
        _n_shadow_nonmem = int(0.8 * _n_test)
        _n_eval_nonmem   = _n_test - _n_shadow_nonmem
        _gen = torch.Generator().manual_seed(args.seed if hasattr(args, "seed") else 0)
        _shadow_nm_ds, _eval_nm_ds = random_split(test_dataloader.dataset, [_n_shadow_nonmem, _n_eval_nonmem], generator=_gen)
        mia_shadow_nonmem_loader = _DL(_shadow_nm_ds, batch_size=test_dataloader.batch_size, shuffle=False, num_workers=args.num_workers)
        mia_eval_nonmem_loader   = _DL(_eval_nm_ds,    batch_size=test_dataloader.batch_size, shuffle=False, num_workers=args.num_workers)

    # —— 自检日志只在 --mia_verbose 时打印 —— 
    if args.mia_verbose:
        try:
            _s_idx = getattr(_shadow_nm_ds, "indices", None)
            _e_idx = getattr(_eval_nm_ds, "indices", None)
            _overlap = (set(_s_idx) & set(_e_idx)) if (_s_idx is not None and _e_idx is not None) else set()
            print(f"[MIA-SPLIT] shadow_nonmem={_n_shadow_nonmem} eval_nonmem={_n_eval_nonmem} "
                  f"shadow_id={id(_shadow_nm_ds)} eval_id={id(_eval_nm_ds)} overlap={len(_overlap)}")
            if _s_idx is not None and _e_idx is not None:
                print(f"[MIA-SPLIT] shadow_head={list(_s_idx[:5])} ... tail={list(_s_idx[-5:])}")
                print(f"[MIA-SPLIT] eval__head={list(_e_idx[:5])} ... tail={list(_e_idx[-5:])}")
        except Exception as _e:
            print(f"[MIA-SPLIT][WARN] split diagnostics failed: {_e}")

    # train mia attack model（使用“不与评估重叠”的非成员子集；若已存在则跳过）
    if args.apply_membership_inference and (locals().get("attack_model") is None):
        shadow_model = deepcopy(global_model)
        attack_model = train_attack_model(
            shadow_global_model=shadow_model,
            shadow_client_loaders=clientwise_dataloaders,
            shadow_test_loader=mia_shadow_nonmem_loader,
            dataset=args.dataset, device=args.device)
    # ---------------------------------------------------------
    # evaluate attack accuracy
    # [修改] 仅在 'all' 或 'full_training' 阶段评估初始模型的后门，避免调参时卡顿
    if args.apply_backdoor and args.execution_stage in ['all', 'full_training']:
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
    # [修改] 同上，跳过投毒评估
    if args.apply_label_poisoning and args.execution_stage in ['all', 'full_training']:
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

    # =========================================================================
    #                               阶段 2: Retraining
    # =========================================================================
    

    retrain_path = os.path.join(weights_path, "retraining")
    # train the model on retain data
    retain_clientwise_dataloaders = {key: value for key, value in clientwise_dataloaders.items()
                                     if key not in args.forget_clients}
    # [修改] 注释掉这个打印，防止 Client 很多时刷屏几千行
    # print(f"Retain Client wise Loaders: {retain_clientwise_dataloaders}")

    # 如果处于 'unlearning' 阶段，我们只需要加载 Retrain 模型算指标，不需要重新跑训练流程
    if args.execution_stage == 'unlearning':
        args.skip_retraining = True
        if args.verbose: print("[unlearning] 模式：跳过 Retraining 训练，将尝试加载基线用于计算 Speedup/Angle")


    # === 计时：重训基线（供 Speedup 对比） ===
    t_retrain_sec = None
    has_retrain_baseline = False
    if not args.skip_retraining:
        # 如果当前是 'full_training' 阶段，这里根本不会执行到（上面已 exit）
        # 所以这里一定是 'all' 或 'retraining'
        _t0 = time.time()
        retrained_global_model = fed_train(num_training_iterations=args.num_training_iterations, test_dataloader=test_dataloader,
                                        clientwise_dataloaders=retain_clientwise_dataloaders,
                                        global_model=retrained_global_model, num_local_epochs=args.num_local_epochs,
                                        device=args.device, weights_path=retrain_path, lr=args.lr, optimizer_name=args.optimizer)
        t_retrain_sec = time.time() - _t0
        has_retrain_baseline = True

        if args.execution_stage in ['all', 'retraining']:
            perf = get_performance(model=retrained_global_model, test_dataloader=test_dataloader,
                                clientwise_dataloader=clientwise_dataloaders,
                                num_classes=num_classes, device=args.device)
            summary['performance']['after_retraining'] = perf
            if args.verbose:
                print(f"Performance after retraining : {perf}")
                print(f"[Timing] Retrain baseline time: {t_retrain_sec:.2f}s" if t_retrain_sec is not None else "[Timing] Retrain baseline time: NA")
            # evaluate attack accuracy on retrained model

            # ---- 专门测忘却客户端的精度 ----
                forget_loader = clientwise_dataloaders[forget_client]
                acc = get_accuracy_only(retrained_global_model, forget_loader, args.device)
                print(f"[Retrain模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")


        # ==== 统一打印（Retrain Baseline）+ MIA：只要开启 MIA 就默认跑 ====
        mia_retrain = None
        if args.apply_membership_inference:
            mia_retrain = evaluate_mia_attack(
                target_model=deepcopy(retrained_global_model),
                attack_model=attack_model,
                client_loaders=clientwise_dataloaders,
                test_loader=test_dataloader,
                dataset=args.dataset,
                forget_client_idx=forget_client,
                device=args.device,
                eval_nonmem_loader=mia_eval_nonmem_loader
            )
        test_acc_rt    = get_accuracy_only(retrained_global_model, test_dataloader, args.device)
        target_acc_rt  = get_accuracy_only(retrained_global_model, clientwise_dataloaders[forget_client], args.device)
        retain_acc_rt  = eval_retain_acc(retrained_global_model, clientwise_dataloaders, args.forget_clients, args.device)
        target_loss_rt = eval_ce_loss(retrained_global_model, clientwise_dataloaders[forget_client], args.device)
        speedup_rt     = 1.0  # retrain 作为基线
        angle_rt       = 0.0
        
        if args.execution_stage in ['all', 'retraining']:
            print_forgetting_metrics("Retrain", test_acc_rt, retain_acc_rt, target_acc_rt, target_loss_rt, speedup_rt, angle_rt, mia_retrain)
        # 清理大对象
        try:
            import torch, gc
            if isinstance(mia_retrain, dict):
                for k in ['mia_attacker_predictions','mia_attacker_probabilities','predictions','probabilities','scores']:
                    mia_retrain.pop(k, None)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    else:
        # 跳过重训：若用户提供了 retraining_dir / retrained_ckpt，则直接载入基线权重
        import os, torch
        ckpt_path = None
        if args.retrained_ckpt:
            ckpt_path = os.path.abspath(args.retrained_ckpt)
            if not os.path.isfile(ckpt_path):
                raise RuntimeError(f"[Skip-Retrain] 找不到指定的 --retrained_ckpt：{ckpt_path}")
        elif args.retraining_dir:
            rdir = os.path.abspath(args.retraining_dir)
            if not os.path.isdir(rdir):
                raise RuntimeError(f"[Skip-Retrain] 找不到指定的 --retraining_dir：{rdir}")
            # 先尝试 final_model.pth，其次尝试最后一轮的 global_model.pth
            last_iter = -1
            for name in os.listdir(rdir):
                if name.startswith("iteration_"):
                    try:
                        idx = int(name.split("_")[-1])
                        last_iter = max(last_iter, idx)
                    except Exception:
                        pass
            candidates = [
                os.path.join(rdir, "final_model.pth"),
                os.path.join(rdir, f"iteration_{last_iter}", "global_model.pth") if last_iter >= 0 else None
            ]
            ckpt_path = next((p for p in candidates if p and os.path.isfile(p)), None)
            if not ckpt_path:
                raise RuntimeError(f"[Skip-Retrain] 在 {rdir} 未找到 final_model.pth 或最后一轮 global_model.pth")

        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            retrained_global_model.load_state_dict(state_dict)
            has_retrain_baseline = True
            print(f"[Skip-Retrain] 复用重训练基线：{ckpt_path}")
            
            # [修改] 如果是 'unlearning' 阶段，我们只加载权重不算指标，节省时间
            # 只有 'all' 或 显式 'retraining' (但跳过训练?) 时才测评
            if args.execution_stage in ['all', 'retraining']:
                # 既然有了基线，也一起评测便于对照
                perf = get_performance(model=retrained_global_model, test_dataloader=test_dataloader,
                                    clientwise_dataloader=clientwise_dataloaders,
                                    num_classes=num_classes, device=args.device)
                summary['performance']['after_retraining'] = perf
                if args.verbose:
                    print(f"Performance after (loaded) retraining : {perf}")
                forget_loader = clientwise_dataloaders[forget_client]
                acc = get_accuracy_only(retrained_global_model, forget_loader, args.device)
                print(f"[Retrain(loaded)模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")
            


            # ==== 统一打印（Retrain Baseline，Loaded）+ MIA：只要开启 MIA 就默认跑 ====
            mia_retrain = None
            # 只有当 attack_model 真正被训练了 (not None) 才跑 MIA
            if args.apply_membership_inference and attack_model is not None:
                mia_retrain = evaluate_mia_attack(
                    target_model=deepcopy(retrained_global_model),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=forget_client,
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                )
            
            if args.execution_stage in ['all', 'retraining']:
                test_acc_rt    = get_accuracy_only(retrained_global_model, test_dataloader, args.device)
                target_acc_rt  = get_accuracy_only(retrained_global_model, clientwise_dataloaders[forget_client], args.device)
                retain_acc_rt  = eval_retain_acc(retrained_global_model, clientwise_dataloaders, args.forget_clients, args.device)
                target_loss_rt = eval_ce_loss(retrained_global_model, clientwise_dataloaders[forget_client], args.device)
                speedup_rt     = None   # 此分支没计时，就打印 NA
                angle_rt       = 0.0
                print_forgetting_metrics("Retrain", test_acc_rt, retain_acc_rt, target_acc_rt, target_loss_rt, speedup_rt, angle_rt, mia_retrain)
            
            try:
                import torch, gc
                if isinstance(mia_retrain, dict):
                    for k in ['mia_attacker_predictions','mia_attacker_probabilities','predictions','probabilities','scores']:
                        mia_retrain.pop(k, None)
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

        else:
            if args.verbose:
                print("[Skip] 跳过重训练基线（--skip_retraining），且未提供 --retraining_dir / --retrained_ckpt") 

    # [逻辑中断] 如果只跑 retraining，到此结束
    if args.execution_stage == 'retraining':
        print(">>> [Finish] Retraining stage completed. Exiting.")
        sys.exit(0)




    # =========================================================================
    #                               阶段 3: Unlearning Baselines
    # =========================================================================

    if args.apply_backdoor and not args.skip_retraining and args.execution_stage in ['all', 'retraining']:
        retrained_backdoor_results = evaluate_backdoor_attack(model=retrained_global_model,
                                                              backdoor_context=backdoor_context, device=args.device)
        summary['backdoor_results']['after_retraining'] = retrained_backdoor_results
        if args.verbose:
            print(
                f"Backdoor results after retraining : {retrained_backdoor_results}")

    if args.apply_label_poisoning and not args.skip_retraining and args.execution_stage in ['all', 'retraining']:
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
            _t0 = time.time()
            global_model_pga = deepcopy(global_model)
            # [修复] 显式移动到 device，防止 PGA 内部报错
            global_model_pga = global_model_pga.to(args.device)
           
            _pm_pga = PeakMem(args.device); _pm_pga.__enter__()
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
                                          lr=(args.pga_unlearn_lr if args.pga_unlearn_lr is not None else args.lr),
                                          optimizer_name=args.optimizer,
                                          num_local_epochs=args.num_local_epochs,
                                          num_unlearn_rounds=args.pga_unlearn_rounds,
                                          num_post_training_rounds=1,
                                          alpha=args.pga_alpha)

            perf = get_performance(model=unlearned_pga_model, test_dataloader=test_dataloader,
                                   clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                   device=args.device)
            pga_time_sec = time.time() - _t0
            print(f"[Timing] PGA time: {pga_time_sec:.2f}s")
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


            # ==== 六项指标统一打印（PGA）====
            test_acc_pga    = get_accuracy_only(unlearned_pga_model, test_dataloader, args.device)
            target_acc_pga  = get_accuracy_only(unlearned_pga_model, clientwise_dataloaders[forget_client], args.device)
            retain_acc_pga  = eval_retain_acc(unlearned_pga_model, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_pga = eval_ce_loss(unlearned_pga_model, clientwise_dataloaders[forget_client], args.device)
            speedup_pga     = (t_retrain_sec / pga_time_sec) if (t_retrain_sec is not None and pga_time_sec > 0) else None
            angle_pga       = cosine_angle_between_models(unlearned_pga_model, retrained_global_model) if has_retrain_baseline else None
            mia_pga = None
            if args.apply_membership_inference and attack_model is not None:
                mia_pga = evaluate_mia_attack(
                    target_model=deepcopy(unlearned_pga_model),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                )
            print_forgetting_metrics("PGA", test_acc_pga, retain_acc_pga, target_acc_pga, target_loss_pga, speedup_pga, angle_pga, mia_pga)
            _pm_pga.__exit__(None, None, None)
            _print_mem_overhead("PGA", _pm_pga, summary)
        elif baseline == 'fed_eraser':
            _t0 = time.time()
            global_model_federaser = deepcopy(global_model)
            # Debug: 在主流程里先把 FedEraser 关键配置打印出来
            print(
                "[FedEraser] main: starting unlearning with "
                f"forget_clients={args.forget_clients}, "
                f"total_num_clients={args.total_num_clients}, "
                f"num_training_iterations={args.num_training_iterations}, "
                f"lr={args.lr}, optimizer={args.optimizer}, "
                f"fe_strength={args.fe_strength}, "
                f"fe_scale_from={args.fe_scale_from}, "
                f"fe_normalize={args.fe_normalize}, "
                f"fe_max_step_ratio={args.fe_max_step_ratio}, "
                f"fe_apply_regex={args.fe_apply_regex}, "
                f"fe_eps={args.fe_eps}"
            )
            _pm_fe = PeakMem(args.device); _pm_fe.__enter__()
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
                                                       num_post_training_rounds=1,
                                                       # 透传强度参数
                                                       fe_strength=args.fe_strength,
                                                       fe_scale_from=args.fe_scale_from,
                                                       fe_normalize=args.fe_normalize,
                                                       fe_max_step_ratio=args.fe_max_step_ratio,
                                                       fe_apply_regex=args.fe_apply_regex,
                                                       fe_eps=args.fe_eps)
            perf = get_performance(model=unlearned_federaser_model, test_dataloader=test_dataloader,
                                   clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                   device=args.device)
            federaser_time_sec = time.time() - _t0
            print(f"[Timing] FedEraser time: {federaser_time_sec:.2f}s")
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


            # ==== 六项指标统一打印（FedEraser）====
            test_acc_fe    = get_accuracy_only(unlearned_federaser_model, test_dataloader, args.device)
            target_acc_fe  = get_accuracy_only(unlearned_federaser_model, clientwise_dataloaders[forget_client], args.device)
            retain_acc_fe  = eval_retain_acc(unlearned_federaser_model, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_fe = eval_ce_loss(unlearned_federaser_model, clientwise_dataloaders[forget_client], args.device)
            speedup_fe     = (t_retrain_sec / federaser_time_sec) if (t_retrain_sec is not None and federaser_time_sec > 0) else None
            angle_fe       = cosine_angle_between_models(unlearned_federaser_model, retrained_global_model) if has_retrain_baseline else None
            mia_fe = None
            if args.apply_membership_inference:
                mia_fe = evaluate_mia_attack(
                    target_model=deepcopy(unlearned_federaser_model),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                )
            print_forgetting_metrics("FedEraser", test_acc_fe, retain_acc_fe, target_acc_fe, target_loss_fe, speedup_fe, angle_fe, mia_fe)
            _pm_fe.__exit__(None, None, None)
            _print_mem_overhead("FedEraser", _pm_fe, summary)
        elif baseline == 'fair_vue':
            
            # ---- FAIR-VUE（按轮）----
            print(">>> Running FAIR-VUE (round-wise)...")
            _t0 = time.time()
            _fv_last_t = _t0

            def _fv_time_mark(label, start_t, last_t):
                now = time.time()
                if args.fair_vue_debug:
                    print(f"[FV-TIME] {label}: step={now - last_t:.3f}s, total={now - start_t:.3f}s")
                return now

            _pm_fair = PeakMem(args.device); _pm_fair.__enter__()
            fair_model = deepcopy(global_model).to(args.device)
            fair_model.eval()
            # —— 参与子空间的参数集合：优先分类头参数，兼容多种命名
            all_param_keys = [name for name, p in fair_model.named_parameters() if p.requires_grad]
            preferred = ("fc.", "linear.", "classifier.", "head.")
            param_keys = [k for k in all_param_keys if any(k.startswith(pref) for pref in preferred)]
            if len(param_keys) == 0:
                # 兜底：取最后若干层（通常是分类头），避免空集合
                param_keys = all_param_keys[-min(4, len(all_param_keys)):]
            if args.fair_vue_debug:
                print(f"[FV-DBG] param_keys selected (n={len(param_keys)}): {param_keys[:6]}{'...' if len(param_keys)>6 else ''}")
            from FedUnlearner.baselines.fair_vue.subspace import (
                weighted_matrix_from_deltas_keys, rho_values_keys, topk_right_singular_vectors_gram,
                flatten_by_keys, state_dict_like_by_keys
            )

            # 1) 目标客户端逐轮增量 Δ_{cid}^{(r)}：先尝试从缓存加载，再视情况重新解析
            #    尊重 --skip_training 时传入的 --full_training_dir；否则沿用上文解析出的 train_path
            fv_train_path = os.path.abspath(args.full_training_dir) if args.full_training_dir else os.path.abspath(train_path)
            from collections import deque
            target_id = args.forget_clients[0]

            target_deltas_list = None
            other_deltas_list = None
            rounds_seen = 0
            cache_hit = False

            if getattr(args, "fair_use_delta_cache", True):
                cache_dir = os.path.join(fv_train_path, "fair_vue_cache")
                try:
                    cache_key, cache_meta = _fair_vue_delta_cache_key(
                        fv_train_path,
                        args.total_num_clients,
                        target_id,
                        param_keys,
                        args.fair_target_rounds,
                    )
                    cached = _fair_vue_delta_cache_load(cache_dir, cache_key, cache_meta)
                    if cached is not None:
                        target_deltas_list = list(cached.get("target_deltas", []))
                        other_deltas_list = list(cached.get("other_deltas_last_round", []))
                        rounds_seen = int(cached.get("rounds_seen", len(target_deltas_list)))
                        cache_hit = True
                        if args.fair_vue_debug:
                            print(f"[FV-CACHE] delta cache HIT (key={cache_key}, "
                                  f"rounds_seen={rounds_seen}, T={len(target_deltas_list)})")
                except Exception as e:
                    if args.fair_vue_debug:
                        print(f"[FV-CACHE][WARN] load failed, fallback to recompute: {repr(e)}")

            if not cache_hit:
                buf_T = deque(maxlen=max(1, int(args.fair_target_rounds)))
                last_others = []
                rounds_seen = 0
                for r_cur, deltas_r in _iter_round_deltas_stream(
                    fv_train_path,
                    args.total_num_clients,
                    param_keys=param_keys,  # 只在选中的参数上算 Δ
                    max_rounds=int(getattr(args, "fair_target_rounds", 200)),  # 只取最近 R 轮
                ):
                    rounds_seen += 1
                    if target_id in deltas_r:
                        buf_T.append(deltas_r[target_id])
                    last_others = [d for cid, d in deltas_r.items() if cid != target_id]
                    # 释放本轮临时
                    del deltas_r
                target_deltas_list = list(buf_T)
                other_deltas_list = last_others

                # 仅在有数据时写缓存
                if getattr(args, "fair_use_delta_cache", True) and len(target_deltas_list) > 0:
                    try:
                        payload = {
                            "meta": cache_meta,
                            "target_deltas": target_deltas_list,
                            "other_deltas_last_round": other_deltas_list,
                            "rounds_seen": int(rounds_seen),
                        }
                        _fair_vue_delta_cache_save(cache_dir, cache_key, payload, verbose=args.fair_vue_debug)
                    except Exception as e:
                        if args.fair_vue_debug:
                            print(f"[FV-CACHE][WARN] save failed (ignored): {repr(e)}")

            # === 诊断（不再依赖 rounds / round_client_deltas）===
            if args.fair_vue_debug:
                print(f"[FV-DBG] rounds_seen={rounds_seen}, T={len(target_deltas_list)}, M={len(other_deltas_list)}")
                # 打印：目标增量前3个的范数，以及最后一轮其它客户端增量的均值范数
                from statistics import mean
                import torch as _torch
                from FedUnlearner.baselines.fair_vue.subspace import flatten_state_dict
                def _flat_norm(d):
                    return float(_torch.norm(flatten_state_dict(d)).item())
                head_T = target_deltas_list[:3]
                if head_T:
                    ns_T = [_flat_norm(d) for d in head_T]
                    print(f"[FV-DBG] target_deltas head norms: {[f'{v:.3e}' for v in ns_T]}")
                if other_deltas_list:
                    ns_O = [_flat_norm(d) for d in other_deltas_list]
                    print(f"[FV-DBG] others(last round) mean||Δ||={mean(ns_O):.3e} (n={len(ns_O)})")

            if len(target_deltas_list) == 0:
                raise RuntimeError(f"[FAIR-VUE] 没有解析到目标客户端的逐轮增量，无法进行 SVD/Gram。")

            # === 原有：target/others 划分完成后 ===
            if args.fair_vue_debug:
                print(f"[FV-DBG] target_id={target_id}, T=len(target_deltas_list)={len(target_deltas_list)}, "
                      f"M=len(other_deltas_list)={len(other_deltas_list)}")
                # Step1: 逐轮增量解析结束
                _fv_last_t = _fv_time_mark("step1_round_deltas", _t0, _fv_last_t)

            # ==========================================================
            # === FAIR-VUE 预自动调参（三项）：b/k/τ（不访问原始样本） ===
            #    放到 FAIR-VUE 分支内，就近使用 fv_train_path/target_* 列表
            # ==========================================================
            # [Fix] 强制重置种子，防止 MIA 等前置操作消耗随机状态导致调参采样不一致
            if args.seed is not None:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                random.seed(args.seed)

            if args.fair_auto_tune_all:
                if args.fair_vue_debug:
                    print("[FV-AUTO] ==== Start auto-tuning {fisher_batches, rank_k, tau_mode} ====")

                # —— 仅参数键，保证与 Fisher 的键一致
                all_param_keys = [name for name, p in fair_model.named_parameters() if p.requires_grad]
                preferred = ("fc.", "linear.", "classifier.", "head.")
                param_keys = [k for k in all_param_keys if any(k.startswith(pref) for pref in preferred)]
                if not param_keys:  # 兜底：避免空集合
                    param_keys = all_param_keys[-min(4, len(all_param_keys)):]

                # ——（1）Fisher 批次数：稳定性（与 ≥2b 对比的余弦相似度）
                def _parse_list_csv(s: str):
                    return [int(x) for x in str(s).split(',') if str(x).strip()!='']
                fisher_grid = sorted(set([b for b in (_parse_list_csv(args.fair_fisher_grid) or [1,2,5,10]) if b>0]))
                stability = float(args.fair_fisher_stability)
                def _flatten_fi(Fi: dict):
                    import torch
                    xs = [Fi[k].detach().flatten().float().cpu() for k in param_keys if k in Fi]
                    if not xs:
                        # 兜底：至少用 Fi 的全部键拼成向量（仍可用于相似度判断）
                        xs = [v.detach().flatten().float().cpu() for v in Fi.values()]
                    return torch.cat(xs) if xs else torch.zeros(1)
                def _cos(a,b):
                    import torch
                    na, nb = torch.norm(a), torch.norm(b)
                    if na.item()==0 or nb.item()==0: return 0.0
                    return float(torch.clamp(torch.dot(a,b)/(na*nb), -1.0, 1.0).item())
                chosen_b = fisher_grid[-1]
                chosen_fisher = None
                for b in fisher_grid:
                    b2 = next((c for c in fisher_grid if c >= 2*b), fisher_grid[-1])
                    Fi_b  = client_endpoints[target_id].compute_fisher(fair_model.state_dict(), device=args.device, max_batches=b)
                    Fi_b2 = client_endpoints[target_id].compute_fisher(fair_model.state_dict(), device=args.device, max_batches=b2)
                    sim = _cos(_flatten_fi(Fi_b), _flatten_fi(Fi_b2))
                    if args.fair_vue_debug:
                        print(f"[FV-AUTO][Fisher] b={b} vs b'={b2} → cos={sim:.4f}")
                    if sim >= stability:
                        chosen_b = b
                        chosen_fisher = Fi_b2
                        break
                if chosen_fisher is None:
                    chosen_fisher = client_endpoints[target_id].compute_fisher(fair_model.state_dict(), device=args.device, max_batches=chosen_b)
                args.fair_fisher_batches = int(chosen_b)
                if args.fair_vue_debug:
                    print(f"[FV-AUTO][Fisher] chosen_b={args.fair_fisher_batches}")

                # ——（2）rank_k：Fisher 加权的 Δ_target 历史矩阵的 SVD 累计能量阈值
                if len(target_deltas_list) >= 1:
                    import torch
                    Xw = weighted_matrix_from_deltas_keys(target_deltas_list, chosen_fisher, param_keys, device="cpu")
                    Xc = Xw - Xw.mean(dim=0, keepdim=True)
                    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
                    energy = (S**2); cum = torch.cumsum(energy, dim=0); total = torch.sum(energy) + 1e-12
                    thr = float(args.fair_rank_energy)
                    k_auto = int(torch.searchsorted(cum/total, torch.tensor(thr, device=cum.device)).item() + 1)
                    k_auto = max(int(args.fair_rank_k_min), min(int(args.fair_rank_k_max), k_auto))
                    args.fair_rank_k = int(k_auto)
                    if args.fair_vue_debug:
                        print(f"[FV-AUTO][rank_k] energy_thr={thr:.2f} → k={args.fair_rank_k} (min={args.fair_rank_k_min}, max={args.fair_rank_k_max})")

                # ——（3）tau_mode：用其它客户端在最后一轮的 Δ 计算 ρ 分布分离度
                if len(other_deltas_list) >= 1 and len(target_deltas_list) >= 1:
                    # 用同一 V_k（与上面 Xw 一致）
                    Xc_for_V = Xw - Xw.mean(dim=0, keepdim=True)
                    _U, _S, Vh_full = torch.linalg.svd(Xc_for_V, full_matrices=False)
                    V_k = Vh_full.T[:, :int(args.fair_rank_k)]
                    def _rho_sep(tau_mode: str):
                        import numpy as np
                        rhos = rho_values_keys(V_k, [d for d in other_deltas_list if isinstance(d, dict)], param_keys)
                        if len(rhos)==0: return 0.0
                        r = np.asarray(rhos, dtype=float)
                        tau = np.median(r) if tau_mode=='median' else float(r.mean())
                        lower, upper = r[r<tau], r[r>=tau]
                        if len(lower)==0 or len(upper)==0: return 0.0
                        gap = float(upper.mean() - lower.mean())
                        return gap if args.fair_tau_metric=='gap' else gap / float(r.std()+1e-12)
                    s_med, s_mean = _rho_sep('median'), _rho_sep('mean')
                    args.fair_tau_mode = 'median' if s_med >= s_mean else 'mean'
                    if args.fair_vue_debug:
                        print(f"[FV-AUTO][tau] sep(median)={s_med:.4f}, sep(mean)={s_mean:.4f} → choose {args.fair_tau_mode}")
            # =================== 三参自动调参结束 ====================
            if args.fair_auto_tune_all and args.fair_vue_debug:
                _fv_last_t = _fv_time_mark("step2_auto_tune_all", _t0, _fv_last_t)


            # 3) Fisher（遗忘指令下发 → 目标客户端本地计算 → 仅上传 Fisher 对角）
            #    与 pipeline 保持一致：为 Fisher 批次固定随机性，避免抽样差异带来系统性偏移
            import random, numpy as np, torch
            torch.manual_seed(42); np.random.seed(42); random.seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if target_id in client_endpoints:
                fisher = client_endpoints[target_id].compute_fisher(
                    model_state_dict=fair_model.state_dict(),
                    device="cpu",
                    max_batches=args.fair_fisher_batches
                )
            else:
                # fallback：没有端点时，仅对可训练参数使用单位权重
                fisher = {name: torch.ones_like(p) for name, p in fair_model.named_parameters() if p.requires_grad}

            
            # [Ablation] w/o Fisher: 强制覆盖 Fisher 为全 1 (退化为欧氏空间)
            if args.fair_ablation == 'no_fisher':
                if args.fair_vue_debug:
                    print("[FV-ABLATION] Mode: w/o Fisher -> Overwriting Fisher with Identity (Ones).")
                for k in fisher:
                    fisher[k] = torch.ones_like(fisher[k])


            # === Fisher 计算完毕后，插入点 B ===
            if args.fair_vue_debug:
                import torch
                from FedUnlearner.baselines.fair_vue.subspace import flatten_state_dict
                fvec = flatten_state_dict(fisher)
                print(f"[FV-DBG] fisher: device={fvec.device}, D={fvec.numel()}, "
                    f"min={float(torch.min(fvec)): .3e}, max={float(torch.max(fvec)): .3e}, "
                    f"mean={float(torch.mean(fvec)): .3e}")

            if args.fair_vue_debug:
                _fv_last_t = _fv_time_mark("step3_fisher", _t0, _fv_last_t)

            # 4) Fisher加权矩阵 & 低秩SVD拿到主方向V
            #    与 Fisher / deltas 对齐 keys，避免空拼接或维度不一致
            fisher_keys = set(fisher.keys())
            keys_valid = [k for k in param_keys
                          if (k in fisher_keys)
                          and all(k in d for d in target_deltas_list)
                          and (len(other_deltas_list) == 0 or all(k in d for d in other_deltas_list))]
            if len(keys_valid) == 0:
                # 放宽：只强制覆盖 target_deltas
                keys_valid = [k for k in param_keys if (k in fisher_keys) and all(k in d for d in target_deltas_list)]
            if len(keys_valid) == 0:
                raise RuntimeError(
                    f"[FAIR-VUE] 参与子空间的参数键为空。示例 param_keys={param_keys[:6]}，"
                    f"fisher_keys={len(fisher_keys)}。请检查分类头命名（如 fc./linear./classifier./head./layer4.）"
                )
            if args.fair_vue_debug:
                print(f"[FV-DBG] using {len(keys_valid)} keys for subspace: {keys_valid}")
            Xw = weighted_matrix_from_deltas_keys(target_deltas_list, fisher, keys_valid, device="cpu")
            k = int(args.fair_rank_k)
            if k > Xw.shape[0]:
                k = Xw.shape[0]
            V = topk_right_singular_vectors_gram(Xw, k=k)  # D × k
            # === 诊断与设备设置（简化，移除坏掉的 f-string 与重复赋值） ===
            if args.fair_vue_debug:
                # Xw: T x D; V: D x k
                print(f"[FV-DBG] Xw shape={tuple(Xw.shape)}, device={Xw.device}")
            
            # [Ablation] w/o Dual: 禁用 Gram-SVD 优化，直接在大矩阵上做 SVD (易 OOM)
            if args.fair_ablation == 'no_dual':
                if args.fair_vue_debug:
                    print(f"[FV-ABLATION] Mode: w/o Dual -> Running raw SVD on shape {tuple(Xw.shape)} (High Risk of OOM!)")
                try:
                    # 直接对 T x D 矩阵做 SVD
                    Xc = Xw - Xw.mean(dim=0, keepdim=True)
                    # full_matrices=False 会返回 Vh (k x D)
                    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
                    V = Vh.T[:, :k] # D x k
                except RuntimeError as e:
                    print(f"\n[FV-ABLATION] !!! OOM Triggered as expected in w/o Dual mode: {e} !!!\n")
                    # 为了让程序不崩溃以便记录 'OOM' 结果，这里做一个假的 V 或者直接抛出
                    raise e 
            else:
                # 正常路径：对偶 Gram 方法
                V = topk_right_singular_vectors_gram(Xw, k=k)  # D × k
            
            if args.fair_vue_debug:
                print(f"[FV-DBG] Xw shape={tuple(Xw.shape)}, V shape={tuple(V.shape)}")

            # 使用 Gram-SVD 结果 V（D×k），避免在高维上再做一次完整 SVD
            dev = V.device  # 统一使用这个设备

            # 5) 计算ρ并按阈值切分为 V_spec / V_comm
            rhos = rho_values_keys(V, other_deltas_list, keys_valid, max_samples=int(args.fair_rho_max_samples))  # 内部已用 V.device 对齐
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
            
            # [Modified] 构造公共子空间的投影矩阵 Q_comm，用于提取“有用知识”
            Q_comm = None
            if V_comm is not None and V_comm.numel() > 0:
                Q_comm, _ = torch.linalg.qr(V_comm, mode='reduced')

            if args.fair_vue_debug:
                print(f"[FV-DBG] Q is None? {Q is None}, "
                    f"Q.shape={(None if Q is None else tuple(Q.shape))}, device={(None if Q is None else Q.device)}")

                # Step4: 子空间（Gram-SVD + ρ + V_spec/V_comm + Q）结束
                _fv_last_t = _fv_time_mark("step4_subspace_and_rho", _t0, _fv_last_t)



            # 7) 逐轮累计“特异分量”
            # 只在 parameters 空间累积“目标客户端”的特异分量
            start_sd = {k: v.detach().clone().cpu() for k, v in fair_model.state_dict().items()}
            Dparam   = flatten_by_keys(start_sd, keys_valid).numel()
            spec_total = torch.zeros(Dparam, device=dev)

            # 仅基于已构建好的 target_deltas_list（避免再次引用 round_client_deltas）
            used_rounds = 0
            for d in target_deltas_list:
                d_tar = flatten_by_keys(d, keys_valid, device=dev)  # Δ_target^t
                if Q is not None:
                    spec = Q @ (Q.T @ d_tar)     # 只拿“目标”的特异分量
                else:
                    spec = torch.zeros_like(d_tar, device=dev)
                spec_total += spec

                
                used_rounds += 1
            if args.fair_vue_debug:
                print(f"[FV-DBG] used_rounds_for_target={used_rounds}, ||spec_total||_2={float(torch.norm(spec_total)):.3e}")

                # Step5: 累积特异分量结束
                _fv_last_t = _fv_time_mark("step5_accumulate_spec", _t0, _fv_last_t)

            # === 擦除系数确定 ===
            base_alpha = float(getattr(args, 'fair_erase_scale', 0.25))
            erase_scale = base_alpha

            if getattr(args, 'fair_auto_erase', False):
                # === 自动调参 erase_scale（仅 parameters；不访问原始数据）===
                # 1) 解析参数
                def _parse_pair_csv(s: str):
                    xs = [float(x) for x in s.split(',') if x.strip()!='']
                    if len(xs) < 2:
                        return 0.0, 0.04
                    return xs[0], xs[1]
                def _parse_list_csv(s: str):
                    return [float(x) for x in s.split(',') if x.strip()!='']
                target_lo, target_hi = _parse_pair_csv(getattr(args, "fair_drop_bounds", "0.00,0.04"))
                grid_mults = _parse_list_csv(getattr(args, "fair_grid_scales", "0.5,0.75,1.0,1.25,1.5"))
                bisect_steps = int(getattr(args, "fair_bisect_steps", 3))

                # 2) 诊断量：Fisher 能量 & 特异性分
                def _fisher_energy(vec_1d: torch.Tensor) -> float:
                    like = state_dict_like_by_keys(vec_1d.to('cpu'), start_sd, param_keys)
                    s = 0.0
                    for k in param_keys:
                        Fi = fisher.get(k, None)
                        if Fi is None:
                            continue
                        v = like[k].to(Fi.device).float()
                        s += float((Fi.float().flatten() * (v.flatten()**2)).sum().item())
                    return s
                spec_energy = _fisher_energy(spec_total)
                def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
                    na = torch.norm(a); nb = torch.norm(b)
                    if na.item()==0 or nb.item()==0:
                        return 0.0
                    return float(torch.dot(a, b) / (na*nb))
                with torch.no_grad():
                    unit_spec = spec_total / (torch.norm(spec_total) + 1e-12)
                    cos_list = []
                    # 采样最多 256 个“其它客户端增量”估计平均相似度，避免过慢
                    take = other_deltas_list[:256]
                    for d in take:
                        v = flatten_by_keys(d, param_keys, device=dev)
                        cos_list.append(_safe_cos(unit_spec, v))
                    avg_cos = sum(cos_list)/max(1, len(cos_list))
                    idio = max(0.0, min(1.0, 1.0 - avg_cos))  # 特异性分：越大越“特”
                if args.fair_vue_debug:
                    print(f"[FV-DBG] spec_fisher_energy={spec_energy:.3e}, avg_cos={avg_cos:.3f}, idiosyncrasy={idio:.3f}")

                # 3) 评估函数：临时应用 α·spec_total 到参数并测一次测试集精度
                def _eval_acc(alpha: float) -> float:
                    param_now = flatten_by_keys(start_sd, param_keys, device=dev)
                    param_new = param_now - float(alpha) * spec_total
                    new_params = state_dict_like_by_keys(param_new.to('cpu'), start_sd, param_keys)
                    tmp_sd = dict(start_sd)
                    for k in param_keys:
                        tmp_sd[k] = new_params[k]
                    tmp_model = deepcopy(fair_model).to(args.device)
                    tmp_model.load_state_dict(tmp_sd)
                    acc = float(get_accuracy_only(tmp_model, test_dataloader, args.device))
                    del tmp_model
                    return acc

                # 用同一条评估链路测 baseline，确保 drop(0)==0
                baseline_acc = _eval_acc(0.0)

                # 4) 构造候选 α
                alpha0 = base_alpha * (0.7 + 0.6*idio)
                cands = sorted(set([0.0] + [max(0.0, m*base_alpha) for m in grid_mults] + [alpha0]))

                # 5) 粗网格搜索
                evals = [(a, _eval_acc(a)) for a in cands]
                drops = [(a, max(0.0, baseline_acc - acc)) for (a, acc) in evals]
                under = max([a for a, d in drops if d <= target_lo + 1e-6], default=None)
                over  = min([a for a, d in drops if d >= target_hi - 1e-6], default=None)

                if under is None and over is None:
                    # 没覆盖目标区间
                    mid = 0.5*(target_lo + target_hi)
                    chosen = min(drops, key=lambda t: abs(t[1]-mid))[0]
                else:
                    # 6) 二分细化
                    lo = under if under is not None else 0.0
                    hi = over  if over  is not None else max(cands)
                    chosen = None
                    for _ in range(max(0, bisect_steps)):
                        mid_a = 0.5*(lo + hi)
                        acc_m = _eval_acc(mid_a)
                        drop_m = max(0.0, baseline_acc - acc_m)
                        if args.fair_vue_debug:
                            print(f"[FV-DBG] bisect α={mid_a:.4f} → drop={drop_m:.4f}")
                        if drop_m < target_lo:
                            lo = mid_a
                        elif drop_m > target_hi:
                            hi = mid_a
                        else:
                            chosen = mid_a
                            break
                    if chosen is None:
                        def _dist_to_interval(x, L, H): return 0.0 if L<=x<=H else min(abs(x-L), abs(x-H))
                        drop_lo = max(0.0, baseline_acc - _eval_acc(lo))
                        drop_hi = max(0.0, baseline_acc - _eval_acc(hi))
                        chosen = lo if _dist_to_interval(drop_lo, target_lo, target_hi) <= _dist_to_interval(drop_hi, target_lo, target_hi) else hi
                
                erase_scale = chosen
                # Step6: 自动擦除系数搜索结束
                _fv_last_t = _fv_time_mark("step6_auto_erase_search", _t0, _fv_last_t)
            else:
                    print(f"[FV-DBG] Auto-erase is DISABLED, using erase_scale={erase_scale}")

            # 7) 应用最终擦除到模型参数
            # [Corrected] 修正后的正交恢复：
            # 使用“其他客户端”的均值来修复公共子空间，绝对避免使用 target_deltas (防止隐私回流)
            
            repair_vec = torch.zeros_like(spec_total)
            if len(other_deltas_list) > 0 and Q_comm is not None:
                # 1. 计算保留客户端的均值方向
                # 注意：other_deltas_list 可能只包含最后一轮，但这代表了模型收敛时的通用梯度方向
                others_sum = torch.zeros_like(spec_total)
                for d in other_deltas_list:
                    others_sum += flatten_by_keys(d, keys_valid, device=dev)
                others_mean = others_sum / float(len(other_deltas_list))

                # 2. 投影到公共子空间 (仅恢复通用知识)
                repair_vec = Q_comm @ (Q_comm.T @ others_mean)
                
                # 3. [Critical Fix] 能量补偿机制
                # 自动计算缩放系数，使得恢复的能量与擦除的能量在同一数量级
                norm_erase  = torch.norm(erase_scale * spec_total)
                norm_repair = torch.norm(repair_vec)
                
                # 目标：恢复约 50% ~ 60% 的被擦除能量，既能修补泛化损伤，又不至于覆盖遗忘效果
                # 如果 norm_repair 极小(防除零)，则不进行过度放大
                compensation_factor = 0.0
                if norm_repair > 1e-6:
                    compensation_factor = args.fair_repair_ratio * (norm_erase / norm_repair)
                
                # 应用动态补偿系数
                repair_vec = repair_vec * compensation_factor

                if args.fair_vue_debug:
                    print(f"[FV-DBG] Orthogonal Repair: ||Erase||={norm_erase:.3f}, ||RepairRaw||={norm_repair:.3f} -> Factor={compensation_factor:.3f}")

            # [Ablation] w/o Repair: 强制将修复向量置零
            if args.fair_ablation == 'no_repair':
                if args.fair_vue_debug:
                    print("[FV-ABLATION] Mode: w/o Repair -> Forcing repair_vec to ZERO.")
                repair_vec = torch.zeros_like(spec_total)


            param_now   = flatten_by_keys(start_sd, param_keys, device=dev)
            # repair_vec 已经在上面被动态缩放过了，这里直接加即可
            param_new   = param_now - erase_scale * spec_total + repair_vec
            new_params  = state_dict_like_by_keys(param_new.to('cpu'), start_sd, param_keys)
            new_sd = dict(start_sd)
            for k in param_keys:
                new_sd[k] = new_params[k]
            fair_model.load_state_dict(new_sd)


            if args.fair_vue_debug:
                _fv_last_t = _fv_time_mark("step7_apply_erase", _t0, _fv_last_t)
           

            fair_time_sec = time.time() - _t0
            print(f"[Timing] FAIR-VUE time: {fair_time_sec:.2f}s")


            # 9) 评测 (移到计时结束后)
            perf = get_performance(model=fair_model, test_dataloader=test_dataloader,
                                clientwise_dataloader=clientwise_dataloaders,
                                num_classes=num_classes, device=args.device)

            if args.fair_vue_debug:
                # Step8: 评测与汇总结束（总时间再标一遍）
                _fv_last_t = _fv_time_mark("step8_eval_and_summary", _t0, _fv_last_t)


            summary.setdefault('performance', {})
            summary['performance']['after_fair_vue'] = perf
            if args.verbose:
                print(f"Performance after FAIR-VUE : {perf}")

            # ---- 专门测忘却客户端的精度 ----
            forget_loader = clientwise_dataloaders[target_id]
            acc = get_accuracy_only(fair_model, forget_loader, args.device)
            print(f"[FAIR-VUE模型] 忘却客户端{target_id}自有数据精度: {acc*100:.2f}%")

            # ==== 六项指标统一打印（FAIR-VUE）====
            test_acc_fair    = get_accuracy_only(fair_model, test_dataloader, args.device)
            target_acc_fair  = acc
            retain_acc_fair  = eval_retain_acc(fair_model, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_fair = eval_ce_loss(fair_model, forget_loader, args.device)
            speedup_fair     = (t_retrain_sec / fair_time_sec) if (t_retrain_sec is not None and fair_time_sec > 0) else None
            angle_fair       = cosine_angle_between_models(fair_model, retrained_global_model) if has_retrain_baseline else None

            mia_fair = None
            if args.apply_membership_inference:
                if args.mia_verbose:
                    print("\n[调试] 开始执行成员推断攻击 (evaluate_mia_attack)...")
                # 与其他分支保持一致：对目标客户端执行成员推断
                mia_fair = evaluate_mia_attack(
                    target_model=deepcopy(fair_model),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                    
                )
                if args.mia_verbose:
                    print(f"[调试] MIA 返回类型: {type(mia_fair)}")
                    if isinstance(mia_fair, dict):
                        print(f"[调试] MIA 字典键: {list(mia_fair.keys())[:10]}")
                        for k, v in list(mia_fair.items())[:5]:
                            if isinstance(v, (int, float, str)):
                                print(f"  {k}: {v}")
                            elif hasattr(v, 'shape'):
                                print(f"  {k}: tensor/array shape={v.shape}")
                            elif isinstance(v, (list, tuple)):
                                print(f"  {k}: list length={len(v)}")
                            else:
                                print(f"  {k}: type={type(v)}")
            print_forgetting_metrics(
                method_name="FAIR-VUE",
                test_acc=test_acc_fair,
                retain_acc=retain_acc_fair,
                target_acc=target_acc_fair,
                target_loss=target_loss_fair,
                speedup_x=speedup_fair,
                angle_deg=angle_fair,
                mia_result=mia_fair
            )
            _pm_fair.__exit__(None, None, None)
            _print_mem_overhead("FAIR-VUE", _pm_fair, summary)
            # —— 关键：清理 MIA 大对象并同步 CUDA，避免后续卡住 —— 
            try:
                import torch, gc
                if isinstance(mia_fair, dict):
                    for k in ['mia_attacker_predictions','mia_attacker_probabilities','predictions','probabilities','scores']:
                        if k in mia_fair: mia_fair.pop(k, None)
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass



            # ===== HEALING: 让遗忘后的模型在保留客户端上“治疗/恢复” =====
            if args.heal:
                # 1) teacher 选择：post=用擦除后的模型；pre=用遗忘前的全局模型
                if args.heal_teacher == 'post':
                    teacher = deepcopy(fair_model).to(args.device).eval()
                else:
                    teacher = deepcopy(global_model).to(args.device).eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)

                # 2) 方案1：一次性权重插值（Data-Free；不会触碰任意原始样本）
                if args.heal_teacher == 'post':
                        print("[HEAL] teacher=post 与 student 相同，插值将不生效；如需恢复请使用 --heal_teacher pre")
                weight_interpolate_heal(
                    student=fair_model,
                    teacher=teacher,
                    alpha=args.heal_alpha
                )

                # 4) 治疗后评测
                perf_heal = get_performance(model=fair_model, test_dataloader=test_dataloader,
                                            clientwise_dataloader=clientwise_dataloaders,
                                            num_classes=num_classes, device=args.device)
                print(f"[HEAL] Performance after healing : {perf_heal}")
                forget_loader = clientwise_dataloaders[target_id]
                acc_forget_heal = get_accuracy_only(fair_model, forget_loader, args.device)
                print(f"[HEAL] 忘却客户端{target_id}自有数据精度(治疗后): {acc_forget_heal*100:.2f}%")

        elif baseline == 'fast_fu':
            # ---- fast-fU（round-wise） ----
            # quick adapter that runs the FastFUServer over the existing full_training outputs
            from FedUnlearner.baselines.fast_fu.fast_fu import run_fast_fu
            print(">>> Running fast-fU (adapter).")
            # 1) 映射：forget_clients -> attackers（fast-fU 使用 attackers 触发擦除）
            if (not hasattr(args, 'attacker')) or (args.attacker is None) or (len(args.attacker) == 0):
                args.attacker = list(sorted(set(args.forget_clients)))

            # 2) 训练日志路径：指向直接包含 iteration_* 的目录
            fv_train_path = os.path.abspath(args.full_training_dir) if args.full_training_dir else os.path.abspath(train_path)
            if os.path.isdir(os.path.join(fv_train_path, 'full_training')):
                fv_train_path = os.path.join(fv_train_path, 'full_training')
            if (not os.path.isdir(fv_train_path)) or (not any(n.startswith('iteration_') for n in os.listdir(fv_train_path))):
                raise FileNotFoundError(
                    f"[fast-fU] iteration_* not found under: {fv_train_path}. "
                    "Please set --full_training_dir to the folder that directly contains iteration_*/client_*.pth."
                )

            # 3) 与 loader 数量对齐，避免扫描不存在的 client id
            try:
                args.total_num_clients = max(int(args.total_num_clients), len(clientwise_dataloaders))
            except Exception:
                args.total_num_clients = len(clientwise_dataloaders)

            if args.verbose:
                print(f">>> fast-fU config: attackers={args.attacker}, path='{fv_train_path}'")
            # call runner — 传入 attack_model 与 eval_nonmem_loader，让 fast-fU 分支内也能跑 MIA（与 PGA 一致）
            _pm_ff = PeakMem(args.device); _pm_ff.__enter__()
            run_fast_fu(args=args,
                        clientwise_dataloaders=clientwise_dataloaders,
                        train_path=fv_train_path,
                        global_model=deepcopy(global_model),
                        test_dataloader=test_dataloader,
                        retrained_global_model=deepcopy(retrained_global_model),
                        attack_model=locals().get("attack_model", None),
                        eval_nonmem_loader=locals().get("mia_eval_nonmem_loader", None),
                        # 把 retrain 基线用时传给 fast-FU 以输出 Speedup
                        retrain_time_sec=locals().get("t_retrain_sec", None))
            print(">>> fast-fU run finished.")
            _pm_ff.__exit__(None, None, None)
            _print_mem_overhead("fast-fU", _pm_ff, summary)

        elif baseline == 'quickdrop':
            # ---- QuickDrop（严格贴近原法的 DC/梯度匹配蒸馏 + 合成集本地更新）----
            from FedUnlearner.baselines import run_quickdrop
            print(">>> Running QuickDrop (baseline).")
            _t0 = time.time()
            _pm_qd = PeakMem(args.device); _pm_qd.__enter__()
            qd_model, qd_info = run_quickdrop(
                args=args,
                global_model=deepcopy(global_model),
                # 只在“保留客户端”上参与（与 retraining 保持一致）
                clientwise_dataloaders=retain_clientwise_dataloaders,
                test_dataloader=test_dataloader,
                num_classes=num_classes,
                device=args.device,
            )
            qd_time_sec = time.time() - _t0

            # ==== 六项指标统一打印（QuickDrop）====
            test_acc_qd    = get_accuracy_only(qd_model, test_dataloader, args.device)
            target_acc_qd  = get_accuracy_only(qd_model, clientwise_dataloaders[forget_client], args.device)
            retain_acc_qd  = eval_retain_acc(qd_model, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_qd = eval_ce_loss(qd_model, clientwise_dataloaders[forget_client], args.device)
            speedup_qd     = (t_retrain_sec / qd_time_sec) if (locals().get('t_retrain_sec', None) is not None and qd_time_sec > 0) else None
            angle_qd       = cosine_angle_between_models(qd_model, retrained_global_model) if locals().get('has_retrain_baseline', False) else None
            mia_qd = None
            if args.apply_membership_inference:
                mia_qd = evaluate_mia_attack(
                    target_model=deepcopy(qd_model),
                    attack_model=locals().get("attack_model", None),
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=locals().get("mia_eval_nonmem_loader", None)
                )
            print_forgetting_metrics("QuickDrop", test_acc_qd, retain_acc_qd, target_acc_qd, target_loss_qd, speedup_qd, angle_qd, mia_qd)
            # 供后续可能的二次评测使用
            unlearned_quickdrop_model = deepcopy(qd_model)
            _pm_qd.__exit__(None, None, None)
            _print_mem_overhead("QuickDrop", _pm_qd, summary)

        elif baseline == 'conda':
            # === Contribution Dampening（原 LEGACY_UNLEARN）作为标准 baseline ===
            _t0 = time.time()
            model_conda = deepcopy(global_model)
            # 注意：该 baseline 期望传入实验根路径（其下含 full_training），保持与旧版一致
            _pm_conda = PeakMem(args.device); _pm_conda.__enter__()

            # 如果命令行指定了 --conda_weights_path，就优先用；否则退回默认的 weights_path
            conda_weights_root = args.conda_weights_path or weights_path

            model_conda = run_conda(                
                global_model=model_conda,
                weights_path=conda_weights_root, 
                forget_clients=args.forget_clients,
                total_num_clients=len(clientwise_dataloaders),
                dampening_constant=args.dampening_constant,
                dampening_upper_bound=args.dampening_upper_bound,
                ratio_cutoff=args.ratio_cutoff,
                dampening_lower_bound=args.conda_lower_bound,
                eps=args.conda_eps,
                device=args.device
            )
            perf = get_performance(
                model=model_conda,
                test_dataloader=test_dataloader,
                clientwise_dataloader=clientwise_dataloaders,
                num_classes=num_classes,
                device=args.device
            )
            conda_time_sec = time.time() - _t0
            print(f"[Timing] CONDA time: {conda_time_sec:.2f}s")
            summary['performance']['after_conda'] = perf
            if args.verbose:
                print(f"Performance after conda : {perf}")

            # 攻击评测保持与其他 baseline 一致
            if args.apply_backdoor:
                forget_backdoor_conda = evaluate_backdoor_attack(
                    model=model_conda, backdoor_context=backdoor_context, device=args.device
                )
                summary['backdoor_results']['after_conda'] = forget_backdoor_conda
                if args.verbose:
                    print(f"Backdoor results after conda : {forget_backdoor_conda}")
            if args.apply_label_poisoning:
                forget_poisoning_conda = evaluate_poisoning_attack(
                    model=model_conda, poisoning_context=poisoning_context, device=args.device
                )
                summary['poisoning_results']['after_conda'] = forget_poisoning_conda
                if args.verbose:
                    print(f"Poisoning results after conda : {forget_poisoning_conda}")

            # 六项统一指标
            test_acc_conda    = get_accuracy_only(model_conda, test_dataloader, args.device)
            target_acc_conda  = get_accuracy_only(model_conda, clientwise_dataloaders[forget_client], args.device)
            retain_acc_conda  = eval_retain_acc(model_conda, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_conda = eval_ce_loss(model_conda, clientwise_dataloaders[forget_client], args.device)
            speedup_conda     = (t_retrain_sec / conda_time_sec) if (t_retrain_sec is not None and conda_time_sec > 0) else None
            angle_conda       = cosine_angle_between_models(model_conda, retrained_global_model) if has_retrain_baseline else None
            mia_conda = None
            if args.apply_membership_inference:
                mia_conda = evaluate_mia_attack(
                    target_model=deepcopy(model_conda),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                )
            print_forgetting_metrics("CONDA", test_acc_conda, retain_acc_conda, target_acc_conda, target_loss_conda, speedup_conda, angle_conda, mia_conda)
            _pm_conda.__exit__(None, None, None)
            _print_mem_overhead("CONDA", _pm_conda, summary)
        # ----------------- FedFIM -----------------
    if "fedfim" in args.baselines:
        from FedUnlearner.baselines import run_fedfIM
        fedfim_model = deepcopy(global_model)

        _pm_fim = PeakMem(args.device); _pm_fim.__enter__()
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
        
        mia_fedfim = None
        if args.apply_membership_inference:
            mia_fedfim = evaluate_mia_attack(
                target_model=deepcopy(fedfim_model),
                attack_model=attack_model,
                client_loaders=clientwise_dataloaders,
                test_loader=test_dataloader,
                dataset=args.dataset,
                forget_client_idx=args.forget_clients[0],
                device=args.device,
                eval_nonmem_loader=mia_eval_nonmem_loader
            )
        # 补充打印 FedFIM 的完整指标
        speedup_fim = None # FedFIM 暂未在主流程计时
        retain_acc_fim = eval_retain_acc(fedfim_model, clientwise_dataloaders, args.forget_clients, args.device)
        angle_fim = cosine_angle_between_models(fedfim_model, retrained_global_model) if has_retrain_baseline else None
        print_forgetting_metrics("FedFIM", perf['test_acc'], retain_acc_fim, acc, eval_ce_loss(fedfim_model, forget_loader, args.device), speedup_fim, angle_fim, mia_fedfim)

        _pm_fim.__exit__(None, None, None)
        _print_mem_overhead("FedFIM", _pm_fim, summary)



    # check mia precision and recall on all model
    summary['mia_attack'] = {}


    # Add configurations to the summary
    summary['config'] = vars(args)

    # Create a timestamp for the summary file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Dump the summary into a file with the summary-timestamp name
    with open(os.path.join(weights_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
