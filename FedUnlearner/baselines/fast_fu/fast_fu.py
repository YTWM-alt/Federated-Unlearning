
import os
import time
import torch
from copy import deepcopy
from collections import deque
from typing import Dict, List
from FedUnlearner.utils import eval_ce_loss, print_forgetting_metrics, cosine_angle_between_models
from FedUnlearner.attacks.mia import evaluate_mia_attack

# minimal fmodule helpers (wraps operations used by fast-fU)
def _model_norm(state_dict):
    v = torch.cat([p.detach().flatten().cpu() for p in state_dict.values() if torch.is_floating_point(p)])
    return float(torch.norm(v).item()) if v.numel() > 0 else 0.0

def _create_new_model_like(ref_sd, *, device="cpu", dtype=torch.float32):
    # 始终在 CPU/float32 上构造遗忘项，避免与 CUDA 权重混用时报 device mismatch
    return {
        k: (torch.zeros_like(v, device=device, dtype=dtype)
            if torch.is_floating_point(v) else
            torch.zeros_like(v, device=device))   # 非浮点 buffer 也放到 CPU
        for k, v in ref_sd.items()
    }

class FastFUServer:
    """
    Minimal wrapper implementing fast-fU core pieces needed to run as a baseline.
    设计目标：读取 train_path（full_training）下的 client_{cid}.pth 每轮权重，
    计算增量、按 fast-fU 策略保存部分增量，最后在尾几轮触发 compute_unlearn_term。
    """
    def __init__(self, args, clientwise_dataloaders: Dict[int, object], train_path: str, global_model):
        self.args = args
        self.train_path = train_path
        self.global_model = deepcopy(global_model)
        self.num_clients = args.total_num_clients

        # 保存 dataloaders 与遗忘客户端 id，后面算指标要用
        self.clientwise_dataloaders = clientwise_dataloaders
        self.forget_client = (args.forget_clients[0] if getattr(args, "forget_clients", None) else None)        
        # client_vols: 用每个客户端样本数近似权重
        self.client_vols = {cid: len(dl.dataset) for cid, dl in clientwise_dataloaders.items()}
        self.data_vol = float(sum(self.client_vols.values()))

        # options that mimic original fast-fU naming
        self.option = {
            'expected_saving': getattr(args, 'fast_expected_saving', 5),
            'num_rounds': args.num_training_iterations,
            'theta_delta': getattr(args, 'fast_theta', 1.0),
            'alpha': getattr(args, 'fast_alpha', 0.5),
            'attacker': set(getattr(args, 'attacker', [])),
            'clean_model': 0
        }

        self.beta: List[float] = []
        # 仅保存“被攻击轮”的攻击者增量，key=round_id -> {str(cid): state_dict(delta)}
        self.attack_grads: Dict[int, Dict[str, dict]] = {}
        self.round_selected = [[] for _ in range(self.num_clients)]
        self.unlearn_term = None
        self.unlearn_time = 0


    # 轻量精度计算（避免依赖 main.py 的 get_accuracy_only）
    def _accuracy_only(self, model, dataloader, device):
        model.eval()
        correct = total = 0
        import torch
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device); y = y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return (correct / total) if total > 0 else 0.0


    def _cast_state_dict(self, sd: dict, dtype=torch.float16):
        # 选用 float16 存，以省内存；需要时再临时转回 float32
        return {k: (v.to('cpu', dtype=dtype) if torch.is_floating_point(v) else v) for k, v in sd.items()}

    # -------------------------
    # I/O helpers: 逐轮读取 client_* 文件（借鉴 main.py 中的流式函数）
    # -------------------------
    def _list_iteration_dirs(self):
        items = []
        if not os.path.isdir(self.train_path):
            return items
        for name in os.listdir(self.train_path):
            if name.startswith("iteration_"):
                try:
                    r = int(name.split("_")[-1])
                    items.append((r, os.path.join(self.train_path, name)))
                except:
                    pass
        items.sort(key=lambda x: x[0])
        return items

    def _load_client_state(self, path, cid):
        p = os.path.join(path, f"client_{cid}.pth")
        if not os.path.isfile(p): return None
        sd = torch.load(p, map_location='cpu', weights_only=True)
        if isinstance(sd, dict) and 'state_dict' in sd:
            return sd['state_dict']
        return sd

    # -------------------------
    # fast-fU core: sampling / process_grad / update_beta / compute_unlearn_term
    # -------------------------
    def efficientlyClientSampling(self, updates: List[dict], j_max=10):
        """Pick top-m updates as binary mask using norms & sample weights (adapted)."""
        m = int(self.option['expected_saving'])
        norms = [_model_norm(u) for u in updates]
        weights = [1.0 * self.client_vols[cid] / self.data_vol for cid in range(len(updates))]
        u_k = [w * n for w, n in zip(weights, norms)]
        sum_uk = sum(u_k) if sum(u_k) != 0 else 1.0
        p_k = [min((m * uk) / sum_uk, 1.0) for uk in u_k]
        for _ in range(j_max):
            I_k = sum(1 for pk in p_k if pk < 1)
            P_k = sum(pk for pk in p_k if pk < 1)
            C_k = (m - len(p_k) + I_k) / (P_k + 1e-12)
            for i in range(len(p_k)):
                if p_k[i] < 1:
                    p_k[i] = min(C_k * p_k[i], 1.0)
            if C_k <= 1: break
        # choose top-m by p_k
        idx_sorted = sorted(range(len(p_k)), key=lambda i: p_k[i], reverse=True)
        top_idx = set(idx_sorted[:m])
        out = []
        for i, upd in enumerate(updates):
            if i in top_idx:
                out.append(upd)
            else:
                # zero-out update dict (same keys, zeros)
                out.append({k: torch.zeros_like(v) for k, v in upd.items()})
        return out

    def _process_grad_attackers(self, round_id: int, iter_dir: str, attacker_ids: List[int]):
        """仅为攻击者计算 Δ，并存入 self.attack_grads[round_id]（float16 压缩存放）。"""
        if not attacker_ids:
            return
        base = self.global_model.state_dict() if hasattr(self.global_model, 'state_dict') else self.global_model
        grads_this_round: Dict[str, dict] = {}
        for cid in attacker_ids:
            p = os.path.join(iter_dir, f"client_{cid}.pth")
            if not os.path.isfile(p): 
                continue
            sd = torch.load(p, map_location='cpu', weights_only=True)
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            delta = {k: (base[k].to('cpu', dtype=torch.float32) - sd[k].to('cpu', dtype=torch.float32))
                     for k in base.keys() if k in sd}
            grads_this_round[str(cid)] = self._cast_state_dict(delta, dtype=torch.float16)
            del sd, delta
        if grads_this_round:
            self.attack_grads[round_id] = grads_this_round

    def update_beta(self, selected_client_ids: List[int]):
        sum_vol = sum(1.0 * self.client_vols[cid] / self.data_vol for cid in selected_client_ids)
        self.beta.append(sum_vol)

    def getAttacker_rounds(self, attackers: List[int]):
        round_attack = set([])
        for cid in attackers:
            round_attack.update(self.round_selected[cid])
        round_attack = sorted(list(round_attack))
        attackers_round = [[] for _ in range(len(round_attack))]
        for idx, r in enumerate(round_attack):
            for cid in attackers:
                if r in self.round_selected[cid]:
                    attackers_round[idx].append(cid)
        return round_attack, attackers_round

    def compute_unlearn_term(self, round_attack, attackers_round, current_round):
        """Core copied-and-adapted from fast-fU compute_unlearn_term (alpha / beta / theta flow)."""
        # 用 CPU/float32 的“零模型”作为累加容器（不随全局模型 device 变化）
        ref_sd = self.global_model.state_dict() if hasattr(self.global_model, 'state_dict') else self.global_model
        unlearning_term = _create_new_model_like(ref_sd, device="cpu", dtype=torch.float32)
        alpha = - self.option['alpha']
        list_beta = []
        for idx in range(len(self.beta)):
            beta = self.beta[idx]
            if idx in round_attack:
                for cid in attackers_round[round_attack.index(idx)]:
                    beta -= 1.0 * self.client_vols[cid] / self.data_vol
            beta = beta * alpha + 1
            list_beta.append(beta)
        # accumulate
        for idx in range(len(round_attack)):
            round_id = round_attack[idx]
            # multiply previous factor
            for k in unlearning_term.keys():
                unlearning_term[k] = unlearning_term[k] * list_beta[round_id]
            # 仅从 attack_grads 里取（存的是 float16，按需转回 float32；全程 CPU，避免显存暴涨）
            for c_id in attackers_round[idx]:
                grads_r = self.attack_grads.get(round_id, {})
                g = grads_r.get(str(c_id))
                if g is not None:
                    for k in unlearning_term.keys():
                        if k in g:
                            # g[k] 也保持在 CPU/float32 上累加
                            unlearning_term[k] = unlearning_term[k] + \
                                (self.client_vols[c_id] / self.data_vol) * g[k].to(device="cpu", dtype=torch.float32)
            if idx == len(round_attack) - 1: continue
            for r_id in range(round_id + 1, round_attack[idx + 1]):
                for k in unlearning_term.keys():
                    unlearning_term[k] = unlearning_term[k] * list_beta[r_id]
        # scale
        theta = float(self.option['theta_delta'])
        for k in unlearning_term.keys():
            unlearning_term[k] = unlearning_term[k] * theta
        return unlearning_term

    # -------------------------
    # Runner: 读取所有轮次，按轮模拟 fast-fU 的 process（仅需要最近几轮触发）
    # -------------------------
    def run(self, test_dataloader, retrained_global_model=None,
            attack_model=None, eval_nonmem_loader=None,
            retrain_time_sec: float | None = None):
        # === timing: start ===
        _t0_total = time.time()
        # stream iterations
        rounds = self._list_iteration_dirs()
        # mapping iteration idx -> dir
        idx_map = {r: d for r, d in rounds}
        sorted_rounds = sorted(idx_map.keys())
        max_round = sorted_rounds[-1] if sorted_rounds else -1
        for t in sorted_rounds:
            iter_dir = idx_map[t]
            # 一遍扫描：确定当轮有哪些客户端参与（存在 client_{cid}.pth 即视为参与）
            sel_clients = [cid for cid in range(self.num_clients)
                           if os.path.isfile(os.path.join(iter_dir, f"client_{cid}.pth"))]
            for cid in sel_clients:
                self.round_selected[cid].append(t)
            if not sel_clients:
                continue
            # 只更新 beta（标量），不加载全部模型
            self.update_beta(sel_clients)
            # 找出当轮攻击者；仅为攻击者计算/保存 Δ（float16）
            attackers = [c for c in sel_clients if c in self.option['attacker']]
            self._process_grad_attackers(t, iter_dir, attackers)

            # 仅在最后一轮构造并评估（避免打印多次）
            if attackers and t == max_round:
                round_attack, attackers_round = self.getAttacker_rounds(attackers)
                self.unlearn_term = self.compute_unlearn_term(round_attack, attackers_round, t)

                # —— on-demand：仅此时读取本轮所有客户端模型做 temp 平均（短时内存占用）
                # 计算 temp = mean(models_this_round)
                first_sd = torch.load(os.path.join(iter_dir, f"client_{sel_clients[0]}.pth"),
                                      map_location='cpu', weights_only=True)
                if isinstance(first_sd, dict) and 'state_dict' in first_sd:
                    first_sd = first_sd['state_dict']
                temp = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in first_sd.items()}
                # 把第一个也纳入平均
                for k in temp.keys():
                    temp[k] += first_sd[k].to('cpu', dtype=torch.float32)
                for cid in sel_clients[1:]:
                    sd = torch.load(os.path.join(iter_dir, f"client_{cid}.pth"),
                                    map_location='cpu', weights_only=True)
                    if isinstance(sd, dict) and 'state_dict' in sd:
                        sd = sd['state_dict']
                    for k in temp.keys():
                        temp[k] += sd[k].to('cpu', dtype=torch.float32)
                    del sd
                for k in temp.keys():
                    temp[k].div_(float(len(sel_clients)))

                # clean_sd = temp + unlearn_term（都在 CPU）
                clean_sd = {k: temp[k] + self.unlearn_term[k].to(dtype=torch.float32) for k in temp.keys()}

                # === 评测：测试集/遗忘客户端/交叉熵/角度 ===
                clean_model = deepcopy(self.global_model)   # 设备与全局一致
                clean_model.load_state_dict(clean_sd)       # PyTorch 会拷到同一 device
                dev = self.args.device
                # 优先使用入参的 test_dataloader；若未传再兜底 args.test_dataloader
                test_loader = test_dataloader if test_dataloader is not None else (
                    self.args.test_dataloader if hasattr(self.args, "test_dataloader") else None
                )
                forget_loader = self.clientwise_dataloaders.get(self.forget_client) if self.forget_client is not None else None
                # 若有 test_loader，就评测；否则打 NA
                if test_loader is not None:
                    test_acc = self._accuracy_only(clean_model, test_loader, dev)
                else:
                    test_acc = None
                if forget_loader is not None:
                    target_acc  = self._accuracy_only(clean_model, forget_loader, dev)
                    target_loss = eval_ce_loss(clean_model, forget_loader, dev)
                else:
                    target_acc = target_loss = None
                angle_deg = cosine_angle_between_models(clean_model, retrained_global_model) if retrained_global_model is not None else None
                # ---- MIA（与 PGA 一致）：需要 attack_model，且开启 --apply_membership_inference ----
                mia_res = None
                if getattr(self.args, "apply_membership_inference", False) and (attack_model is not None) and (test_loader is not None):
                    mia_res = evaluate_mia_attack(
                        target_model=deepcopy(clean_model),
                        attack_model=attack_model,
                        client_loaders=self.clientwise_dataloaders,
                        test_loader=test_loader,
                        dataset=self.args.dataset,
                        forget_client_idx=self.forget_client,
                        device=self.args.device,
                        eval_nonmem_loader=eval_nonmem_loader
                    )
                # === timing: end & speedup ===
                self.elapsed_total_sec = time.time() - _t0_total
                if getattr(self.args, "verbose", True):
                    print(f"[Timing] fast-fU time: {self.elapsed_total_sec:.2f}s")
                speedup_x = (float(retrain_time_sec) / self.elapsed_total_sec) if (retrain_time_sec is not None and self.elapsed_total_sec > 0) else None
                print_forgetting_metrics("fast-fU", test_acc, target_acc, target_loss, speedup_x, angle_deg, mia_res)


def run_fast_fu(args, clientwise_dataloaders, train_path, global_model,
                test_dataloader=None, retrained_global_model=None,
                attack_model=None, eval_nonmem_loader=None,
                retrain_time_sec: float | None = None):
    server = FastFUServer(args, clientwise_dataloaders, train_path, global_model)
    server.run(test_dataloader=test_dataloader,
               retrained_global_model=retrained_global_model,
               attack_model=attack_model,
               eval_nonmem_loader=eval_nonmem_loader,
               retrain_time_sec=retrain_time_sec)
    return server
