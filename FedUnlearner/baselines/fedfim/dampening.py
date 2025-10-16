from collections import OrderedDict
import torch


def apply_fedfIM_dampening(
    state_dict: "OrderedDict[str, torch.Tensor]",
    F_r: "OrderedDict[str, torch.Tensor]",
    F_f: "OrderedDict[str, torch.Tensor]",
    dampening_constant: float = 0.2,
    ratio_cutoff: float = 1.0,
    upper_bound: float = 1.0,
    eps: float = 1e-8,
) -> "OrderedDict[str, torch.Tensor]":
    """
    用 F_r / F_f 对参数进行乘性衰减，返回更新后的 state_dict。

    参数:
      state_dict:      当前全局模型参数
      F_r:             保留 FIM
      F_f:             遗忘 FIM
      dampening_constant: 缩放强度 γ (0~1)
      ratio_cutoff:    保留/遗忘 比值阈值 (retain/forget < cutoff 才衰减)
      upper_bound:     更新上界，避免参数被放大
      eps:             数值稳定常数
    """
    new_state = OrderedDict()

    for k in state_dict.keys():
        w = state_dict[k]
        if k not in F_r or k not in F_f:
            new_state[k] = w.clone()
            continue

        retain = F_r[k]
        forget = F_f[k]

        ratio = (retain + eps) / (forget + eps)

        # 找到需要衰减的位置
        mask = ratio < ratio_cutoff

        # 计算缩放因子，限制在 (0, upper_bound]
        scale = torch.ones_like(w)
        scale[mask] = (ratio[mask] / ratio_cutoff).clamp(max=upper_bound)
        scale = 1.0 - dampening_constant * (1.0 - scale)

        new_state[k] = w * scale

    return new_state
