import torch
from typing import Iterable, Optional

@torch.no_grad()
def weight_interpolate_heal(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    alpha: float = 0.05,
    only_names: Optional[Iterable[str]] = None,
) -> torch.nn.Module:
    """
    方案1：参数平滑/权重插值（Data-Free）
    student <- (1 - alpha) * student + alpha * teacher

    Args:
        student: 遗忘后的模型（将被原地更新）
        teacher: 教师模型（一般用遗忘前全局模型；如用 post=student 本身，插值将不生效）
        alpha:   插值系数 ∈ [0, 1]，建议 0.02~0.10
        only_names: 仅插值这些参数名（可选）。None 表示对全部可训练浮点参数插值。

    Returns:
        student（已原地更新）
    """
    assert 0.0 <= alpha <= 1.0, f"alpha={alpha} 不在 [0,1]"

    # 对齐设备，避免跨设备 copy
    device = next(student.parameters()).device
    teacher = teacher.to(device).eval()

    s_sd = student.state_dict()
    t_sd = teacher.state_dict()

    # 只处理浮点张量、且 shape 完全一致的 key
    def _is_float_tensor(x: torch.Tensor) -> bool:
        return x.is_floating_point()

    # 选择范围
    allowed = set(only_names) if only_names is not None else None

    updated, skipped = 0, 0
    for k, s_val in s_sd.items():
        if allowed is not None and k not in allowed:
            skipped += 1
            continue
        t_val = t_sd.get(k, None)
        if t_val is None or not isinstance(s_val, torch.Tensor) or not isinstance(t_val, torch.Tensor):
            skipped += 1
            continue
        if not _is_float_tensor(s_val) or not _is_float_tensor(t_val):
            skipped += 1
            continue
        if s_val.shape != t_val.shape:
            skipped += 1
            continue
        # in-place: s = (1 - a) * s + a * t
        s_val.mul_(1.0 - alpha).add_(t_val.to(device), alpha=alpha)
        updated += 1

    print(f"[HEAL/Interpolate] alpha={alpha:.4f} | updated={updated} | skipped={skipped}")
    return student
