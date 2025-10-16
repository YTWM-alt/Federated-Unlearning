import torch
from collections import OrderedDict
from fedfim_dampening import apply_fedfIM_dampening

# 假设有一个简单模型参数
state = OrderedDict({
    "w": torch.tensor([1.0, 2.0, 3.0]),
    "b": torch.tensor([0.5])
})

F_r = OrderedDict({
    "w": torch.tensor([10.0, 0.1, 1.0]),
    "b": torch.tensor([5.0])
})
F_f = OrderedDict({
    "w": torch.tensor([1.0, 10.0, 1.0]),
    "b": torch.tensor([2.0])
})

new_state = apply_fedfIM_dampening(state, F_r, F_f, dampening_constant=0.5, ratio_cutoff=1.0)

print("原始参数:", state)
print("更新后参数:", new_state)
