import torch
from collections import OrderedDict
from fedfim_server import aggregate_fim, partition_and_aggregate

# 构造假数据
fim1 = OrderedDict({"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])})
fim2 = OrderedDict({"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([1.5])})

all_fims = {
    0: (fim1, 10),  # client 0, 样本数 10
    1: (fim2, 30),  # client 1, 样本数 30
}

# 聚合所有客户端
agg = aggregate_fim([fim1, fim2], [10, 30])
print("加权聚合:", agg)

# 划分遗忘 client 0
F_r, F_f = partition_and_aggregate(all_fims, forget_clients=[0])
print("F_r:", F_r)
print("F_f:", F_f)
