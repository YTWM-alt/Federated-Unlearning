import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ====== 1. 定义 SmallCNN ======
class SmallCNN(nn.Module):
    """一个很小的卷积网络，适合 MNIST (1×28×28)"""
    def __init__(self, num_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,64,1,1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平为 (B,64)
        return self.classifier(x)


# ====== 2. 导入我们实现的 client_compute_fim_diag ======
from fedfim_client import client_compute_fim_diag   # 你写的函数所在文件


# ====== 3. 准备数据集 ======
transform = T.Compose([T.ToTensor()])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ====== 4. 初始化模型 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(num_channels=1, num_classes=10)

# 如果你有全局权重文件，可以加载： 
state = torch.load("/home/cloudwaves/CloudwavesWork/scientific_research/ConDa-Federated-Unlearning/experiments/mnist_small/full_training/final_model.pth", map_location="cpu")
model.load_state_dict(state)

# ====== 5. 调用 FIM 计算 ======
fim_dict, n_samples = client_compute_fim_diag(
    model, loader, device,
    num_classes=10,
    mode="prob",      # 用概率期望的方式
    max_passes=1,
    max_batches=2     # 只取两个 batch，快速测试
)

# ====== 6. 打印检查结果 ======
print("总样本数:", n_samples)
for name, val in list(fim_dict.items())[:5]:  # 只看前5个参数
    print(f"{name}: shape={val.shape}, mean={val.mean().item():.6f}, min={val.min().item():.6f}")
