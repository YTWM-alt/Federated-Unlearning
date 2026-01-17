from torchvision import datasets, transforms
from torch.utils.data import Subset
import os
import numpy as np
import torch
from typing import Tuple, Dict
from typeguard import typechecked

# 在 data_utils.py 顶部添加
from torch.utils.data import Dataset

class InMemoryImageFolder(Dataset):
    """
    一次性把 ImageFolder 的数据全读进内存，训练时不再读盘。
    """
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.transform = transform

            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
       return self.dataset[idx]

@typechecked
def get_dataset(dataset_name: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    """
    Get the dataset.
    Args:
        dataset_name: string, name of the dataset
    Returns:
        train_dataset: torch.utils.data.Dataset, training dataset
        test_dataset: torch.utils.data.Dataset, testing dataset
        num_classes: int, number of classes in the dataset
    """
    if dataset_name == 'cifar10':
        data_dir = './data/cifar/'
        # [修复] 增加 CIFAR-10 标准数据增强，防止过拟合
        apply_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        # 测试集不需要增强，只需要归一化
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
             
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=test_transform)
        return train_dataset, test_dataset, 10

    elif dataset_name == 'cifar100':
        data_dir = './data/cifar100/'
        # [修改] 迁移学习建议使用 ImageNet 的均值和方差，而不是 CIFAR 自己的
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        
        # 训练增强：标准 CIFAR 套餐
        train_transform = transforms.Compose([
            # [修改] 配合预训练模型：放大到 224x224 (ResNet 标准输入)
            # [修改] 224太慢了，改成 112 或者 96，既能利用预训练，速度也能快4-5倍
            transforms.Resize((96, 96)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            #transforms.RandomErasing(p=0.25)
        ])
        # 测试/评估：只做归一化
        test_transform = transforms.Compose([
            # [修改] 测试集也要放大
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=test_transform
        )
        return train_dataset, test_dataset, 100

    elif dataset_name == 'tinyimagenet':
        data_dir = './data/tiny-imagenet-200/'
        # Tiny-ImageNet 统计均值和方差
        mean = (0.4802, 0.4481, 0.3975)
        std  = (0.2302, 0.2265, 0.2262)
        
        # 64x64 图像的增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # [新增] 增加颜色抖动，进一步防止过拟合
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_path = os.path.join(data_dir, 'train')
        test_path = os.path.join(data_dir, 'val') # 通常用 val 作为测试集
        
        # 注意：需确保 val 目录下已按类别分子文件夹，否则 ImageFolder 无法识别分类        train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
        # [修改] 不再使用 InMemoryImageFolder 预加载，直接使用 ImageFolder 以实现秒级启动
        # 配合 main.py 中的 persistent_workers=True，训练速度不会受影响
        train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
        
        return train_dataset, test_dataset, 200


    elif dataset_name == 'mnist':
        data_dir = './data/mnist/'
        apply_transform = transforms.Compose(
            [transforms.Resize(size=(28, 28)),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        return train_dataset, test_dataset, 10


# Added alpha parameter
@typechecked
def create_dirichlet_data_distribution(dataset: torch.utils.data.Dataset, num_clients: int, 
                                       num_classes: int, alpha: float = 0.5) -> Dict[int, torch.utils.data.Dataset]:
    """
    Create a dirichlet data distribution.
    Args:
        dataset: torch.utils.data.Dataset, dataset
        num_clients: int, number of clients
        num_classes: int, number of classes
        alpha: float, concentration of data distribution (<1 for non-iid)
    Returns:
        client_groups: dict, client groups
    """
    min_size = 0  # minimum batch size threshold
    K = num_classes
    labels = np.array(dataset.targets)
    N = len(labels)

    # Dictionary to store client-wise data indexes (initially empty)
    dict_users = {}

    while min_size < 10:
        # Create empty lists to store data indexes for each client in this batch
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            # Get indexes of data points belonging to class k
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            # Generate Dirichlet proportions for each client (controls data distribution)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

            # Ensure each client gets at least one class/sample if their current batch is less than average size
            proportions = np.array([p*(len(idx_j) < N/num_clients)
                                   for p, idx_j in zip(proportions, idx_batch)])

            # Normalize proportions to sum to 1 (represents probability distribution)
            proportions = proportions/proportions.sum()

            # Calculate number of samples from class k assigned to each client based on proportions
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]

            # Split indexes for class k based on calculated cumulative proportions and add them to client batches
            idx_batch = [idx_j + idx.tolist() for idx_j,
                         idx in zip(idx_batch, np.split(idx_k, proportions))]

            # Update minimum size based on the smallest batch in this iteration
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # Final shuffle within each client's batch to avoid ordering bias

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        # 只取前 MAX_PER_CLIENT 个索引，保证每个客户端 ~600 张
        dict_users[j] = Subset(dataset, idx_batch[j])
    return dict_users


@typechecked
def create_iid_data_distribution(dataset: torch.utils.data.Dataset, num_clients: int, 
                                 num_classes: int) -> Dict[int, torch.utils.data.Dataset]:
    """
    Create an iid data distribution.
    Args:
        dataset: torch.utils.data.Dataset, dataset
        num_clients: int, number of clients
        num_classes: int, number of classes
    Returns:
        client_groups: dict, client groups
    """
    n_train = len(dataset)
    idxs = np.random.permutation(n_train)  # Randomly shuffle the data indexes

    # Split the shuffled indexes into batches for each client device
    batch_idxs = np.array_split(idxs, num_clients)

    # Create a dictionary to map client IDs to their assigned batch indexes

    dict_users = {i: Subset(dataset, batch_idxs[i])
                for i in range(num_clients)}
    return dict_users



@typechecked
def create_class_exclusive_distribution(dataset: torch.utils.data.Dataset, num_clients: int, 
                                        num_classes: int, exclusive_client: int = 0,
                                        exclusive_classes: list = [0, 1],
                                        shared_ratio: float = 0.3) -> Dict[int, torch.utils.data.Dataset]:
    """
    创建一个"双类私有+稀释共享"的分布 (User Proposed Strategy)：
    1. Client 0 拥有 exclusive_classes (例如 [0, 1]) 的 100% 数据 (绝对权威)。
    2. 选取这两类数据的 shared_ratio (20%) 作为公共池。
    3. 将公共池的数据**平均切分** (Split) 给剩余客户端 (而不是复制)。
    目的：Client 0 占绝对主导，其他客户端仅有少量样本维持基本认知，移除 Client 0 后模型性能应显著下降。
    """
    # 获取所有标签
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        # 兼容 ImageFolder 等没有直接 targets 属性的情况
        labels = np.array([y for _, y in dataset])

    dict_users = {}

    # 1. 准备数据容器
    target_indices_all = [] # 存放给 Client 0 的所有目标类数据 (0和1的全量)
    shared_pool_all = []    # 存放给其他客户端的"公共知识" (0和1的20%)
    
    # 2. 遍历每一个目标类别 (0 和 1)
    for cls in exclusive_classes:
        idx_c = np.where(labels == cls)[0]
        np.random.shuffle(idx_c)
        
        # Client 0 拿走全量
        target_indices_all.extend(idx_c.tolist())
        
        # 截取一部分放入公共池
        split_point = int(len(idx_c) * shared_ratio)
        shared_pool_all.extend(idx_c[:split_point].tolist())

    # 3. 找出非目标类别 (2-9) 的索引
    # 使用 np.isin 创建掩码
    mask_others = ~np.isin(labels, exclusive_classes)
    idx_others = np.where(mask_others)[0]

    # 4. 配置 Client 0：拥有 Class 0+1 的全部数据
    dict_users[exclusive_client] = Subset(dataset, target_indices_all)

    # 5. 配置其他客户端：(其他类别 + 公共池的拷贝)
    # 先将其他类别的数据平均分给剩余客户端
    np.random.shuffle(idx_others)

    
    other_clients = [i for i in range(num_clients) if i != exclusive_client]
    # [修改] 将公共池切分为 N 份 (Split)，而不是复制
    # 如果 shared_pool_all 有 2000 张，19个客户端，每人分到约 105 张
    np.random.shuffle(shared_pool_all) # 先打乱公共池
    shared_chunks = np.array_split(shared_pool_all, len(other_clients))

    if len(other_clients) > 0:
        batch_idxs = np.array_split(idx_others, len(other_clients))
        for i, cid in enumerate(other_clients):
            # batch_idxs[i] (其他类) + shared_chunks[i] (极少量的目标类)
            combined_idxs = np.concatenate((batch_idxs[i], shared_chunks[i]))
            np.random.shuffle(combined_idxs) # 混合打乱，避免训练时 batch 分布不均
            dict_users[cid] = Subset(dataset, combined_idxs)

    return dict_users