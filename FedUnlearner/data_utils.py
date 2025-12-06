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
        self.dataset = datasets.ImageFolder(root) # 这里不加 transform，保持原始 PIL
        self.transform = transform
        self.images = []
        self.targets = []
        
        print(f"正在预加载 {root} 到内存，请稍候...")
        for idx in range(len(self.dataset)):
            img, label = self.dataset[idx]
            # 这里为了省内存，可以先 resize 到 64x64 再存，或者直接存 PIL
            # TinyImageNet 本身就是 64x64，直接存 PIL 对象开销很小
            self.images.append(img) 
            self.targets.append(label)
        print(f"预加载完成！共 {len(self.images)} 张图片。")
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

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
        apply_transform = transforms.Compose(
            [transforms.Resize(size=(32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
        return train_dataset, test_dataset, 10

    elif dataset_name == 'cifar100':
        data_dir = './data/cifar100/'
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        # 训练增强：标准 CIFAR 套餐
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25)
        ])
        # 测试/评估：只做归一化
        test_transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
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
        train_dataset = InMemoryImageFolder(root=train_path, transform=train_transform)
        test_dataset = InMemoryImageFolder(root=test_path, transform=test_transform)
        
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
