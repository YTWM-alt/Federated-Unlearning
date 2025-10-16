import torch, os
from torch.utils.data import DataLoader
from FedUnlearner.data_utils import get_dataset, create_dirichlet_data_distribution
from FedUnlearner.models import SmallCNN
from FedUnlearner.fed_learn import fed_train, get_performance
from FedUnlearner.baselines.fair_vue import fair_vue_unlearn

def main():
    device = "cpu"  # 有GPU改成 cuda
    num_clients = 5
    forget_clients = [0]

    # === Step 1: 加载数据 ===
    trainset, testset, num_classes = get_dataset("mnist")

    # 生成 Dirichlet 分布的 client 数据子集
    client_subsets = create_dirichlet_data_distribution(
        trainset, num_clients=num_clients, num_classes=num_classes, alpha=0.1
    )

    # 手动包装为 DataLoader
    clientwise_loaders = {
        cid: DataLoader(subset, batch_size=64, shuffle=True)
        for cid, subset in client_subsets.items()
    }
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

    # === Step 2: 初始化模型 ===
    model = SmallCNN(num_classes=num_classes)
    weights_path = "./weights_demo"
    os.makedirs(weights_path, exist_ok=True)

    # === Step 3: 训练一轮（生成client_{id}.pth）===
    fed_train(
        num_training_iterations=1,
        test_dataloader=test_loader,
        clientwise_dataloaders=clientwise_loaders,
        weights_path=weights_path,
        global_model=model,
        num_local_epochs=1,
        lr=0.01,
        optimizer_name="sgd",
        device=device
    )

    # === Step 4: FAIR-VUE 遗忘 ===
    new_sd = fair_vue_unlearn(
        global_model=model,
        weights_path=weights_path,
        clientwise_dataloaders=clientwise_loaders,
        forget_clients=forget_clients,
        rank_k=8,
        tau_mode="median",
        fisher_batches=5,
        device=device
    )
    model.load_state_dict(new_sd)

    # === Step 5: 评估 ===
    result = get_performance(
        model,
        test_dataloader=test_loader,
        clientwise_dataloader=clientwise_loaders,
        num_classes=num_classes,
        device=device
    )
    print("\n[FAIR-VUE结果]")
    print(f"全局测试准确率: {result['test_acc']:.4f}")
    print(f"客户端准确率: {result['clientwise_acc']}")

if __name__ == "__main__":
    main()
