

````markdown
# 🧩 联邦遗忘研究框架（FedUnlearner）

本仓库是一个 **联邦学习遗忘算法研究平台**，主要围绕 **FairVUE** 方法展开，  
同时集成了多种经典的联邦遗忘算法作为对照实验（PGA、FedEraser、Contribution Dampening）。

---

## 📂 文件结构概览

> 💡 仅 `.py` 文件为主要代码，其余多为实验报告与环境配置。  
> 若文件名含 `_test`，一般为中间调试文件，可忽略。

```text
📁 ./
├── 📁 .git/
├── 📁 FedUnlearner/
│   ├── 📁 __pycache__/
│   ├── 📁 attacks/                # 攻击模块（后门、MIA、投毒）
│   │   ├── 📄 backdoor.py
│   │   ├── 📄 mia.py
│   │   └── 📄 poisoning.py
│   ├── 📁 baselines/
│   │   ├── 📄 __init__.py
│   │   ├── 📁 fair_vue/          # ⭐ 主方法（FairVUE）
│   │   │   ├── 📄 fisher.py      # Fisher 信息矩阵估计
│   │   │   ├── 📄 healing.py     # Healing阶段：遗忘后模型恢复
│   │   │   ├── 📄 pipeline.py    # FairVUE整体流程封装
│   │   │   ├── 📄 projection.py  # 投影矩阵构造与应用
│   │   │   └── 📄 subspace.py    # SVD分解与ρ划分的核心算子
│   │   ├── 📁 fed_eraser/        # FedEraser算法：几何重构式遗忘
│   │   │   └── 📄 fed_eraser.py
│   │   ├── 📁 fedfim/            # Fisher聚合方法（尚未包含遗忘模块）
│   │   │   ├── 📄 client.py
│   │   │   ├── 📄 dampening.py
│   │   │   ├── 📄 fedfim.py
│   │   │   └── 📄 server.py
│   │   └── 📁 pga/               # PGA算法：梯度上升式遗忘
│   │       ├── 📄 pga.py
│   │       ├── 📄 pga_model.py
│   │       └── 📄 pga_utils.py
│   ├── 📄 data_utils.py          # 数据加载与划分（CIFAR-10 / MNIST，支持 IID/Dirichlet 非IID）
│   ├── 📄 fed_learn.py           # 标准 FedAvg 联邦训练流程
│   ├── 📄 models.py              # 模型定义（CNN / ResNet）
│   ├── 📄 unlearn.py             # Contribution Dampening 遗忘方法（未使用）
│   └── 📄 utils.py               # 工具函数集合
├── 📁 data/                      # 数据集
├── 📁 experiments/               # 实验（模型）
├── 📁 logs/                      # 日志
├── 📁 venv/
├── 📁 weights_demo/
├── 📄 .gitignore
├── 📄 GPU_test.py                # 测试 PyTorch 环境是否可用
├── 📄 fair_vue_demo.py           # FairVUE 简易演示脚本
├── 📄 main.py                    # 主程序入口
├── 📄 requirements.txt           # 依赖文件
└── 📄 实验报告 / 研究总结 / 实验结果 相关文档
````

---

## ⚙️ 已包含的联邦遗忘算法

| 算法                         | 理论基础              | 主要文件                                                   | 是否为 FairVUE 核心组成 | 适合作为对照     |
| -------------------------- | ----------------- | ------------------------------------------------------ | ---------------- | ---------- |
| **FairVUE**                | Fisher 加权 + 子空间投影 | `fisher.py`, `subspace.py`, `projection.py`, `main.py` | ✅ 是              | ✅ 主算法      |
| **PGA**                    | 梯度上升方向优化          | `pga.py`, `pga_utils.py`                               | ❌ 否              | ✅ baseline |
| **FedEraser**              | 参数几何重构            | `fed_eraser.py`                                        | ❌ 否              | ✅ baseline |
| **Contribution Dampening** | 参数贡献度衰减           | `unlearn.py`                                           | ❌ 否              | ✅ baseline |

> 🧩 FairVUE 是主研究对象，其他三种算法作为性能、稳定性、时间效率等指标的对照组。

---

## 🧠 FairVUE 方法简介

**FairVUE（Federated Unlearning via Fair Variance Subspace Projection）**
是一种基于**Fisher 信息加权的参数子空间投影方法**，用于在不访问原始数据的前提下进行客户端遗忘。

主要步骤：

1. 计算各客户端模型更新的 Fisher 加权矩阵；
2. 对加权矩阵执行 SVD 分解，提取敏感方向（专属子空间）与公共方向；
3. 构造投影矩阵，在敏感方向上剔除目标客户端的特征；
4. 可选执行 Healing 阶段以修复模型性能。

**优点：**

* 理论基础清晰（信息几何 + 公平性约束）
* 不依赖原始数据，计算高效
* 可视化解释性强

---

## 🧪 环境与运行

### 🧰 环境配置

```bash
conda create -n fedunlearner python=3.10
conda activate fedunlearner
pip install -r requirements.txt
```

### 🚀 运行示例

```bash
python main.py
```

或运行 FairVUE 简化演示：

```bash
python fair_vue_demo.py
```

---

## 📜 说明

* 本仓库旨在提供 **可扩展的联邦遗忘研究框架**；
* FairVUE 为主算法，其余为 **对照算法（baseline）**；
* 若希望加入自定义方法，可直接在 `FedUnlearner/baselines/` 下新增模块；
* 部分 `fedfim` 模块正在开发中，用于探索 **Fisher 聚合 + 遗忘协同机制**。

---

## ✍️ 作者与声明

> 本项目由 **0XCOFFEE** 基于公开框架二次开发与理论复现，
> 主要用于科研探索与 FairVUE 方法验证。
> 转载请注明出处。


