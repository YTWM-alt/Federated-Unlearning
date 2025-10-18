# 介绍 文档
## 文件结构一览(主要代码文件，只看.py结尾的就行)

#### tips：没注释的文件不用管，如各类_test结尾的临时阶段化测试文件
📁 ./
├── 📁 .git/
├── 📁 Assets/
├── 📁 FedUnlearner/
│   ├── 📁 __pycache__/
│   ├── 📁 attacks/                # 攻击相关
│   │   ├── 📁 __pycache__/
│   │   ├── 📄 backdoor.py
│   │   ├── 📄 mia.py
│   │   └── 📄 poisoning.py
│   ├── 📁 baselines/
│   │   ├── 📄 __init__.py
│   │   ├── 📁 __pycache__/
│   │   ├── 📁 fair_vue/          # 主方法，**只有这个是有用的**
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📁 __pycache__/
│   │   │   ├── 📄 fisher.py      # Fisher对角信息估计模块
│   │   │   ├── 📄 healing.py     # 治疗阶段
│   │   │   ├── 📄 pipeline.py    # FairVUE 工作流封装
│   │   │   ├── 📄 projection.py  # 投影操作：构造和应用子空间投影矩阵
│   │   │   └── 📄 subspace.py    # 子空间与矩阵变换核心：SVD分解与ρ划分的基础算子
│   │   ├── 📁 fed_eraser/        # 一种高效的联邦遗忘算法
│   │   │   ├── 📁 __pycache__/
│   │   │   └── 📄 fed_eraser.py
│   │   ├── 📁 fedfim/            # Fisher聚合的方法，还未包括遗忘
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📁 __pycache__/
│   │   │   ├── 📄 client.py
│   │   │   ├── 📄 dampening.py
│   │   │   ├── 📄 fedfim.py
│   │   │   └── 📄 server.py
│   │   └── 📁 pga/               # 梯度上升遗忘方法
│   │       ├── 📁 __pycache__/
│   │       ├── 📄 pga.py
│   │       ├── 📄 pga_model.py
│   │       └── 📄 pga_utils.py
│   ├── 📄 data_utils.py          # 数据加载与划分（CIFAR-10/MNIST，IID/Dirichlet非IID）
│   ├── 📄 fed_learn.py           # 实现标准 FedAvg 联邦平均训练流程
│   ├── 📄 models.py              # 模型定义（CNN/ResNet）
│   ├── 📄 unlearn.py             # 自带遗忘方法（Contribution Dampening，未使用）
│   └── 📄 utils.py               # 工具函数集合
├── 📁 __pycache__/
├── 📁 data/
├── 📁 docs/
├── 📁 experiments/
├── 📁 logs/
├── 📁 venv/
├── 📁 weights_demo/
├── 📄 .gitignore
├── 📄 GPU_test.py #可以用来测当前python，torch环境
├── 📄 README.md
├── 📄 dempanding_test.py
├── 📄 fair_vue_demo.py
├── 📄 fedserver_test.py
├── 📄 fim_test.py
├── 📄 main.py #主程序
├── 📄 requirements.txt
├── 📄 test.py
├── 📄 实验报告.md
├── 📄 实验报告.pdf
├── 📄 实验报告——第二次.md
├── 📄 实验报告——第二次.pdf
├── 📄 实验结果.txt
├── 📄 实验结果2.txt
├── 📄 打印文件夹结构.py
├── 📄 现阶段研究总结.md


### 所包含遗忘方法附表：
| 算法                         | 理论基础              | 主要文件                                                | 是否为 FairVUE 组成部分 | 适合作为对照     |
| -------------------------- | ----------------- | --------------------------------------------------- | ---------------- | ---------- |
| **FairVUE**                | Fisher 加权 + 子空间投影 | `fisher.py` `subspace.py` `projection.py` `main.py` | ✅ 是              | ✅ 是（主算法）   |
| **PGA**                    | 梯度上升方向优化          | `pga.py` `pga_utils.py`                             | ❌ 否              | ✅ baseline |
| **FedEraser**              | 几何重构              | `fed_eraser.py`                                     | ❌ 否              | ✅ baseline |
| **Contribution Dampening** | 参数贡献度统计           | `unlearn.py`                                        | ❌ 否              | ✅ baseline |
