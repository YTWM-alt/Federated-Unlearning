import os

# 目标路径
folder_path = "./cifar10+allcnn批量实验结果"

# 关键短语
targets = [
    "[Training模型] 忘却客户端",
    "[Retrain模型] 忘却客户端"
]

# 遍历文件夹
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".log"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines:
                    for target in targets:
                        if target in line:
                            print(f"{file}: {line.strip()}")
