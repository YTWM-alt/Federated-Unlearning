import os

# 目标路径
folder_path = "./logs/fairvue_grid"

# 关键短语
targets = [
    "Performance after FAIR-VUE",
    "[FAIR-VUE模型] 忘却客户端"
]

# 收集所有 .log 文件及其修改时间
log_files = []
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".log"):
            path = os.path.join(root, file)
            mtime = os.path.getmtime(path)
            log_files.append((path, mtime))

# 按修改时间降序排列（最新的在前）
log_files.sort(key=lambda x: x[1], reverse=True)

# 遍历文件并筛选目标内容
for file_path, _ in log_files:
    matched_lines = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for target in targets:
                if target in line:
                    matched_lines.append(line.strip())

    if matched_lines:
        print(f"\n{file_path}")
        for l in matched_lines:
            print("  " + l)