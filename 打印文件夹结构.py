import os

def print_structure(root, target="FedUnlearner", indent=0):
    """
    打印文件夹结构：
    - 普通文件夹：只打印文件夹名
    - target 文件夹：打印完整结构
    - 根目录下额外打印非文件夹文件
    """
    items = sorted(os.listdir(root))
    prefix = " " * indent
    print(f"{prefix}📁 {os.path.basename(root) or root}/")

    # 先打印根目录下的非文件夹文件
    if indent == 0:
        for item in items:
            path = os.path.join(root, item)
            if not os.path.isdir(path):
                print(f"{' ' * (indent + 4)}📄 {item}")

    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            # 如果是目标目录，递归打印完整结构
            if item == target:
                print_structure(path, target, indent + 4)
            # 如果当前目录已经在目标目录内部，也继续递归打印
            elif target in os.path.relpath(path).split(os.sep):
                print_structure(path, target, indent + 4)
            else:
                print(f"{' ' * (indent + 4)}📁 {item}/")
        elif target in os.path.relpath(path).split(os.sep):
            # 打印目标目录中的文件
            print(f"{' ' * (indent + 4)}📄 {item}")

if __name__ == "__main__":
    root_dir = "."  # 当前根目录
    print_structure(root_dir)
