import os
import requests
import zipfile
import io
from tqdm import tqdm
import shutil

# 配置路径
DATA_ROOT = "./data"
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TARGET_DIR = os.path.join(DATA_ROOT, "tiny-imagenet-200")

def download_and_extract():
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    # 1. 检查是否已存在
    if os.path.exists(TARGET_DIR):
        print(f"[提示] 目录 {TARGET_DIR} 已存在，跳过下载。")
        return

    print(f"[1/3] 正在下载 Tiny-ImageNet ({URL})...")
    response = requests.get(URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # 使用内存缓冲进行解压，避免产生临时的 zip 文件
    with io.BytesIO() as buffer:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载中") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)
                pbar.update(len(chunk))
        
        print("[2/3] 正在解压...")
        with zipfile.ZipFile(buffer) as zf:
            zf.extractall(path=DATA_ROOT)
            
    print("下载并解压完成！")

def format_val_dataset():
    """
    Tiny-ImageNet 的验证集原始格式是所有图片混在一个文件夹里，
    我们需要根据 val_annotations.txt 把它整理成 ImageFolder 能识别的格式：
    val/
       class_1/
          img1.jpg
       class_2/
          img2.jpg
    """
    val_dir = os.path.join(TARGET_DIR, "val")
    img_dir = os.path.join(val_dir, "images")
    annot_file = os.path.join(val_dir, "val_annotations.txt")
    
    if not os.path.exists(img_dir):
        print("[提示] 验证集似乎已经格式化过了 (images 文件夹不存在)，跳过整理。")
        return

    print("[3/3] 正在重新格式化验证集 (适配 PyTorch ImageFolder)...")
    
    # 读取标注
    with open(annot_file, 'r') as f:
        lines = f.readlines()
    
    # 移动图片
    count = 0
    for line in tqdm(lines, desc="移动图片"):
        parts = line.strip().split('\t')
        file_name = parts[0]
        class_label = parts[1]
        
        # 创建类别子文件夹
        class_sub_dir = os.path.join(val_dir, class_label)
        os.makedirs(class_sub_dir, exist_ok=True)
        
        # 源路径和目标路径
        src_path = os.path.join(img_dir, file_name)
        dst_path = os.path.join(class_sub_dir, file_name)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            count += 1
            
    # 清理空的 images 文件夹
    if os.path.exists(img_dir) and not os.listdir(img_dir):
        os.rmdir(img_dir)
        
    print(f"完成！已成功整理 {count} 张验证集图片。")

if __name__ == "__main__":
    try:
        download_and_extract()
        format_val_dataset()
        print(f"\n成功！数据集已准备好，位于: {os.path.abspath(TARGET_DIR)}")
    except Exception as e:
        print(f"\n[错误] 发生异常: {e}")
        print("如果下载太慢，建议你在本地电脑下载好 zip 包，上传到服务器的 ./data/ 目录下手动解压。")