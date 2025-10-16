import torch
import time

def benchmark(device, size=5000, rounds=5):
    print(f"\n在 {device} 上测试矩阵乘法 (size={size}x{size}, {rounds} 次)...")
    x = torch.rand((size, size), device=device)
    y = torch.rand((size, size), device=device)

    # 预热 (避免第一次计算时间异常)
    torch.matmul(x, y)

    start = time.time()
    for _ in range(rounds):
        z = torch.matmul(x, y)
    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()

    avg_time = (end - start) / rounds
    print(f"平均耗时: {avg_time:.4f} 秒/次")
    print(f"结果矩阵在设备: {z.device}")


if __name__ == "__main__":
    # CPU benchmark
    print(torch.version.cuda)
    benchmark("cpu")

    # GPU benchmark（如果可用）
    if torch.cuda.is_available():
        benchmark("cuda")
    else:
        print("CUDA 不可用，未进行 GPU 测试。")
