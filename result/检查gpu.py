import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"发现 {gpu_count} 个GPU 可用:")

    # 遍历每个GPU并打印设备名称
    for i in range(gpu_count):
        print(f"GPU 设备 {i}: {torch.cuda.get_device_name(i)}")
else:
    print("未发现可用的GPU，将使用CPU进行计算。")