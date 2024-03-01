import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(torch.__version__)
print(torch.cuda.device_count())  # 显示CUDA设备数量

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))  # 显示每个CUDA设备的名称
# 检查CUDA是否可用
if torch.cuda.is_available():
    # 创建一个张量并将其放置在CUDA设备上
    tensor = torch.tensor([1, 2, 3]).cuda()

    # 打印张量及其所在设备

    print("Tensor on CUDA device:", tensor)
    print("Device:", tensor.get_device())
else:
    print("CUDA is not available.")