import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 版本:", torch.version.cuda)
print("GPU 是否可用:", torch.cuda.is_available())
print("GPU 名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU")
