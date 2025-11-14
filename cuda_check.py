import torch
print("cuda is available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
print("device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

#%%
print("torch version:", torch.__version__)
print("torch.cuda version:", torch.version.cuda)
