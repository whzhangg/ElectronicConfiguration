import torch

torch_dtype = torch.float
#torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_device = 'cpu'   # it turns out that cuda is available on linux as well