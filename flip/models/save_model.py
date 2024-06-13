import torch
from .CNN import CNN
from .FC import FC

def save(model, cfg):
    torch.save(model.state_dict(), cfg.model.path + cfg.model.file_name)