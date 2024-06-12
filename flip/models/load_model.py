import torch
from .CNN import CNN


def load_CNN(cfg):
    model = CNN(mean = cfg.data.mean, std = cfg.data.std, 
                       ksize1 = 5, ksize2 = 5, stride = 1).to(cfg.device)
    name = getattr(cfg.model, 'file_name', 'cnn.pt')
    model.load_state_dict(torch.load(cfg.model.path + name, map_location=cfg.device))
    model.eval()
    return model


model_dict = {'CNN': load_CNN,}

def load(cfg):
    name = cfg.model.name
    if name in model_dict.keys():
        return model_dict[name](cfg)
    else:
        raise ValueError('Unknown model: ' +str(name))