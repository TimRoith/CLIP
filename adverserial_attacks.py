import torch








# get gauss noise augmentation. Magnitude is dependent on noise level
def gauss_noise(nl, x):
    return torch.randn_like(x) * nl
