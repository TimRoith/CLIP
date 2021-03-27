import torch




def no_att():
    #
    def no_attack(model, x, y):
        return x
    #
    return no_attack


# get gauss noise augmentation. Magnitude is dependent on noise level
def gauss_noise(nl):
    #
    def gauss_attack(model, x, y):
        return x+torch.randn_like(x) * nl
    #
    return gauss_attack
