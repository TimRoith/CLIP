import torch
from flip.attacks import pgd, fgsm
from flip.utils.config import cfg, dataset, model_attributes

def eval_acc(model, x, y):
    return torch.sum(model(x).topk(1)[1][:,0]==y)

def get_attack(type = pgd, max_iter=100, epsilon=0.8, proj='linf', tau = None, loss = None, targeted = False, x_range = None, opt_kwargs = None):
    attack = type(proj=proj, max_iters=max_iter, epsilon=epsilon)
    return attack

def attack_model(model, dataloader, attack_kwargs = None, attack_iter = 10):
    attack_kwargs = {'type':"pgd"} if attack_kwargs is None else attack_kwargs
    type = attack_kwargs.get('type', "pgd")
    attack_kwargs = {k:v for k,v in attack_kwargs.items() if k != 'type'}
    if type == "pgd":
        attack = get_attack(type=pgd, **attack_kwargs)
    elif type == "fgsm":
        attack = get_attack(type=fgsm, **attack_kwargs)
    else:
        raise ValueError("Unknown attack type: " + str(type))
    eval = 0
    for i in range(attack_iter):
        x,y = next(iter(dataloader))
        attack(model, x, y)
        delta = attack.delta
        eval += eval_acc(model, x+delta, y)/len(y)
    return eval/attack_iter
    