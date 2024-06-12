import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.load_data import load_MNIST_test
from flip.utils.config import cfg, dataset, model_attributes




CFG = cfg(data=dataset(), model=model_attributes())

model = load_model.load(CFG)
dataloader= load_MNIST_test(CFG)
attack = pgd(proj='linf', max_iters=10, epsilon=0.1)

#%%
def eval_acc(model, x, y):
    return torch.sum(model(x).topk(1)[1][:,0]==y)

#%% attack
x,y = next(iter(dataloader))
attack(model, x, y)
delta = attack.delta
eval_acc(model, x+delta, y)