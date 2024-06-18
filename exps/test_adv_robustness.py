import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.load_data import load_MNIST_test
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer
from flip.utils.config import cfg, dataset, model_attributes
import matplotlib.pyplot as plt




CFG = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_adv_training_2.pth', #'model_sum_clip.pth'
              )
          )

model = load_model.load(CFG)
dataloader= load_MNIST_test(CFG)
#%%
Trainer = AdversarialTrainer(model, dataloader,
                          opt_kwargs={'type': torch.optim.Adam },
                          verbosity=1,
                          epochs=50,)

Trainer.train()
#%%
attack = pgd(proj='linf', max_iters=1000, epsilon=0.8)

#%%
def eval_acc(model, x, y):
    return torch.sum(model(x).topk(1)[1][:,0]==y)

#%% attack
x,y = next(iter(dataloader))
attack(model, x, y)
delta = attack.delta
eval_acc(model, x+delta, y)
print('Accuracy: ', eval_acc(model, x+delta, y)/len(y))


#%%
logit = model(x+delta).topk(1)[1][:,0]
for i in range(x.shape[0]):
    if logit[i] == y[i]:
        print(i)
        print(logit[i])
        plt.imshow((x+delta)[i,0,...].detach())
        break