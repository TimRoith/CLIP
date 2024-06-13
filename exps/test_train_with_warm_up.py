import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer
from flip.load_data import load_MNIST_test, load_MNIST
from flip.utils.config import cfg, dataset, model_attributes
import matplotlib.pyplot as plt

#%%
CFG = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_warmup_clip.pth', #'model_sum_clip.pth'
              )
          )

model = load_model.load(CFG)
dataloader= load_MNIST(CFG)
trainer = StandardTrainer(model, dataloader, 
                          opt_kwargs={'type': torch.optim.Adam},
                          verbosity=1,
                          epochs=1,)
trainer.train()
#%%
set_acc = 0.9
while trainer.hist['acc'][-1] < set_acc:
    trainer.train()

print('end of warm up with accuracy: ', trainer.hist['acc'][-1].item())
#%%

trainer = FLIPTrainer(model, dataloader, 
                          opt_kwargs={'type': torch.optim.Adam},
                          adv_kwargs={'name' : 'Adam', 'lr' : 0.01},
                          verbosity=1,
                          epochs=100,)

trainer.train()



#%%
attack = pgd(proj='linf', max_iters=500, epsilon=0.3)

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