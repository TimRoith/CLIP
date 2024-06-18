import torch
from flip.models import load_model, save_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer
from flip.load_data import load_MNIST_test, load_MNIST
from flip.utils.config import cfg, dataset, model_attributes
import matplotlib.pyplot as plt
import time 

#%%
def eval_acc(model, x, y):
    return torch.sum(model(x).topk(1)[1][:,0]==y)

#%%
time_v = time.time()

CFG = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_warmup_clip_v' + str(round(time_v)) + '.pth',
              )
          )

model = load_model.load(CFG)
dataloader= load_MNIST(CFG)
x,y = next(iter(dataloader))
starting_eval = eval_acc(model, x, y)/len(y)
print('Starting accuracy: ', starting_eval)
#%%
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
                          lamda=0,
                          num_iters=1,
                          estimation='sum',
                          opt_kwargs={'type': torch.optim.Adam },
                          adv_kwargs={'name' : 'Adam', 'lr' : 0.01},
                          verbosity=1,
                          epochs=100,
                          min_acc=1.,)

start_time = time.time()
trainer.train()
elapsed_time = time.time() - start_time


#%%
attack = pgd(proj='linf', max_iters=500, epsilon=0.8)


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

#%%
#save model
save_model.save(model, CFG)