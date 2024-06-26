import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer
from flip.load_data import load_MNIST_test, load_MNIST
from flip.utils.config import cfg, dataset, model_attributes
from flip.test import attack_model, eval_acc
import matplotlib.pyplot as plt
import time 

#%%
CFG = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_compare_adv.pth',
              )
          )

#%%

dataloader= load_MNIST(CFG)
epochs = 10

#%%
model_ADV = load_model.load(CFG)
trainer = AdversarialTrainer(model_ADV, dataloader,
                          opt_kwargs={'type': torch.optim.Adam },
                          verbosity=1,
                          epochs=epochs,)

start_time = time.time()
trainer.train()
elapsed_time_ADV = time.time() - start_time

acc_ADV = attack_model(model_ADV, dataloader, attack_kwargs = {'type':"pgd"}, attack_iter = 20)

hist_ADV = trainer.hist.copy()

#%%
CFG.model.file_name = 'model_compare_max.pth'
model_MAX = load_model.load(CFG)
trainer = FLIPTrainer(model_MAX, dataloader,
                          lamda=0.1,
                          num_iters=2,
                          estimation='max',
                          opt_kwargs={'type': torch.optim.Adam },
                          adv_kwargs={'name' : 'SGD', 'lr' : 0.07},
                          verbosity=1,
                          epochs=epochs,
                          min_acc=1.,)

start_time = time.time()
trainer.train()
elapsed_time_MAX = time.time() - start_time

acc_MAX = attack_model(model_MAX, dataloader, attack_kwargs = {'type':"pgd"}, attack_iter = 20)

hist_MAX = trainer.hist.copy()

#%%
CFG.model.file_name = 'model_compare_sum.pth'
model_SUM = load_model.load(CFG)
trainer = FLIPTrainer(model_SUM, dataloader,
                          lamda=0.1,
                          num_iters=2,
                          estimation='sum',
                          opt_kwargs={'type': torch.optim.Adam },
                          adv_kwargs={'name' : 'SGD', 'lr' : 0.07},
                          verbosity=1,
                          epochs=epochs,
                          min_acc=1.,)

start_time = time.time()
trainer.train()
elapsed_time_SUM = time.time() - start_time

acc_SUM = attack_model(model_SUM, dataloader, attack_kwargs = {'type':"pgd"}, attack_iter = 20)

hist_SUM = trainer.hist.copy()


#%%
CFG.model.file_name = 'model_compare_sta.pth'
model_STA = load_model.load(CFG)
trainer = StandardTrainer(model_STA, dataloader,
                          opt_kwargs={'type': torch.optim.Adam },
                          verbosity=1,
                          epochs=epochs,)

start_time = time.time()
trainer.train()
elapsed_time_STA = time.time() - start_time

acc_STA = attack_model(model_STA, dataloader, attack_kwargs = {'type':"pgd"}, attack_iter = 20)

hist_STA = trainer.hist.copy()

#%%
plt.figure(figsize=(10, 5))

# Plot acc
plt.subplot(1, 2, 1)
plt.plot(hist_ADV['acc'], label='ADV')
plt.plot(hist_SUM['acc'], label='SUM')
plt.plot(hist_MAX['acc'], label='MAX')
plt.plot(hist_STA['acc'], label='STA')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(hist_ADV['loss'], label='ADV')
plt.plot(hist_SUM['loss'], label='SUM')
plt.plot(hist_MAX['loss'], label='MAX')
plt.plot(hist_STA['loss'], label='STA')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#%%
print('time SUM: ', elapsed_time_SUM)
print('time MAX: ', elapsed_time_MAX)
print('time ADV: ', elapsed_time_ADV)
print('time STA: ', elapsed_time_STA)
print('adv acc SUM: ', acc_SUM)
print('adv acc MAX: ', acc_MAX)
print('adv acc ADV: ', acc_ADV)
print('adv acc STA: ', acc_STA)