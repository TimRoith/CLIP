import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer, TVTrainer
from flip.load_data import load_MNIST_test, load_MNIST, split_loader
from flip.utils.config import cfg, dataset, model_attributes
from flip.test import attack_model, eval_acc
import numpy as np
import matplotlib.pyplot as plt
import time 

#%%
epsilon_test_list = np.logspace(-3, -0.2, 20)
adversarial_test = "pgd"

time_v = time.time()

CFG = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_compare_adv_v' + str(round(time_v)) + '.pth',
              )
          )

#%%

# split data
dataloader, validation_loader, test_loader = split_loader(CFG, train_split=0.8)

epochs = 4

#%%
acc_ADV =[]
acc_TV = []
acc_SUM = []
acc_STA = []
acc_pTV = []


model_ADV = load_model.load(CFG)
trainer = AdversarialTrainer(model_ADV, dataloader, val_loader=validation_loader,
                        opt_kwargs={'type': torch.optim.Adam },
                        adv_kwargs = {'type' : "fgsm", 'epsilon' : 0.05},
                        verbosity=1,
                        epochs=epochs,)
print('init adv acc for adversarial: ', attack_model(model_ADV, dataloader, attack_kwargs = {'type':"fgsm", 'epsilon':1., 'max_iters':1})) # Expected to be near 0
print('Begin Adversarial Training')
start_time = time.time()
trainer.train()
elapsed_time_ADV = time.time() - start_time

for epsilon_test in epsilon_test_list:
    acc_ADV.append(attack_model(model_ADV, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test}))

#%%
CFG.model.file_name = 'model_compare_TV_v' + str(round(time_v)) + '.pth'
model_TV = load_model.load(CFG)
trainer = TVTrainer(model_TV, dataloader, val_loader=validation_loader,
                        lamda=0.7,
                        num_iters=2,
                        approximation_decrep=0.3,
                        opt_kwargs={'type': torch.optim.Adam },
                        upd_kwargs={'name' : 'SGD', 'lr' : 0.07},
                        verbosity=1,
                        epochs=epochs,
                        min_acc=1.,)

print('init adv acc for TV: ', attack_model(model_TV, dataloader, attack_kwargs = {'type':"fgsm", 'epsilon':1., 'max_iters':1})) # Expected to be near 0
print('Begin FLIP - TV Training')
start_time = time.time()
trainer.train()
elapsed_time_TV = time.time() - start_time

for epsilon_test in epsilon_test_list:
    acc_TV.append(attack_model(model_TV, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test}))

#%%
CFG.model.file_name = 'model_compare_sum_v' + str(round(time_v)) + '.pth'
model_SUM = load_model.load(CFG)
trainer = FLIPTrainer(model_SUM, dataloader, val_loader=validation_loader,
                        lamda=0.7,
                        num_iters=2,
                        estimation='sum',
                        opt_kwargs={'type': torch.optim.Adam },
                        upd_kwargs={'name' : 'SGD', 'lr' : 0.07},
                        verbosity=1,
                        epochs=epochs,
                        min_acc=1.,)

print('init adv acc for SUM: ', attack_model(model_SUM, dataloader, attack_kwargs = {'type':"fgsm", 'epsilon':1., 'max_iters':1})) # Expected to be near 0
print('Begin FLIP - SUM Training')
start_time = time.time()
trainer.train()
elapsed_time_SUM = time.time() - start_time
for epsilon_test in epsilon_test_list:
    acc_SUM.append(attack_model(model_SUM, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test}))

#%%
CFG.model.file_name = 'model_compare_sta_v' + str(round(time_v)) + '.pth'
model_STA = load_model.load(CFG)
trainer = StandardTrainer(model_STA, dataloader, val_loader=validation_loader,
                        opt_kwargs={'type': torch.optim.Adam },
                        verbosity=1,
                        epochs=epochs,)

print('init adv acc for STA: ', attack_model(model_STA, dataloader, attack_kwargs = {'type':"fgsm", 'epsilon':1., 'max_iters':1})) # Expected to be near 0
print('Begin Standard Training')
start_time = time.time()
trainer.train()
elapsed_time_STA = time.time() - start_time
for epsilon_test in epsilon_test_list:
    acc_STA.append(attack_model(model_STA, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test}))

#%%
CFG.model.file_name = 'model_compare_pTV_v' + str(round(time_v)) + '.pth'
model_pTV = load_model.load(CFG)
trainer = TVTrainer(model_pTV, dataloader, val_loader=validation_loader,
                        lamda=0.7,
                        num_iters=2,
                        projection=True,
                        approximation_decrep=0.3,
                        opt_kwargs={'type': torch.optim.Adam },
                        upd_kwargs={'name' : 'SGD', 'lr' : 0.07},
                        verbosity=1,
                        epochs=epochs,
                        min_acc=1.,)

print('init adv acc for pTV: ', attack_model(model_pTV, dataloader, attack_kwargs = {'type':"fgsm", 'epsilon':1., 'max_iters':1})) # Expected to be near 0
print('Begin FLIP - pTV Training')
start_time = time.time()
trainer.train()
elapsed_time_pTV = time.time() - start_time
for epsilon_test in epsilon_test_list:
    acc_pTV.append(attack_model(model_pTV, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test}))

#%%

print('time SUM: ', elapsed_time_SUM)
print('time TV: ', elapsed_time_TV)
print('time pTV: ', elapsed_time_pTV)
print('time ADV: ', elapsed_time_ADV)
print('time STA: ', elapsed_time_STA)

plt.figure(figsize=(10, 5))

# Plot acc
plt.plot(acc_ADV, label='ADV')
plt.plot(acc_SUM, label='SUM')
plt.plot(acc_TV, label='TV')
plt.plot(acc_pTV, label='pTV')
plt.plot(acc_STA, label='STA')
plt.xlabel('Epsilon')
plt.xscale('log')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
