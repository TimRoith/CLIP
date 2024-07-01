import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer, AdvFlipTrainer
from flip.load_data import load_MNIST_test, load_MNIST, split_loader
from flip.utils.config import cfg, dataset, model_attributes
from flip.test import attack_model, eval_acc
import matplotlib.pyplot as plt
import time 

#%%
time_v = time.time()

CFG = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_lipschitz_adv_v' + str(round(time_v)) + '.pth',
              )
          )

#%%

# split data
dataloader, validation_loader, test_loader = split_loader(CFG, train_split=0.8)

epochs = 10

#%%
model_ADV = load_model.load(CFG)
trainer = AdversarialTrainer(model_ADV, dataloader, val_loader=validation_loader,
                          opt_kwargs={'type': torch.optim.Adam },
                          adv_kwargs = {'type' : "pgd", 'epsilon' : 0.05},
                          verbosity=1,
                          epochs=epochs,)
print('init adv acc for adversarial: ', attack_model(model_ADV, dataloader, attack_kwargs = {'type':"pgd", 'epsilon':0.3, 'max_iter':1})) # Expected to be near 0
print('Begin Adversarial Training')
start_time = time.time()
trainer.train()
elapsed_time_ADV = time.time() - start_time

acc_ADV = attack_model(model_ADV, test_loader, attack_kwargs = {'type':"fgsm", 'epsilon': 0.1})

hist_ADV = trainer.hist.copy()

#%%
model_final = load_model.load(CFG)
trainer = FLIPTrainer(model_final, dataloader, val_loader=validation_loader,
                          lamda=0.7,
                          num_iters=2,
                          estimation='sum',
                          opt_kwargs={'type': torch.optim.Adam },
                          upd_kwargs={'name' : 'SGD', 'lr' : 0.07},
                          verbosity=1,
                          epochs=1,
                          min_acc=1.,)

print('init adv acc for already adversarial trained model : ', attack_model(model_final, dataloader, attack_kwargs = {'type':"fgsm", 'epsilon':0.3, 'max_iter':1})) # Expected to be near 0
print('Begin FLIP - Sum new training')
start_time = time.time()
trainer.train()
elapsed_time_final = time.time() - start_time

acc_final = attack_model(model_final, test_loader, attack_kwargs = {'type':"fgsm", 'epsilon': 0.1})

hist_final = trainer.hist.copy()

#%%
CFG.model.file_name = 'model_lipschitz_v' + str(round(time_v)) + '.pth'
model = load_model.load(CFG)
trainer = AdvFlipTrainer(model, dataloader, val_loader=validation_loader,
                            lamda=0.7,
                            num_iters=2,
                            estimation='sum',
                            opt_kwargs={'type': torch.optim.Adam },
                            adv_kwargs = {'type' : "fgsm", 'epsilon' : 0.05},
                            upd_kwargs={'name' : 'SGD', 'lr' : 0.07},
                            verbosity=1,
                            epochs=epochs,
                            min_acc=0.9,)
print('init adv acc for adversarial & lipschitz trained model : ', attack_model(model, dataloader, attack_kwargs = {'type':"fgsm", 'epsilon':0.3, 'max_iter':1})) # Expected to be near 0
print('Begin FLIP - ADV new training')
start_time = time.time()
trainer.train()
elapsed_time_advflip = time.time() - start_time

acc_advflip = attack_model(model, test_loader, attack_kwargs = {'type':"fgsm", 'epsilon': 0.1})

hist_advflip = trainer.hist.copy()

#%%
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(hist_ADV['acc'], label='Adversarial')
plt.plot(hist_final['acc'], label='Final')
plt.plot(hist_advflip['acc'], label='AdvFlip')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(hist_ADV['loss'], label='Adversarial')
plt.plot(hist_final['loss'], label='Final')
plt.plot(hist_advflip['loss'], label='AdvFlip')
plt.legend()
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

#%%
print('Adversarial Training Time: ', elapsed_time_ADV)
print('Final Training Time: ', elapsed_time_final)
print('AdvFlip Training Time: ', elapsed_time_advflip)
print('Adversarial Test Accuracy: ', acc_ADV)
print('Final Test Accuracy: ', acc_final)
print('AdvFlip Test Accuracy: ', acc_advflip)
