import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.load_data import load_MNIST_test, split_loader
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer
from flip.utils.config import cfg, dataset, model_attributes
from flip.test import attack_model, eval_acc, test_acc
import matplotlib.pyplot as plt
import time 

time_v = time.time()

CFG = cfg(data=dataset(name='MNIST'), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10], # [3072, 128, 80, 10]
              act_fun = 'ReLU',
              file_name = 'model_adv_training_v'+ str(round(time_v)) + '.pth',
              )
          )

model = load_model.load(CFG)

#%%

# split data
dataloader, validation_loader, test_loader = split_loader(CFG, train_split=0.8)

epochs = 100
#%%
Trainer = AdversarialTrainer(model, dataloader,
                          opt_kwargs={'type': torch.optim.Adam },
                          adv_kwargs = {'type' : "fgsm", 'epsilon' : 4/255},
                          verbosity=1,
                          epochs=epochs,)

Trainer.train()
#%%

acc_final = attack_model(model, test_loader, attack_kwargs = {'type':"fgsm", 'epsilon': 4/255, 'proj':'l2', 'max_iters': 10})
print('Accuracy adv : ', acc_final)

#%%

acc_test = test_acc(model, test_loader)
print('Accuracy : ', acc_test)
