import torch
import numpy as np
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer, TVTrainer
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
              file_name = 'model_compare_adv_v' + str(round(time_v)) + '.pth',
              )
          )

dataloader, validation_loader, test_loader = split_loader(CFG, train_split=0.8)

epochs = 5
epsilon_list = np.logspace(-6, 0, 30)

#%%
acc_TV = []
hist_TV = []
elapsed_time_TV = []

for epsilon in epsilon_list:
    time_v = time.time()
    CFG.model.file_name = 'model_compare_TV_v' + str(round(time_v)) + '.pth'
    model_TV = load_model.load(CFG)
    trainer = TVTrainer(model_TV, dataloader, val_loader=validation_loader,
                            lamda=0.7,
                            num_iters=2,
                            approximation=epsilon,
                            approximation_decrep=1,
                            opt_kwargs={'type': torch.optim.Adam },
                            upd_kwargs={'name' : 'SGD', 'lr' : 0.07},
                            verbosity=1,
                            epochs=epochs,
                            min_acc=1.,)

    start_time = time.time()
    trainer.train()
    elapsed_time_TV.append(time.time() - start_time)

    acc_TV.append(attack_model(model_TV, test_loader, attack_kwargs = {'type':"fgsm", 'epsilon': 0.1}))

    hist_TV.append(trainer.hist.copy())

#%%

plt.subplot(1,3,1)
plt.plot(epsilon_list, acc_TV)
plt.xlabel('Approximation')
plt.ylabel('Adversarial Accuracy')
plt.title('Adversarial Accuracy (approximation)')
plt.grid()

plt.subplot(1,3,2)
plt.plot(epsilon_list, elapsed_time_TV)
plt.xlabel('Approximation')
plt.ylabel('Elapsed Time')
plt.title('Elapsed Time (approximation)')
plt.grid()

plt.subplot(1,3,3)
plt.plot(epsilon_list, [h['acc'][-1] for h in hist_TV])
plt.xlabel('Approximation')
plt.ylabel('Accuracy')
plt.title('Accuracy (approximation)')
plt.grid()

plt.show()
