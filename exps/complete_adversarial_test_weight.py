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


lenght = 20 # lenght of epsilon list
epsilon_test_list = np.logspace(-3, -0.2, lenght)

dataset_name = 'MNIST'
epochs = 30 # for each model : how many time is it trained

n_model = 2

time_v = []
for i in range(n_model):
    time_v.append(time.time())

CFG = cfg(data=dataset(name=dataset_name, download = False), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_compare_adv_v' + str(round(time_v[0])) + '.pth',
              )
          )

# split data
dataloader, validation_loader, test_loader = split_loader(CFG, train_split=0.8)

time_v = [1720527201,1720604115] # to be changed with name of the model


acc_ADV =[0]*lenght
acc_TV = [0]*lenght
acc_SUM = [0]*lenght
acc_STA = [0]*lenght
acc_pTV = [0]*lenght

adversarial_test = "fgsm"
print("------ Computation of the accuracy for each model ------")

for e in range(lenght):
    for i in range(n_model):
        CFG.model.file_name = 'model_compare_adv_v' + str(round(time_v[i])) + '.pth'
        model_ADV = load_model.load(CFG)
        acc_ADV[e] += attack_model(model_ADV, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test_list[e]})
        print(acc_ADV[e])
    acc_ADV[e] /= n_model


for e in range(lenght):
    for i in range(n_model):
        CFG.model.file_name = 'model_compare_TV_v' + str(round(time_v[i])) + '.pth'
        model_TV = load_model.load(CFG)
        acc_TV[e] += attack_model(model_TV, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test_list[e]})
    acc_TV[e] /= n_model


for e in range(lenght):
    for i in range(n_model):
        CFG.model.file_name = 'model_compare_SUM_v' + str(round(time_v[i])) + '.pth'
        model_SUM = load_model.load(CFG)
        acc_SUM[e] += attack_model(model_SUM, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test_list[e]})
    acc_SUM[e] /= n_model
    

for e in range(lenght):
    for i in range(n_model):
        CFG.model.file_name = 'model_compare_STA_v' + str(round(time_v[i])) + '.pth'
        model_STA = load_model.load(CFG)
        acc_STA[e] += attack_model(model_STA, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test_list[e]})
    acc_STA[e] /= n_model


for e in range(lenght):
    for i in range(n_model):
        CFG.model.file_name = 'model_compare_pTV_v' + str(round(time_v[i])) + '.pth'
        model_pTV = load_model.load(CFG)
        acc_pTV[e] += attack_model(model_pTV, test_loader, attack_kwargs = {'type': adversarial_test, 'epsilon': epsilon_test_list[e]})
    acc_pTV[e] /= n_model
    
print("------ Plotting ------")
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