import os
import torch
from utils.configuration import Conf
from utils.datasets import get_data_set
import models
from train import train_step, validation_step
import adversarial_attacks as at

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

data_file = "/"

conf_args = {'alpha':0.00,'data_file':data_file, 'use_cuda':True, 'train_split':0.9, 'num_workers':4,
             'regularization': "none", 'reg_init': "partial_random"}
conf = Conf(**conf_args)


#%% get train, validation and test loader

train_loader, valid_loader, test_loader = get_data_set(conf)

# # %% deine the model
model = models.fully_connected([784, 400, 200, 10], conf.activation_function)
model.to(conf.device)

# define the adverserial attack
conf.attack = at.fgsm(model, conf.loss)
#conf.attack = at.pgd(model, conf.loss, attack_iters=5)



print("Train model: {}".format(conf.model))
opt = torch.optim.SGD(model.parameters(), lr = 0.1, momentum =0.9)

# initalize history
history = {}
history['train_loss'] = []
history['train_acc'] = []
history['train_lip_loss'] = []
history['val_loss'] = []
history['val_acc'] = []
history['val_lip_loss'] = []

for i in range(conf.epochs):
    print(25*"<>")
    print(50*"|")
    print(25*"<>")
    print('Epoch', i)
    
    # train_step
    train_data = train_step(conf, model, opt, train_loader, valid_loader)
    
    # get adverserial pair from train_step
    idx = train_data['highest_loss_idx']
    u_reg = train_data['u'][idx:idx+1]
    v_reg = train_data['v'][idx:idx+1]
    
    # validation step
    val_data = validation_step(conf, model, valid_loader, u_reg, v_reg)
    
    
    # ------------------------------------------------------------------------
    # update history
    history['train_loss'].append(train_data['train_loss'])
    history['train_acc'].append(train_data['train_acc'])
    history['train_lip_loss'].append(train_data['train_lip_loss'])
    #
    history['val_loss'].append(val_data['val_loss'])
    history['val_acc'].append(val_data['val_acc'])
    history['val_lip_loss'].append(val_data['val_lip_loss'])
