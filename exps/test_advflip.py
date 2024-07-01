import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer, AdvFlipTrainer
from flip.load_data import load_MNIST_test, load_MNIST, split_loader
from flip.utils.config import cfg, dataset, model_attributes
from flip.test import attack_model, eval_acc
import matplotlib.pyplot as plt
import time 



CFG = cfg(data=dataset(),model = model_attributes(name = 'FC',sizes=[784, 200, 80, 10],act_fun = 'ReLU',file_name ='model_lipschitz_v1719581713.pth',))

dataloader, validation_loader, test_loader = split_loader(CFG, train_split=0.8)

model = load_model.load(CFG)

accuracy = attack_model(model, test_loader, attack_kwargs = {'type':"pgd", 'epsilon': 0.1})

print('advflip : ', accuracy)

CFG.model.file_name = 'model_lipschitz_adv_v1719581713.pth'
model_adv = load_model.load(CFG)

accuracy = attack_model(model_adv, test_loader, attack_kwargs = {'type':"pgd", 'epsilon': 0.1})

print('adv : ', accuracy)

CFG.model.file_name = 'model_compare_sum_v1719580187.pth'
model_sum = load_model.load(CFG)

accuracy = attack_model(model_sum, test_loader, attack_kwargs = {'type':"pgd", 'epsilon': 0.1})

print('sum :', accuracy)

accuracy = attack_model(model_sum, test_loader, attack_kwargs = {'type':"fgsm", 'epsilon': 0.05})

print('sum vs fgsm : ', accuracy)

CFG.model.file_name = 'model_compare_max_v1719581300.pth'
model_max = load_model.load(CFG)

accuracy = attack_model(model_max, test_loader, attack_kwargs = {'type':"pgd", 'epsilon': 0.1})

print('max : ', accuracy)

CFG.model.file_name = 'model_lipschitz_adv_v1719828279.pth'
model_adv = load_model.load(CFG)

accuracy = attack_model(model_adv, test_loader, attack_kwargs = {'type':"fgsm", 'epsilon': 0.05})

print('adv_pgd vs fgsm : ', accuracy)

CFG.model.file_name = 'model_lipschitz_adv_v1719828279.pth'
model_adv = load_model.load(CFG)

accuracy = attack_model(model_adv, test_loader, attack_kwargs = {'type':"pgd", 'epsilon': 0.1})

print('adv_pgd : ', accuracy)