import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer
from flip.load_data import load_MNIST_test, load_MNIST, split_loader
from flip.utils.config import cfg, dataset, model_attributes
from flip.test import attack_model, eval_acc, test_acc
import matplotlib.pyplot as plt
import time 

CFG = cfg(data=dataset(name='MNIST'), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10], # [3072, 128, 80, 10]
              act_fun = 'ReLU',
              file_name = 'model_compare_STA_v1720527201.pth', # change this line here to test your model (make sure that the model is defined correctly)
              )
          )

model = load_model.load(CFG)

# split data
dataloader, validation_loader, test_loader = split_loader(CFG, train_split=0.8)

pgd_accuracy = attack_model(model, test_loader, attack_kwargs = {'type':"pgd", 'epsilon': 0.6})
fgsm_accuracy = attack_model(model, test_loader, attack_kwargs = {'type':"fgsm", 'epsilon': 0.07})
test_accuracy = test_acc(model, test_loader)
print('PGD accuracy : ', pgd_accuracy)
print('FGSM accuracy : ', fgsm_accuracy)
print('Test accuracy : ', test_accuracy)