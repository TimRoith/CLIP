import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update, lip_constant_estimate, Flatten
from train_regularizer import Trainer, Polynom, scattered_points
device ="cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.close('all')
coeffs = torch.tensor([0.5,0.01,-0.,0.,0.,0.,1]).to(device)
polynom = Polynom(coeffs, scaling=0.00005)
xmin,xmax=(-3,3)
plt.show()
x = scattered_points(num_pts=50, xmin=xmin, xmax=xmax, percent_loss=0.75, random=False).to(device)
xy_loader = polynom.createdata(x,sigma=0.0)[0]
XY = torch.stack([xy_loader.dataset.tensors[i] for i in [0,1]])

model = fully_connected([1, 50, 100, 50, 1], "ReLU")
model = model.to(device)
num_total_iters = 100

#lambda_max = 100000
#lamda = np.linspace(0, lambda_max, int(lambda_max*10))
lamda = np.logspace(-3, 5, 30)
num_trainer = 5
value_lip = []
value_loss = []
value_train_acc = []
value_train_mse_loss = []
max_lip = 0
max_loss = 0
max_mse = 0
for lam in lamda:
    trainers = [Trainer(model, xy_loader, 100, lamda=lam, lr=0.001, adversarial_name="SGD", num_iters=5) for _ in range(num_trainer)] #, backtracking=0.9)
    train_acc = 0
    train_loss = 0
    train_lip_loss = 0
    train_mse_loss = 0
    for i in range(num_total_iters):
        for j in range(num_trainer):
            trainers[j].train_step()
    for j in range(num_trainer):
        train_acc += trainers[j].train_acc/num_trainer
        train_loss += trainers[j].train_loss/num_trainer
        train_lip_loss += trainers[j].train_lip_loss/num_trainer
        train_mse_loss += trainers[j].saved_mse_loss/num_trainer
    print("lambda: ", lam)
    print("loss: ", train_loss)
    print("lip: ", train_lip_loss)
    print("train accuracy : ", train_acc)
    value_lip.append(train_lip_loss)
    value_loss.append(train_loss)
    value_train_acc.append(train_acc)
    value_train_mse_loss.append(train_mse_loss)
    max_lip = max(max_lip, train_lip_loss)
    max_loss = max(max_loss, train_loss)
    max_mse = max(max_mse, train_mse_loss)

plt.figure()

plt.plot(lamda, np.array(value_lip)/max_lip, label="Lip")
plt.plot(lamda, np.array(value_loss)/max_loss, label="Loss")
plt.plot(lamda, np.array(value_train_acc), label="Train Acc")
plt.plot(lamda, np.array(value_train_mse_loss)/max_mse, label="MSE Loss")
plt.title('Regularization')

plt.xlabel('Lambda (log scale)')
plt.ylabel('Normalized Value')

plt.xscale('log')

plt.legend()

plt.savefig('final_plot.png')
plt.show()