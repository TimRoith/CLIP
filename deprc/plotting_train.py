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
x = scattered_points(num_pts=50, xmin=xmin, xmax=xmax, percent_loss=0.75, random=False).to(device)
xy_loader, xy = polynom.createdata(x,sigma=0.0)
XY = torch.stack([xy_loader.dataset.tensors[i] for i in [0,1]])

xy_mean =lambda x : [torch.sum(xy[:,1])/len(xy[:,1]) for _ in x]


num_total_iters = 100

#lambda_max = 100000
#lamda = np.linspace(0, lambda_max, int(lambda_max*10))
num_total_lamda = 50
lamda = np.logspace(-5, 4, num_total_lamda)
num_trainer = 10

x_test = torch.linspace(xmin, xmax, 100).unsqueeze(1).to(device)
results = []

value_lip = []
value_loss = []
value_train_acc = []
value_train_mse_loss = []
max_lip = 0
max_loss = 0
max_mse = 0
for lam in lamda:
    model = [fully_connected([1, 50, 100, 50, 1], "ReLU") for _ in range(num_trainer)]
    trainers = [Trainer(model[i], xy_loader, 100, lamda=lam, lr=0.001, adversarial_name="SGD", num_iters=5, min_accuracy=0.9, CLIP_estimation="sum", iter_warm_up=20, change_lamda_in=False, epsilon=5e-3) for i in range(num_trainer)] #, backtracking=0.9)
    y_mean = None
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
        train_mse_loss += trainers[j].saved_basic_loss/num_trainer
        if y_mean is None:
            y_mean = trainers[j].model(x_test)*(1/num_trainer)
        else : 
            y_mean += trainers[j].model(x_test)*(1/num_trainer)
    results.append(y_mean)
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

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
polynom.plot(xmin=xmin,xmax=xmax, ax=ax)
ax.scatter(XY[0,:].cpu(),XY[1,:].cpu())
ax.plot(x_test, xy_mean(x_test), 'r--')

def compute_interval(frame):
        initial_interval = 1000  # Start with 1000 ms
        final_interval = 100  # End with 100 ms
        alpha = frame / num_total_lamda
        return initial_interval * (1 - alpha) + final_interval * alpha

line, = ax.plot(x_test, results[0].detach().cpu().numpy())

def update(frame):
    ax.set_title('Lamda: ' + str(lamda[frame]))
    line.set_data(x_test, results[frame].detach().cpu().numpy())
    #line = ax.plot(x_test, results[frame].detach().cpu().numpy())
    return line,

ani = FuncAnimation(fig, update, frames=num_total_lamda, blit=True, interval=compute_interval(0))
ax.legend(["True polynom","Data Value", "Data Mean", "Fully Connected Model"])
ani.save('lamda_model_evolution.mp4', writer='ffmpeg')
plt.show()