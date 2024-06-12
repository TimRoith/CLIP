import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update, lip_constant_estimate, Flatten
from train_regularizer import Trainer, Polynom, scattered_points
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.close('all')
coeffs = torch.tensor([0.5, 0.01, -0., 0., 0., 0., 1]).to(device)
polynom = Polynom(coeffs, scaling=0.00005)
xmin, xmax = (-3, 3)
x = scattered_points(num_pts=50, xmin=xmin, xmax=xmax, percent_loss=0.75, random=False).to(device)
xy_loader, xy = polynom.createdata(x, sigma=0.0)
XY = torch.stack([xy_loader.dataset.tensors[i] for i in [0, 1]])

xy_mean = lambda x: [torch.sum(xy[:, 1]) / len(xy[:, 1]) for _ in x]

model = fully_connected([1, 50, 100, 50, 1], "ReLU")
model = model.to(device)
num_total_iters = 150

# Change: vary learning rate instead of lambda
num_total_lr = 25
lr_values = np.logspace(-5, -1, num_total_lr)
num_trainer = 20

x_test = torch.linspace(xmin, xmax, 100).unsqueeze(1).to(device)
results = []

value_lip = []
value_loss = []
value_train_acc = []
value_train_mse_loss = []
value_time = []
max_lip = 0
max_loss = 0
max_mse = 0
max_time = 0
lam = 0.01  # Fixed lambda value
for lr in lr_values:
    model = [fully_connected([1, 50, 100, 50, 1], "ReLU") for _ in range(num_trainer)]
    trainers = [Trainer(model[i], xy_loader, 100, lamda=lam, lr=lr, adversarial_name="SGD", num_iters=7, epsilon=5e-3, min_accuracy=None, CLIP_estimation="sum", iter_warm_up=num_total_iters//4) for i in range(num_trainer)]
    y_mean = None
    train_acc = 0
    train_loss = 0
    train_lip_loss = 0
    train_mse_loss = 0
    for i in range(num_total_iters):
        time_start = time.time()
        for j in range(num_trainer):
            trainers[j].train_step()
        time_total = time.time() - time_start
    time_mean = time_total / num_trainer
    for j in range(num_trainer):
        train_acc += trainers[j].train_acc / num_trainer
        train_loss += trainers[j].train_loss / num_trainer
        train_lip_loss += trainers[j].train_lip_loss / num_trainer
        train_mse_loss += trainers[j].saved_basic_loss / num_trainer
        if y_mean is None:
            y_mean = trainers[j].model(x_test) * (1 / num_trainer)
        else:
            y_mean += trainers[j].model(x_test) * (1 / num_trainer)
    results.append(y_mean)
    print("Learning rate: ", lr)
    print("Loss: ", train_loss)
    print("Lip: ", train_lip_loss)
    print("Train accuracy: ", train_acc)
    value_lip.append(train_lip_loss)
    value_loss.append(train_loss)
    value_train_acc.append(train_acc)
    value_train_mse_loss.append(train_mse_loss)
    value_time.append(time_mean)
    max_lip = max(max_lip, train_lip_loss)
    max_loss = max(max_loss, train_loss)
    max_mse = max(max_mse, train_mse_loss)
    max_time = max(max_time, time_mean)

plt.figure()

plt.plot(lr_values, np.array(value_lip) / max_lip, label="Lip")
plt.plot(lr_values, np.array(value_loss) / max_loss, label="Loss")
plt.plot(lr_values, np.array(value_train_acc), label="Train Acc")
plt.plot(lr_values, np.array(value_train_mse_loss) / max_mse, label="MSE Loss")
plt.plot(lr_values, np.array(value_time) / max_time, label="Computing Time")
plt.title('Regularization')

plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Normalized Value')

plt.xscale('log')

plt.legend()

plt.savefig('final_plot.png')
plt.show()

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
polynom.plot(xmin=xmin, xmax=xmax, ax=ax)
ax.scatter(XY[0, :].cpu(), XY[1, :].cpu())
ax.plot(x_test, xy_mean(x_test), 'r--')

def compute_interval(frame):
    initial_interval = 1000  # Start with 1000 ms
    final_interval = 100  # End with 100 ms
    alpha = frame / num_total_lr
    return initial_interval * (1 - alpha) + final_interval * alpha

line, = ax.plot(x_test, results[0].detach().cpu().numpy())

def update(frame):
    ax.set_title('Learning rate: ' + str(lr_values[frame]))
    line.set_data(x_test, results[frame].detach().cpu().numpy())
    return line,

ani = FuncAnimation(fig, update, frames=num_total_lr, blit=True, interval=compute_interval(0))
ax.legend(["True polynom", "Data Value", "Data Mean", "Fully Connected Model"])
ani.save('lr_model_evolution.mp4', writer='ffmpeg')
plt.show()
print("Max Time: ", max_time)