import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update, lip_constant_estimate, Flatten
from train_regularizer import Polynom
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrainer:
    def __init__(self, model, train_loader, lip_reg_max, lamda=0.1, lr=0.1, adversarial_name="gradient_ascent", num_iters=7, epsilon=1e-1):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.lr = lr
        self.reg_max = lip_reg_max
        self.num_iters = num_iters
        self.adversarial = lambda u: adversarial_update(self.model, u, u+torch.rand_like(u)*0.1, opt_kwargs={'name':adversarial_name, 'lr':self.lr})
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.lamda = lamda
        self.epsilon = epsilon
        self.train_acc = 0.0
        self.train_loss = 0.0
        self.tot_steps = 0
        self.train_lip_loss = 0.0

    def train_step(self):
        # train phase
        self.model.train()
        opt = self.optimizer

        # initialize values for train accuracy and train loss
        self.train_acc = 0.0
        self.train_loss = 0.0
        self.train_lip_loss = 0.0
        self.tot_steps = 0

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            # zero the parameter gradients
            opt.zero_grad()

            # forward pass
            y_pred = self.model(x)
            loss = F.mse_loss(y_pred, y)
            # backward pass
            loss.backward()
            # update parameters
            opt.step()
            # calculate train accuracy and loss
            _, predicted = torch.max(y_pred.data, 1)
            self.train_acc += (predicted == y).sum().item()
            self.train_loss += loss.item()
            self.tot_steps += 1
        # calculate average train accuracy and loss
        self.train_acc /= self.tot_steps
        self.train_loss /= self.tot_steps

# Fill in the required variables
input_size = 1
hidden_size = 20
output_size = 1
batch_size = 32
lip_reg_max = 0.1
lamda = 0.1
lr = 0.01
adversarial_name = "gradient_ascent"
num_iters = 7
epsilon = 1e-1
num_epochs = 100

# Create a fully connected model
model = fully_connected([input_size, hidden_size, output_size], "ReLU")

# Create a train loader using data from Polynom
polynom = Polynom([1.,2.,3.])
x = torch.linspace(-1, 1, 100).to(device)
train_loader = polynom.createdata(x)[0]

# Create an instance of BasicTrainer
trainer = BasicTrainer(model, train_loader, lip_reg_max, lamda, lr, adversarial_name, num_iters, epsilon)

# Train the model
train_losses = []
train_accuracies = []
for epoch in range(num_epochs):
    trainer.train_step()
    train_losses.append(trainer.train_loss)
    train_accuracies.append(trainer.train_acc)
    print(f"Epoch {epoch+1}: Train Loss: {trainer.train_loss:.4f}, Train Accuracy: {trainer.train_acc:.4f}")

# Plot the training loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()
# Plot the polynom
polynom.plot()
plt.show()
# Plot the model
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x.cpu().numpy(), model(x).detach().cpu().numpy(), label='Model')
plt.plot(x.cpu().numpy(), polynom(x.cpu().numpy()), label='Polynom')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model vs Polynom')
plt.legend()
plt.tight_layout()
plt.show()