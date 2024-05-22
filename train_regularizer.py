import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update, lip_constant_estimate, Flatten
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Polynom:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.degree = len(coeffs) - 1

    def __call__(self, x):
        s=0
        for i in range(self.degree+1):
            s+=self.coeffs[i] * x**i
        return s

    def createdata(self, x):
        y = self(x) + torch.randn_like(x) * 0.1
        xy = torch.stack((x, y), dim=1)
        dataset = torch.utils.data.TensorDataset(xy[:, 0], xy[:, 1])
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        return loader, xy

    def plot(self, ax=None):
        x = torch.linspace(-1, 1, 100)
        y = self.createdata(x)[1].cpu().detach()[:, 1]
        if ax is None:
            plt.plot(y)
        else:
            ax.plot(y)

class Trainer:
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
        self.tot_steps = 0
        self.train_lip_loss = 0.0

        #
        u = None
        v = None

        # -------------------------------------------------------------------------
        # loop over all batches
        for batch_idx, (x, y) in enumerate(self.train_loader):
            # get batch data
            x, y = x.to(self.device), y.to(self.device)
            # ---------------------------------------------------------------------
            # Adversarial update
            # ---------------------------------------------------------------------

            # adversarial update on the Lipschitz set
            adv = self.adversarial(x)
            for _ in range(self.num_iters):
                adv.step()
            u, v = adv.u, adv.v
            # ---------------------------------------------------------------------

            # Compute the Lipschitz constant
            c_reg_loss = lip_constant_estimate(self.model, mean = True)(u, v)
            # reset gradients
            opt.zero_grad()

            # evaluate model on batch
            logits = self.model(x)
            logits = logits.reshape(logits.shape[0])

            # Get classification loss
            c_loss = F.mse_loss(logits, y)

            # Change regularization parameter lamda
            lamda = self.lamda
            # check if Lipschitz term is too large. Note that regularization with
            # large Lipschitz terms yields instabilities!
            if not (c_reg_loss.item() > self.reg_max):
                c_loss = c_loss + lamda * c_reg_loss
            else:
                print('The Lipschitz constant was too big:', c_reg_loss.item(), ". No Lip Regularization for this batch!")

            # Update model parameters
            c_loss.backward()
            opt.step()

            # update accuracy and loss
            self.train_loss += c_loss.item()
            self.train_lip_loss = max(c_reg_loss.item(), self.train_lip_loss)
            for i in range(y.shape[0]):
                self.tot_steps += 1
                equal = torch.isclose(logits[i], y[i], atol=self.epsilon)
                self.train_acc += equal.item()*1.0
        self.train_acc /= self.tot_steps
    def plot(self, ax=None):
        x = torch.linspace(-1, 1, 100).to(device)
        y = self.model(x.t()).cpu().detach()
        if ax is None:
            plt.plot(y)
        else:
            ax.plot(y)

if __name__ == "__main__":
    plt.close('all')
    fig, ax = plt.subplots(1,)
    coeffs = [1., 2., 3.]
    polynom = Polynom(coeffs)
    polynom.plot()
    x = torch.linspace(-1, 1, 100).to(device)
    xy_loader = polynom.createdata(x)[0]
    model = fully_connected([1, 10, 10, 1], "ReLu")
    model = model.to(device)
    ax.plot(x.cpu(),model(x).cpu().detach())
    trainer = Trainer(model, xy_loader, 10, lamda=0.1, lr=0.1, adversarial_name="SGD", num_iters=1000)
    trainer.plot()
    plt.show()
    num_total_iters = 5
    for i in range(num_total_iters):
        trainer.train_step()
        if i % 1 == 0:
            print(i)
            ax.set_title('Iteration: ' + str(i))
            print("train accuracy : ", trainer.train_acc)
            print("train loss : ", trainer.train_loss)
            print("train lip loss : ", trainer.train_lip_loss)
            polynom.plot(ax=ax)
            trainer.plot(ax=ax)
            plt.pause(0.1)
    ax.set_title('Iteration: ' + str(num_total_iters))
    polynom.plot(ax=ax)
    trainer.plot(ax=ax)
    ax.legend(["True", "Model"])
    plt.show()
