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
        return torch.stack((x, y), dim=1)

    def plot(self):
        x = torch.linspace(-1, 1, 100)
        y = self.createdata(x).cpu().detach()
        plt.plot(y)
        plt.show()

class Trainer:
    def __init__(self, model, train_loader, lip_reg_max, lamda=0.1, lr=0.1, adversarial_name="gradient_ascent", num_iters=1000):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.lr = lr
        self.reg_max = lip_reg_max
        self.num_iters = num_iters
        self.adversarial = lambda u: adversarial_update(self.model, u, u+torch.rand_like(u)*0.1, opt_kwargs={'name':adversarial_name, 'lr':self.lr})
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.lamda = lamda

    def train_step(self):
        # train phase
        self.model.train()
        opt = self.optimizer

        # initialize values for train accuracy and train loss
        train_acc = 0.0
        train_loss = 0.0
        tot_steps = 0
        train_lip_loss = 0.0

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
            adv = self.adversarial(torch.tensor([x.item()]).to(self.device))
            for _ in range(self.num_iters):
                adv.step()
            u, v = adv.u, adv.v
            # ---------------------------------------------------------------------

            # Compute the Lipschitz constant
            c_reg_loss = lip_constant_estimate(self.model)(u, v)
            # reset gradients
            opt.zero_grad()

            # evaluate model on batch
            logits = self.model(x.unsqueeze(0))

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
            train_acc += (logits.item() == y).sum().item()
            train_loss += c_loss.item()
            train_lip_loss = max(c_reg_loss.item(), train_lip_loss)
            tot_steps += 1

        return {'train_loss': train_loss, 'train_acc': train_acc / tot_steps,
                'train_lip_loss': train_lip_loss}

    def plot(self):
        x = torch.linspace(-1, 1, 100).to(device)
        y = self.model(x.t()).cpu().detach()
        plt.plot(y)
        plt.show()

if __name__ == "__main__":
    coeffs = [1., 2., 3.]
    polynom = Polynom(coeffs)
    polynom.plot()
    x = torch.linspace(-1, 1, 100).to(device)
    xy = polynom.createdata(x).to(device)
    model = fully_connected([1, 10, 10, 1], "ReLu")
    model = model.to(device)
    trainer = Trainer(model, xy, 10, lamda=0.1, lr=0.1, adversarial_name="SGD", num_iters=1000)
    trainer.plot()
    for i in range(100):
        trainer.train_step()
        if i % 10 == 0:
            print(i)
    trainer.plot()
    print(coeffs)
