from matplotlib.animation import FuncAnimation
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
        x = x[None, :]
        d = torch.arange(self.degree + 1)[:, None].to(device)
        x = x ** d
        return torch.sum(self.coeffs[:, None] * x, dim=0)
    
    def createdata(self, x, sigma=0.1):
        y = self(x) + torch.randn_like(x) * sigma
        xy = torch.stack((x, y), dim=1)
        dataset = torch.utils.data.TensorDataset(xy[:, 0], xy[:, 1])
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        return loader, xy

    def plot(self, ax=None, num_pts=100, xmin=-1, xmax=1):
        ax = plt if ax is None else ax
        x = torch.linspace(xmin, xmax, num_pts).to(device)
        y = self.createdata(x, sigma=0.)[1].cpu().detach()[:, 1]
        ax.plot(x.cpu(), y)

class Trainer:
    def __init__(self, model, train_loader, lip_reg_max, lamda=0.1, lr=0.1, adversarial_name="gradient_ascent", num_iters=7, epsilon=1e-1):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.lr = lr
        self.reg_max = lip_reg_max
        self.num_iters = num_iters
        self.adversarial = lambda u: adversarial_update(self.model, u, u + torch.rand_like(u) * 0.1, opt_kwargs={'name': adversarial_name, 'lr': self.lr})
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lamda = lamda
        self.epsilon = epsilon
        self.train_acc = 0.0
        self.train_loss = 0.0
        self.tot_steps = 0
        self.train_lip_loss = 0.0

    def train_step(self):
        self.model.train()
        opt = self.optimizer
        self.train_acc = 0.0
        self.train_loss = 0.0
        self.tot_steps = 0
        self.train_lip_loss = 0.0
        u, v = None, None

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            adv = self.adversarial(x)
            for _ in range(self.num_iters):
                adv.step()
            u, v = adv.u, adv.v

            c_reg_loss = lip_constant_estimate(self.model, mean=True)(u, v)
            opt.zero_grad()

            logits = self.model(x).reshape(-1)

            c_loss = F.mse_loss(logits, y)
            c_loss = torch.sum((logits - y) ** 2)

            lamda = self.lamda
            if not (c_reg_loss.item() > self.reg_max):
                pass
            else:
                print('The Lipschitz constant was too big:', c_reg_loss.item(), ". No Lip Regularization for this batch!")

            c_loss.backward()
            opt.step()

            self.train_loss += c_loss.item()
            self.train_lip_loss = max(c_reg_loss.item(), self.train_lip_loss)
            for i in range(y.shape[0]):
                self.tot_steps += 1
                equal = torch.isclose(logits[i], y[i], atol=self.epsilon)
                self.train_acc += equal.item() * 1.0
        self.train_acc /= self.tot_steps
    
    def plot(self, ax=None, line=None, xmin=-1, xmax=1):
        x = torch.linspace(xmin, xmax, 100).to(device)
        y = self.model(x.t()).cpu().detach()
        x = x.cpu().detach()
        ax = plt if ax is None else ax
        
        if line is None:
            line = ax.plot(x, y)[0]
        else:
            line.set_ydata(y)
            
        return line

if __name__ == "__main__":
    plt.close('all')
    fig, ax = plt.subplots(1,)
    coeffs = torch.tensor([0, 0, 0.1, 0.1, 0.05]).to(device)
    polynom = Polynom(coeffs)
    xmin, xmax = (-2, 2)
    polynom.plot(ax=ax, xmin=xmin, xmax=xmax)
    x = torch.linspace(xmin, xmax, 200).to(device)
    xy_loader = polynom.createdata(x, sigma=0.1)[0]
    XY = torch.stack([xy_loader.dataset.tensors[i] for i in [0, 1]])
    
    model = fully_connected([1, 50, 50, 50, 50, 50, 1], "ReLU")
    model = model.to(device)

    trainer = Trainer(model, xy_loader, 10, lamda=0.1, lr=0.005, adversarial_name="SGD", num_iters=1000)
    line = trainer.plot(ax=ax, xmin=xmin, xmax=xmax)
    ax.scatter(XY[0, :].cpu(), XY[1, :].cpu())

    num_total_iters = 50

    def compute_interval(frame):
        initial_interval = 1000  # Start with 1000 ms
        final_interval = 100  # End with 100 ms
        alpha = frame / num_total_iters
        return initial_interval * (1 - alpha) + final_interval * alpha

    def update(frame):
        global line
        trainer.train_step()
        print(f"Epoch {frame+1}: Train Loss: {trainer.train_loss:.4f}, Train Accuracy: {trainer.train_acc:.4f}")
        ax.set_title('Iteration: ' + str(frame))
        line = trainer.plot(ax=ax, line=line, xmin=xmin, xmax=xmax)
        return line,

    ani = FuncAnimation(fig, update, frames=num_total_iters, blit=True, interval=compute_interval(0))

    ani.save('training_animation.mp4', writer='ffmpeg')

    plt.show()

