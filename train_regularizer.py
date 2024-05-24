import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update, lip_constant_estimate, Flatten
device ="cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Polynom:
    def __init__(self, coeffs, scaling=1.):
        self.coeffs = coeffs
        self.degree = len(coeffs) - 1
        self.scaling = scaling

    def __call__(self, x):
        x = x[None, :]
        d = torch.arange(self.degree+1)[:,None].to(device)
        x = x**d
        return self.scaling * torch.sum(self.coeffs[:,None] * x, dim=0)
    
    def createdata(self, x, sigma=0.1):
        y = self(x) + torch.randn_like(x) * sigma
        xy = torch.stack((x, y), dim=1)
        dataset = torch.utils.data.TensorDataset(xy[:, 0], xy[:, 1])
        loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
        return loader, xy

    def plot(self, ax=None, num_pts=100, xmin=-1, xmax=1):
        ax = plt if ax is None else ax
        x = torch.linspace(xmin, xmax, num_pts).to(device)
        y = self.createdata(x, sigma=0.)[1].cpu().detach()[:, 1]
        ax.plot(x.cpu(), y)

class Trainer:
    def __init__(self, model, train_loader, lip_reg_max, lamda=0.1, lr=0.1, adversarial_name="gradient_ascent", num_iters=1, epsilon=1e-1, backtracking=None):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.lr = lr
        self.backtracking = backtracking
        self.reg_max = lip_reg_max
        self.num_iters = num_iters
        self.adversarial = lambda u: adversarial_update(self.model, u, u+torch.rand_like(u)*0.1, opt_kwargs={'name':adversarial_name, 'lr':self.lr})
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lamda = lamda
        self.epsilon = epsilon
        self.train_acc = 0.0
        self.train_loss = 0.0
        self.tot_steps = 0
        self.train_lip_loss = 0.0

    def set_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def train_step(self):
        # train phase
        self.model.train()
        opt = self.optimizer
        #scheduler = ReduceLROnPlateau(opt, 'min')
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
            ux = torch.linspace(-3, 3, 40).to(device)
            adv = self.adversarial(ux)
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
            c_loss = torch.sum((logits-y)**2)

            # Change regularization parameter lamda
            lamda = self.lamda
            # check if Lipschitz term is too large. Note that regularization with
            # large Lipschitz terms yields instabilities!
            if not (c_reg_loss.item() > self.reg_max):
                c_loss = c_loss + lamda * c_reg_loss
                #pass
            else:
                print('The Lipschitz constant was too big:', c_reg_loss.item(), ". No Lip Regularization for this batch!")

            # Backtracking line search
            if self.backtracking is not None:
                initial_lr = self.lr
                beta = 0.5
                tau = self.backtracking

                # Compute the objective function with the current parameters
                f_up = c_loss.item()
                success = False

                while True:
                    # Backup current model parameters
                    backup_params = [p.clone() for p in self.model.parameters()]

                    # Perform a gradient descent step with the reduced learning rate
                    opt.step()

                    # Compute the new objective function
                    with torch.no_grad():
                        new_logits = self.model(x)
                        new_logits = new_logits.reshape(new_logits.shape[0])
                        new_c_loss = F.mse_loss(new_logits, y)
                        new_c_reg_loss = lip_constant_estimate(self.model, mean=True)(u, v)
                        if not (new_c_reg_loss.item() > self.reg_max):
                            new_c_loss = new_c_loss + lamda * new_c_reg_loss

                    f_down = new_c_loss.item()

                    if f_down <= f_up - tau * initial_lr:
                        success = True
                        break
                    else:
                        # Restore model parameters and reduce the learning rate
                        for original, backup in zip(self.model.parameters(), backup_params):
                            original.data.copy_(backup.data)
                        initial_lr *= beta
                        self.set_learning_rate(opt, initial_lr)

                if not success:
                    print("Backtracking line search failed to reduce the objective function.")

            # Update model parameters
            c_loss.backward()
            opt.step()
            #scheduler.step(c_loss)
            
            # update accuracy and loss
            self.train_loss += c_loss.item()
            self.train_lip_loss = max(c_reg_loss.item(), self.train_lip_loss)
            for i in range(y.shape[0]):
                self.tot_steps += 1
                equal = torch.isclose(logits[i], y[i], atol=self.epsilon)
                self.train_acc += equal.item()*1.0
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

def scattered_points(num_pts=100, xmin=-1, xmax=1, percent_loss=0.3, random=True):
    if random:
        if percent_loss > 0.5:
            raise ValueError("percent_loss should be less than 0.5 for random loss of data")
        xloss_down = xmin + (xmax - xmin) *percent_loss
        xloss_up = xmax - (xmax - xmin) *percent_loss
        #return one random point between xloss_down and xloss_up
        xloss = torch.rand(1).item()*(xloss_up - xloss_down) + xloss_down
        x_down = torch.linspace(xmin, xloss, num_pts//2)
        x_up = torch.linspace(xloss + (xmax - xmin) *percent_loss, xmax, num_pts//2)
        x = torch.cat([x_down, x_up])
    else:
        if percent_loss > 0.99:
            raise ValueError("percent_loss should be less than 0.99 for fixed loss of data")
        x_mid = (xmin + xmax) / 2
        x_down = torch.linspace(xmin, x_mid - (x_mid - xmin) * percent_loss, num_pts//2)
        x_up = torch.linspace(x_mid + (xmax - x_mid) * percent_loss, xmax, num_pts//2)
        x = torch.cat([x_down, x_up])
    return x

if __name__ == "__main__":
    plt.close('all')
    fig, ax = plt.subplots(1,)
    coeffs = torch.tensor([0.5,0.01,-0.,0.,0.,0.,1]).to(device)
    polynom = Polynom(coeffs, scaling=0.00005)
    xmin,xmax=(-3,3)
    polynom.plot(xmin=xmin,xmax=xmax)
    x = scattered_points(num_pts=50, xmin=xmin, xmax=xmax, percent_loss=0.75, random=False).to(device)
    xy_loader = polynom.createdata(x,sigma=0.0)[0]
    XY = torch.stack([xy_loader.dataset.tensors[i] for i in [0,1]])
    
    ax.set_ylim(XY[1, :].min()-0.01, XY[1, :].max()+0.01)
    
    model = fully_connected([1, 50, 100, 50, 1], "ReLU")
    model = model.to(device)

    ax.plot(x.cpu(),model(x).cpu().detach())
    trainer = Trainer(model, xy_loader, 100, lamda=10.1, lr=0.001, adversarial_name="SGD", num_iters=10, backtracking=0.9)
    line = trainer.plot(ax=ax, xmin=xmin,xmax=xmax)
    #plt.show()
    num_total_iters = 300
    ax.scatter(XY[0,:].cpu(),XY[1,:].cpu())
    for i in range(num_total_iters):
        trainer.train_step()
        if i % 1 == 0:
            print(i)
            print(model.layers[1].weight)
            #ax.set_title('Iteration: ' + str(i))
            print("train accuracy : ", trainer.train_acc)
            print("train loss : ", trainer.train_loss)
            print("train lip loss : ", trainer.train_lip_loss)
            #polynom.plot(ax=ax)
            trainer.plot(ax=ax, line=line, xmin=xmin,xmax=xmax)
            plt.pause(0.1)

    ax.set_title('Iteration: ' + str(num_total_iters))
    polynom.plot(ax=ax, xmin=xmin,xmax=xmax)
    trainer.plot(ax=ax, xmin=xmin,xmax=xmax)
    ax.legend(["Sample","True polynom", "Fully Connected Model"])
    #fig.savefig('final_plot.png')
    #plt.show()
