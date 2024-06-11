import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import adversarial_attacks as at
import time
import os

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update, lip_constant_estimate, Flatten
device ="cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Polynom:
    def __init__(self, coeffs, scaling=1.):
        self.coeffs = coeffs
        self.degree = len(coeffs) - 1 #if len(coeffs.shape) == 1 else coeffs.shape[0] - 1
        self.scaling = scaling

    def __call__(self, x):
        if len(x.shape) <= 1:
            x = x[None, :]
            d = torch.arange(self.degree+1)[:,None].to(device)
            x = x**d
            return self.scaling * torch.sum(self.coeffs[:,None] * x, dim=0)
        else :
            for i in range(x.shape[1]):
                x_i = x[:, i]
                x_i = x_i[None, :]
                d = torch.arange(self.degree+1)[:,None].to(device)
                x_i = x_i**d
                if i == 0:
                    y = self.scaling * torch.sum(self.coeffs[:,None] * x_i, dim=0)
                else:
                    y += self.scaling * torch.sum(self.coeffs[:,None] * x_i, dim=0)
            return y
    
    def createdata(self, x, sigma=0.1):
        if len(x.shape) <= 1:
            y_true = self(x)
            y = y_true + torch.randn_like(y_true) * sigma
            xy = torch.stack((x, y), dim=1)
            dataset = torch.utils.data.TensorDataset(xy[:, 0], xy[:, 1])
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(x), shuffle=True)
        else :
            n = x.shape[1]
            y_true = self(x)
            y = y_true + torch.randn_like(y_true) * sigma
            xy = torch.cat((x, y[:,None]), dim=1)
            dataset = torch.utils.data.TensorDataset(xy[:, :n], xy[:, n])
            loader = torch.utils.data.DataLoader(dataset, batch_size=x.shape[0], shuffle=True)
        return loader, xy

    def plot(self, dim = None, ax=None, num_pts=100, xmin=-1, xmax=1, projection_dim=1):
        if dim is None:
            dim = 1
            print("dim is None, assuming 1D")
        ax = plt if ax is None else ax
        if dim == 1:
            x = torch.linspace(xmin, xmax, num_pts).to(device)
            y = self.createdata(x, sigma=0.)[1].cpu().detach()[:, 1]
        else :
            x, x_true = linear_points(num_pts=num_pts, xmin=xmin, xmax=xmax, dim=dim, coordinate=projection_dim)
            y = self.createdata(x_true, sigma=0.)[1].cpu().detach()[:, dim]
        ax.plot(x.cpu(), y)

class All_MNIST:
    def __init__(self, data_file, download=False, data_set = "MNIST", batch_size = 100, train_split = 0.9, num_workers=1):
        self.data_set = data_set
        self.data_file = data_file
        self.download = download
        self.im_shape = None
        self.data_set_mean = None
        self.data_set_std = None
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def __call__(self, test_size=1):
        return self.get_data_set(test_size=test_size)
    
    def get_data_set(self, test_size=1):
        train, valid, test, train_loader, valid_loader, test_loader = [None] * 6
        if self.data_set == "MNIST":
            self.im_shape = [1,28,28]
            
            # set mean and std for this dataset
            self.data_set_mean = 0.1307
            self.data_set_std = 0.3081
            train, test = self.get_mnist()
        elif self.data_set == "Fashion-MNIST":
            self.data_set_mean = 0.5
            self.data_set_std = 0.5
            
            self.im_shape = [1,28,28]
            train, test = self.get_fashion_mnist()
        else:
            raise ValueError("Dataset:" + self.data_set + " not defined")
        train_loader, valid_loader, test_loader = self.train_valid_test_split(train, test, test_size=test_size)

        return train_loader, valid_loader, test_loader

    def get_mnist(self):
        transform = self.get_transform(1, 28, True)
        #
        train = datasets.MNIST(self.data_file, train=True, download=self.download, transform=transform)
        test = datasets.MNIST(self.data_file, train=False, download=self.download, transform=transform)
        return train, test

    def get_fashion_mnist(self):
        transform_train = self.get_transform("FashionMNIST", 1, 28, True)
        transforms_test = self.get_transform("FashionMNIST", 1, 28, False)

        train = datasets.FashionMNIST(self.data_file, train=True, download=self.download,transform=transform_train)
        test = datasets.FashionMNIST(self.data_file, train=False, download=self.download, transform=transforms_test)
        return train, test


    def get_transform(self, channels, size, train):
        t = []
        t.append(transforms.ToTensor())
        # compose the transform
        transform = transforms.Compose(t)
        return transform

    def train_valid_test_split(self, train, test, test_size=1):
        batch_size = self.batch_size
        train_split = self.train_split
        num_workers = self.num_workers
        total_count = len(train)
        train_count = int(train_split * total_count)
        val_count = total_count - train_count
        train, val = torch.utils.data.random_split(train, [train_count, val_count],generator=torch.Generator().manual_seed(42))

        if test_size != 1:
            test_count = int(len(test) * test_size)
            _count = len(test) - test_count
            test, _ = torch.utils.data.random_split(test, [test_count, _count])

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(val, batch_size=1000, shuffle=True, pin_memory=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

        return train_loader, valid_loader, test_loader

class Trainer:
    def __init__(self, model, train_loader, lip_reg_max,loss = F.mse_loss , lamda=0.1, percent_of_lamda=0.1, min_accuracy=None, lr=0.1, adversarial_name="SGD", num_iters=1, epsilon=1e-2, backtracking=None, in_norm=None, out_norm=None, CLIP_estimation = "sum", iter_warm_up=None, lamda_stuck = None, change_lamda_in = True):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.lr = lr
        self.backtracking = backtracking
        self.reg_max = lip_reg_max
        self.num_iters = num_iters
        self.loss = loss
        self.lipschitz = lambda u, v: lip_constant_estimate(self.model, estimation = CLIP_estimation)(u, v)
        self.adversarial = lambda u: adversarial_update(self.model, u, u+torch.rand_like(u)*0.1, opt_kwargs={'name':adversarial_name, 'lr':self.lr}, in_norm = in_norm, out_norm = out_norm)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lamda = lamda
        self.lamda_stuck = lamda_stuck
        self.maximal_lamda = lamda*4
        self.minimal_lamda = lamda*0.25
        self.dlamda = lamda*percent_of_lamda
        # self.saved_min_acc = min_accuracy
        self.min_acc = min_accuracy
        self.change_lamda_in = change_lamda_in
        self.epsilon = epsilon
        self.train_acc = 0.0
        self.train_loss = 0.0
        self.tot_steps = 0
        self.train_lip_loss = 0.0
        self.saved_basic_loss = 0.0
        self.random_lip_constant = 0.0
        self.warm_up = True
        if iter_warm_up is not None:
            self.iter_warm_up = iter_warm_up
        else:
            print("No warm up iteration defined, using default value of 10.")
            self.iter_warm_up = 10

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
            #ux = torch.linspace(-3, 3, 40).to(device)
            if not self.warm_up:
                if sum(x.shape)-x.shape[0] > 1:
                    t_weight = torch.logspace(-3, 3, x.shape[0]).to(device)
                    ux = x[:, None]
                    ux.requires_grad = True
                    vx = x + torch.rand_like(x)*1000
                    lip_ux = self.lipschitz(ux, vx)
                    grad_lip_ux = torch.autograd.grad(lip_ux, ux, create_graph=False)[0]
                    max_grad = 0
                    index = 0
                    for i in range(x.shape[0]):
                        if torch.norm(grad_lip_ux[i]) > max_grad:
                            max_grad = torch.norm(grad_lip_ux[i])
                            index = i
                    best_ux_value = ux[index]
                    ux = best_ux_value
                    ux_grad = grad_lip_ux[index]*t_weight[0]
                    #print("x shape", x.shape)
                    for i in range(x.shape[0]-1):
                        ux = torch.cat([ux, best_ux_value], dim=0)
                        ux_grad = torch.cat([ux_grad, grad_lip_ux[index]*t_weight[i+1]], dim=0)
                    #print("grad shape", t_weight[:,None].shape)
                    ux = ux + ux_grad
                else:
                    ux = torch.linspace(-3, 3, x.shape[0]).to(device)
                #ux = x[:, None]
                #print("x shape", x.shape)
                adv = self.adversarial(ux)
                for _ in range(self.num_iters):
                    adv.step()
                u, v = adv.u, adv.v
                # print("u or v are nan", torch.isnan(u).any() or torch.isnan(v).any())
                # ---------------------------------------------------------------------

                # Compute the Lipschitz constant
                c_reg_loss = self.lipschitz(u, v)
            
            # reset gradients
            opt.zero_grad()

            # evaluate model on batch
            logits = self.model(x)
            #print("logits are nan", torch.isnan(logits).any())
            if torch.isnan(logits).any() and torch.isnan(x).any():
                print("logits and x are nan")
            if self.loss == F.mse_loss:
                logits = logits.reshape(logits.shape[0])

            # Get classification loss
            c_loss = self.loss(logits, y)
            #c_loss = torch.sum((logits-y)**2)
            self.saved_basic_loss = c_loss.detach().item()
            # Use regularization parameter lamda
            lamda = self.lamda
            # check if Lipschitz term is too large. Note that regularization with
            # large Lipschitz terms yields instabilities!
            if not self.warm_up:
                if not (c_reg_loss.item() > self.reg_max):
                    c_loss = c_loss + lamda * c_reg_loss
                    pass
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
                        if self.loss == F.mse_loss:
                            new_logits = new_logits.reshape(new_logits.shape[0])
                        new_c_loss = self.loss(new_logits, y)
                        if not self.warm_up:
                            new_c_reg_loss = self.lipschitz(u, v)
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
            if not self.warm_up:
                self.train_lip_loss = max(c_reg_loss.item(), self.train_lip_loss)
            if self.loss == F.mse_loss:
                for i in range(y.shape[0]):
                    self.tot_steps += 1
                    equal = torch.isclose(logits[i], y[i], atol=self.epsilon)
                    self.train_acc += equal.item()*1.0
            else:
                self.train_acc += (logits.max(1)[1] == y).sum().item()
                self.tot_steps += y.shape[0]
            if self.min_acc is not None and self.change_lamda_in: 
                if self.train_acc > self.min_acc*self.tot_steps:
                    self.warm_up = False
                    self.lamda = min(self.lamda + self.dlamda, self.maximal_lamda)
                    self.dlamda = self.dlamda*0.99
                    # weight = 0.6
                    # self.min_acc = (self.train_acc*weight + self.saved_min_acc*(2-weight))/2
                elif not self.warm_up:
                     self.lamda = max(self.lamda - self.dlamda, self.minimal_lamda)
                     self.dlamda = self.dlamda*0.99
                    #  if self.lamda == self.minimal_lamda:
                    #      self.min_acc = self.saved_min_acc
            elif self.iter_warm_up == 0:
                self.warm_up = False
                if self.min_acc is None:
                    self.min_acc = self.train_acc
                # self.saved_min_acc = (self.train_acc+0.5)/2
            else:
                self.iter_warm_up -= 1
            self.lamda = self.lamda_stuck if self.lamda_stuck is not None else self.lamda
        u_rand = torch.rand_like(x)*torch.rand(1).item()*10
        v_rand = torch.rand_like(x)*torch.rand(1).item()*10
        self.random_lip_constant = self.lipschitz(u_rand, v_rand).item()
        self.train_acc /= self.tot_steps
        if not self.change_lamda_in and self.train_acc > self.min_acc:
            self.lamda = min(self.lamda + self.dlamda, self.maximal_lamda)
            self.dlamda = self.dlamda*0.95
        elif not self.change_lamda_in :
            self.lamda = max(self.lamda - self.dlamda, self.minimal_lamda)
            self.dlamda = self.dlamda*0.95
    
    def adversarial_trainer_step(self):
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
            # adversarial update
            x_adv = x.clone().detach().requires_grad_(True)
            if not self.warm_up:
                for _ in range(self.num_iters):
                    logits = self.model(x_adv)
                    c_loss = self.loss(logits, y)
                    c_loss.backward()
                    x_adv = x_adv + self.lr * x_adv.grad.sign()
                    x_adv = torch.clamp(x_adv, min=0, max=1)
                    x_adv = x_adv.detach().requires_grad_(True)
            
            # reset gradients
            opt.zero_grad()

            # evaluate model on batch
            logits = self.model(x_adv)
            #print("logits are nan", torch.isnan(logits).any())
            if torch.isnan(logits).any() and torch.isnan(x_adv).any():
                print("logits and x_adv are nan")
            if self.loss == F.mse_loss:
                logits = logits.reshape(logits.shape[0])

            # Get classification loss
            c_loss = self.loss(logits, y)
            #c_loss = torch.sum((logits-y)**2)
            self.saved_basic_loss = c_loss.detach().item()
            # Use regularization parameter lamda
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
                        new_logits = self.model(x_adv)
                        if self.loss == F.mse_loss:
                            new_logits = new_logits.reshape(new_logits.shape[0])
                        new_c_loss = self.loss(new_logits, y)
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
            if self.loss == F.mse_loss:
                for i in range(y.shape[0]):
                    self.tot_steps += 1
                    equal = torch.isclose(logits[i], y[i], atol=self.epsilon)
                    self.train_acc += equal.item()*1.0
            else:
                self.train_acc += (logits.max(1)[1] == y).sum().item()
                self.tot_steps += y.shape[0]
            if self.iter_warm_up == 0:
                self.warm_up = False
                # self.min_acc = self.train_acc
            else:
                self.iter_warm_up -= 1
            self.lamda = self.lamda_stuck if self.lamda_stuck is not None else self.lamda
        u_rand = torch.rand_like(x)*torch.rand(1).item()*10
        v_rand = torch.rand_like(x)*torch.rand(1).item()*10
        self.random_lip_constant = self.lipschitz(u_rand, v_rand).item()
        self.train_acc /= self.tot_steps
    
    def test_step(self, test_loader, attack = None, verbosity = 1):
        self.model.eval()
        
        test_acc = 0.0
        test_loss = 0.0
        tot_steps = 0
        
        # -------------------------------------------------------------------------
        # loop over all batches
        for batch_idx, (x, y) in enumerate(test_loader):
            #print("batch_idx", batch_idx)
            #print("x shape", x.shape)
            # get batch data
            x, y = x.to(self.device), y.to(self.device)

            # update x to a adverserial example
            if attack is not None:
                x = attack(x, y)
            
            # evaluate model on batch
            logits = self.model(x)
            
            # Get classification loss
            #print("y shape", y.shape)
            #print("logits shape", logits.shape)
            c_loss = self.loss(logits, y)
            test_loss += c_loss.item()

            if self.loss == F.mse_loss:
                for i in range(y.shape[0]):
                    tot_steps += 1
                    equal = torch.isclose(logits[i], y[i], atol=self.epsilon)
                    test_acc += equal.item()*1.0
            elif self.loss == F.cross_entropy:
                test_acc += (logits.max(1)[1] == y).sum().item()
                tot_steps += y.shape[0]
            else:
                raise ValueError("loss should be F.mse_loss or F.cross_entropy")
        
        test_acc /= tot_steps
            
        # print accuracy
        if verbosity > 0: 
            print(50*"-")
            print('Test Accuracy:', test_acc)
        return {'test_loss':test_loss, 'test_acc':test_acc}
    
    def plot(self, ax=None, line=None, xmin=-1, xmax=1, dim=1, projection_dim=1, polynom=None):
        ax = plt if ax is None else ax
        if dim == 1:
            x = torch.linspace(xmin, xmax, 100).to(device)
            y = self.model(x.t()).cpu().detach()
            x = x.cpu().detach()
            if line is None:
                line = ax.plot(x, y)[0]
            else:
                line.set_ydata(y)
                
            return line
        else:
            x, x_true = linear_points(num_pts=100, xmin=xmin, xmax=xmax, dim=dim, coordinate=projection_dim)
            y = self.model(x_true).cpu().detach()
            x = x.cpu().detach()
            if polynom is not None:
                poly = polynom.createdata(x_true, sigma=0.)[1].cpu().detach()[:, dim]
                ax.plot(x.cpu(), poly, lw=2, color='black')
            ax.plot(x, y)
            return None

class adversarial_attack:
    def __init__(self, model, loss = F.mse_loss, attack_type="gauss_attack", nl =1, epsilon=1e-1, xmin=-1, xmax=1, num_iters=1):
        self.model = model
        self.loss = loss
        self.epsilon = epsilon
        self.nl = nl
        self.xmin = xmin
        self.xmax = xmax
        self.num_iters = num_iters
        self.attack_type = attack_type
        self.attack = None
        if attack_type == "gauss_attack":
            self.attack = at.gauss_attack(nl=self.nl, x_min=self.xmin, x_max=self.xmax)
        elif attack_type == "fgsm_attack":
            self.attack = at.fgsm(self.loss, epsilon=self.epsilon, x_min=self.xmin, x_max=self.xmax)
        elif attack_type == "pgd_attack":
            self.attack = at.pgd(self.loss, x_min=self.xmin, x_max=self.xmax, attack_iters=self.num_iters, restarts=1, epsilon=self.epsilon, alpha=None, alpha_mul=1.0, norm_type="l2")
        else:
            raise ValueError("Attack type not defined")
    
    def update_attack(self):
        if self.attack_type == "gauss_attack":
            self.attack = at.gauss_attack(nl=self.nl, x_min=self.xmin, x_max=self.xmax)
        elif self.attack_type == "fgsm_attack":
            self.attack = at.fgsm(self.loss, epsilon=self.epsilon, x_min=self.xmin, x_max=self.xmax)
        elif self.attack_type == "pgd_attack":
            self.attack = at.pgd(self.loss, x_min=self.xmin, x_max=self.xmax, attack_iters=self.num_iters, restarts=1, epsilon=self.epsilon, alpha=None, alpha_mul=1.0, norm_type="l2")
        else:
            raise ValueError("Attack type not defined")

    def __call__(self, x, y):
        return self.attack(self.model, x, y)

def scattered_points(num_pts=100, xmin=-1, xmax=1, percent_loss=0.3, random=True, dim=1):
    if dim == 1:
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
    else:
        O = torch.zeros(dim)
        x = []
        if not random:
            for _ in range(num_pts):
                u =  (torch.rand_like(O) - torch.rand_like(O)).to(device)
                up_or_down = torch.rand(1).item()
                ray = torch.rand(1).item()*((xmax - xmin)/4)*(1-percent_loss)*(up_or_down < 0.5) + (((xmax - xmin)/4)*(1+percent_loss) + torch.rand(1).item()*((xmax - xmin)/4)*(1-percent_loss))*(up_or_down > 0.5)
                u = (u / torch.norm(u))*ray
                x_next = O + u
                x.append(x_next.detach().cpu().numpy())
        else:
            m = torch.randint(1, dim, (1,)).item()
            v = torch.zeros(dim)
            v[m] = 1
            n = 0
            while n < num_pts:
                u =  (torch.rand_like(O) - torch.rand_like(O)).to(device)
                up_or_down = torch.rand(1).item()
                u = (u / torch.norm(u))*(xmax - xmin)/2
                x_next = O + u
                xv_scalar = torch.abs(torch.dot(x_next, v)/(torch.norm(v)*torch.norm(x_next)))
                if xv_scalar > percent_loss:
                    x.append(x_next.detach().cpu().numpy())
                    n += 1
        x = torch.tensor(x)
    return x

def linear_points(num_pts=100, xmin=-1, xmax=1, dim=1, coordinate=0):
    if dim == 1:
        x = torch.linspace(xmin, xmax, num_pts)
        return x
    else:
        x_i = torch.linspace(xmin, xmax, num_pts)
        x = torch.rand(num_pts, dim)
        print(x.shape)
        for i in range(num_pts):
            x[i, coordinate] = x_i[i]
        return x_i, x
    

if __name__ == "__main__":
    # plt.close('all')
    # fig, ax = plt.subplots(1,)
    # coeffs = torch.tensor([0.5,0.01,-0.,0.,0.,0.,1]).to(device)
    # polynom = Polynom(coeffs, scaling=0.00005)
    # xmin,xmax=(-3,3)
    # polynom.plot(xmin=xmin,xmax=xmax)
    # x = scattered_points(num_pts=50, xmin=xmin, xmax=xmax, percent_loss=0.75, random=False).to(device)
    # xy_loader = polynom.createdata(x,sigma=0.0)[0]
    # XY = torch.stack([xy_loader.dataset.tensors[i] for i in [0,1]])
    
    # ax.set_ylim(XY[1, :].min()-0.01, XY[1, :].max()+0.01)
    
    # model = fully_connected([1, 50, 100, 50, 1], "ReLU")
    # model = model.to(device)

    # #ax.plot(x.cpu(),model(x).cpu().detach())
    # num_total_iters = 500
    # lamda = 0.1

    # trainer = Trainer(model, xy_loader, 100, lamda=lamda, lr=0.001, adversarial_name="Nesterov", num_iters=50, epsilon=5e-3, min_accuracy=None, CLIP_estimation="sum", iter_warm_up=num_total_iters//4)#, lamda_stuck=lamda)#, backtracking=0.9)
    # line = trainer.plot(ax=ax, xmin=xmin,xmax=xmax)
    # #plt.show()
    # ax.scatter(XY[0,:].cpu(),XY[1,:].cpu())
    # start_time = time.time()
    # for i in range(num_total_iters):
    #     trainer.train_step()
    #     if i % 1 == 0:
    #         print(i)
    #         #print(model.layers[1].weight)
    #         #ax.set_title('Iteration: ' + str(i))
    #         print("train accuracy : ", trainer.train_acc)
    #         print("train loss : ", trainer.train_loss)
    #         print("train lip loss : ", trainer.train_lip_loss)
    #         print("current lamda : ", trainer.lamda)
    #         #polynom.plot(ax=ax)
    #         trainer.plot(ax=ax, line=line, xmin=xmin,xmax=xmax)
    #         plt.pause(0.1)
    # print("time for training : ", time.time() - start_time)
    # ax.set_title('Iteration: ' + str(num_total_iters))
    # #polynom.plot(ax=ax, xmin=xmin,xmax=xmax)
    # trainer.plot(ax=ax, line=line, xmin=xmin,xmax=xmax)
    # ax.legend(["True polynom", "Fully Connected Model"])
    # fig.savefig('final_plot.png')
    # plt.show()

############################################################################################################################################################################
    sizes = [784, 200, 80, 10]
    lamda = 0.5
    num_total_iters = 50
    min_accuracy = 0.9
    warm_up_iter = num_total_iters//4
    model = fully_connected(sizes, "ReLU")
    model = model.to(device)
    data_file = "/home/bernas/VSC/dataset_MNIST/MNIST"
    xy_loader, _, test_loader = All_MNIST(data_file, download=False, data_set="MNIST", batch_size=100, train_split=0.9, num_workers=1)()
    trainer = Trainer(model, xy_loader, 100, lamda=lamda, loss =  F.cross_entropy, lr=0.001, adversarial_name="Adam", num_iters=5, min_accuracy=min_accuracy, CLIP_estimation="sum", iter_warm_up=warm_up_iter, change_lamda_in=False)#, backtracking=0.9)
    model_max = fully_connected(sizes, "ReLU")
    model_max = model_max.to(device)
    trainer_max = Trainer(model_max, xy_loader, 100, lamda=lamda, lr=0.001, loss =  F.cross_entropy, adversarial_name="Adam", num_iters=5, min_accuracy=min_accuracy, CLIP_estimation="max", iter_warm_up=warm_up_iter, change_lamda_in=False)#, backtracking=0.9)
    model_adv = fully_connected(sizes, "ReLU")
    model_adv = model_adv.to(device)
    trainer_adv = Trainer(model_adv, xy_loader, 100, lr=0.001, loss =  F.cross_entropy, min_accuracy=min_accuracy, iter_warm_up=warm_up_iter)#, backtracking=0.9)

    print("test result for sum CLIP :")
    test_result = trainer.test_step(test_loader, verbosity=1)
    print("test result for max CLIP :")
    test_result = trainer_max.test_step(test_loader, verbosity=1)
    print("adversarial attack test result for sum CLIP :")
    attack = adversarial_attack(model, loss = F.cross_entropy, attack_type="pgd_attack", nl=0.1, epsilon=1.0, xmin=-1, xmax=1, num_iters=100)
    attack_on_max = adversarial_attack(model_max, loss = F.cross_entropy, attack_type="pgd_attack", nl=0.1, epsilon=0.1, xmin=-1, xmax=1, num_iters=100)
    test_result = trainer.test_step(test_loader, attack = attack, verbosity=1)
    print("adversarial attack test result for max CLIP :")
    test_result = trainer_max.test_step(test_loader, attack = attack_on_max, verbosity=1)
    print("test result for adversarial training :")
    test_result = trainer_adv.test_step(test_loader, verbosity=1)
    print("adversarial attack test result for adversarial training :")
    test_result = trainer_adv.test_step(test_loader, attack = attack, verbosity=1)
    for i in range(num_total_iters):
        trainer.train_step()
        if i % 1 == 0:
            print(i)
            print("train accuracy : ", trainer.train_acc)
            print("train loss : ", trainer.train_loss)
            print("train lip loss : ", trainer.train_lip_loss)
            print("random lip constant : ", trainer.random_lip_constant)
            print("saved basic loss : ", trainer.saved_basic_loss)
            print("current lamda : ", trainer.lamda if not trainer.warm_up else 0)

    for i in range(num_total_iters):
        trainer_max.train_step()
        if i % 1 == 0:
            print(i)
            print("train accuracy : ", trainer_max.train_acc)
            print("train loss : ", trainer_max.train_loss)
            print("train lip loss : ", trainer_max.train_lip_loss)
            print("random lip constant : ", trainer_max.random_lip_constant)
            print("saved basic loss : ", trainer_max.saved_basic_loss)
            print("current lamda : ", trainer.lamda if not trainer.warm_up else 0)
    
    for i in range(num_total_iters):
        trainer_adv.adversarial_trainer_step()
        if i % 1 == 0:
            print(i)
            print("train accuracy : ", trainer_adv.train_acc)
            print("train loss : ", trainer_adv.train_loss)
            print("train lip loss : ", trainer_adv.train_lip_loss)
            print("random lip constant : ", trainer_adv.random_lip_constant)
            print("saved basic loss : ", trainer_adv.saved_basic_loss)
            print("current lamda : ", trainer.lamda if not trainer.warm_up else 0)

    print("test result for sum CLIP :")
    test_result = trainer.test_step(test_loader, verbosity=1)
    print("test result for max CLIP :")
    test_result = trainer_max.test_step(test_loader, verbosity=1)
    print("adversarial attack test result for sum CLIP :")
    test_result = trainer.test_step(test_loader, attack = attack, verbosity=1)
    print("adversarial attack test result for max CLIP :")
    test_result = trainer_max.test_step(test_loader, attack = attack_on_max, verbosity=1)
    print("test result for adversarial training :")
    test_result = trainer_adv.test_step(test_loader, verbosity=1)
    print("adversarial attack test result for adversarial training :")
    test_result = trainer_adv.test_step(test_loader, attack = attack, verbosity=1)
    
    espilon_list = torch.logspace(-3, 0, 50)
    adv_acc = []
    adv_acc_max = []
    adv_acc_adv = []
    for epsilon in espilon_list:
        attack.epsilon = epsilon
        attack_on_max.epsilon = epsilon
        attack.update_attack()
        attack_on_max.update_attack()
        adv_acc.append(trainer.test_step(test_loader, attack = attack, verbosity=0)['test_acc'])
        adv_acc_max.append(trainer_max.test_step(test_loader, attack = attack_on_max, verbosity=0)['test_acc'])
        adv_acc_adv.append(trainer_adv.test_step(test_loader, attack = attack, verbosity=0)['test_acc'])

    fig, ax = plt.subplots(1,1)
    ax.plot(espilon_list, adv_acc, label="sum CLIP")
    ax.plot(espilon_list, adv_acc_max, label="max CLIP")
    ax.plot(espilon_list, adv_acc_adv, label="adversarial training")
    ax.set_xscale('log')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('accuracy')
    ax.legend()
    plt.show()