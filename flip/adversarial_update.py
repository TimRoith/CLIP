import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from itertools import cycle
from torch.optim import SGD, Adam

def l2_norm(x):
    return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)

class lip_constant_estimate:
    def __init__(
        self, model, 
        out_norm = None, 
        in_norm=None, 
        estimation="sum"
    ):

        self.model = model
        self.out_norm = out_norm if out_norm is not None else l2_norm
        self.in_norm = in_norm if in_norm is not None else l2_norm
        self.estimation = estimation

    def __call__(self, u, v):
        u_out = self.model(u)
        v_out = self.model(v)
        loss = self.out_norm(u_out - v_out) / self.in_norm(u - v)
        if self.estimation == "sum":
            return torch.mean(torch.square(loss))
        elif self.estimation == "max":
            return torch.square(torch.max(loss))
        elif self.estimation == "1D":
            return torch.square(loss)
        else :
            raise ValueError("Lipschitz estimation should be '1D', 'max' or 'sum'")
        
class adversarial_update:
    def __init__(self, 
               model,
               u, v, 
               adv_kwargs,
               estimation,
               in_norm = None,
               out_norm = None):
        
        self.model = model
        self.lip_constant_estimate = lambda u, v: lip_constant_estimate(self.model, estimation = estimation)(u, v)
        self.u = nn.Parameter(u.clone())
        self.v = nn.Parameter(v.clone())
        
        opt_name = adv_kwargs.get('name', 'SGD')
        if opt_name == 'SGD':
            self.opt = SGD([self.u, self.v], 
                           lr=adv_kwargs.get('lr', 0.1), 
                           momentum=adv_kwargs.get('lr', 0.9))
        elif opt_name == 'Adam':
            self.opt = Adam([self.u, self.v], 
                           lr=adv_kwargs.get('lr', 0.001),)
        elif opt_name == 'Nesterov':
            self.opt = SGD([self.u, self.v], 
                           lr=adv_kwargs.get('lr', 0.1), 
                           momentum=adv_kwargs.get('lr', 0.9),
                           nesterov=True)
        else:
            raise ValueError('Unknown optimizer: ' + str(adv_kwargs['name']))
        
        
    def step(self,):
        self.opt.zero_grad()
        
        loss_ = self.lip_constant_estimate(self.u, self.v)
        loss_sum = -torch.sum(loss_)
        gradu, gradv = torch.autograd.grad(loss_sum, [self.u, self.v])
        self.u.grad = gradu
        self.v.grad = gradv

        # loss_sum.backward()
        # loss_ = -loss_
        # loss_.backward()
        
        self.opt.step()

class projected_adversarial_update :
    def __init__(self, 
               model,
               u, v, 
               adv_kwargs,
               estimation,
               ray_kwargs,
               in_norm = None,
               out_norm = None):
        
        self.model = model
        self.lip_constant_estimate = lambda u, v: lip_constant_estimate(self.model, estimation = estimation)(u, v)
        self.u = nn.Parameter(u.clone())
        self.v = nn.Parameter(v.clone())
        self.x = nn.Parameter(ray_kwargs.get('x', u.clone()))
        self.ray = ray_kwargs.get('ray', 0.1)
        
        opt_name = adv_kwargs.get('name', 'SGD')
        if opt_name == 'SGD':
            self.opt = SGD([self.u, self.v], 
                           lr=adv_kwargs.get('lr', 0.1), 
                           momentum=adv_kwargs.get('lr', 0.9))
        elif opt_name == 'Adam':
            self.opt = Adam([self.u, self.v], 
                           lr=adv_kwargs.get('lr', 0.001),)
        elif opt_name == 'Nesterov':
            self.opt = SGD([self.u, self.v], 
                           lr=adv_kwargs.get('lr', 0.1), 
                           momentum=adv_kwargs.get('lr', 0.9),
                           nesterov=True)
        else:
            raise ValueError('Unknown optimizer: ' + str(adv_kwargs['name']))
        
    def step(self,):
        self.opt.zero_grad()
        loss_ = self.lip_constant_estimate(self.u, self.v)
        loss_sum = -torch.sum(loss_)
        gradu, gradv = torch.autograd.grad(loss_sum, [self.u, self.v])
        self.u.grad = gradu
        self.v.grad = gradv
        self.opt.step()
        
        self.u.data = self.x + self.ray * (self.u - self.x) / torch.norm(self.u - self.x)
        self.v.data = self.x + self.ray * (self.v - self.x) / torch.norm(self.v - self.x)