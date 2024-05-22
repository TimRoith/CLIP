import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from itertools import cycle
from torch.optim import SGD, Adam

def l2_norm(x):
    return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)

class Flatten(nn.Module):
    def forward(self, x):
        if sum(x.shape)>2:
            return x.view(x.size(0), -1)
        else:
            return x

def get_reg_batch(conf, iterator):
    data, target = next(iterator)
    return data.to(conf.device), target.to(conf.device)

def default_initializer():
    pass

class lip_constant_estimate:
    def __init__(
        self, model, 
        out_norm = None, 
        in_norm=None, 
        mean=False
    ):

        self.model = model
        self.out_norm = out_norm if out_norm is not None else l2_norm
        self.in_norm = in_norm if in_norm is not None else l2_norm
        self.mean = mean

    def __call__(self, u, v):
        u_out = self.model(u)
        v_out = self.model(v)
        loss = self.out_norm(u_out - v_out) / self.in_norm(u - v)
        if self.mean:
            return torch.mean(torch.square(loss))
        else:
            return torch.square(loss)
        
        
        
class adversarial_update:
    def __init__(self, 
               model,
               u, v, 
               opt_kwargs):
        
        self.lip_constant_estimate = lip_constant_estimate(model)
        self.u = nn.Parameter(u.clone())
        self.v = nn.Parameter(v.clone())
        
        opt_name = opt_kwargs.get('name', 'SGD')
        if opt_name == 'SGD':
            self.opt = SGD([self.u, self.v], 
                           lr=opt_kwargs.get('lr', 0.1), 
                           momentum=opt_kwargs.get('lr', 0.9))
        elif opt_name == 'Adam':
            self.opt = Adam([self.u, self.v], 
                           lr=opt_kwargs.get('lr', 0.001),)
        elif opt_name == 'Nesterov':
            self.opt = SGD([self.u, self.v], 
                           lr=opt_kwargs.get('lr', 0.1), 
                           momentum=opt_kwargs.get('lr', 0.9),
                           nesterov=True)
        else:
            raise ValueError('Unknown optimizer: ' + str(opt_kwargs['name']))
        
        
    def step(self,):
        self.opt.zero_grad()
        
        loss = self.lip_constant_estimate(self.u, self.v)
        loss_sum = -torch.sum(loss)
        loss_sum.backward()
        
        self.opt.step()
        
        

class adversarial_gradient_ascent:
    def __init__(self,
        model,
        u, v,
        lr = 0.1,
        lr_decay = .95,
        reg_init = "partial_random"):

        self.lr = lr
        self.lr_decay = lr_decay
        self.lip_constant_estimate = lip_constant_estimate(model)
        self.reg_init = reg_init

        self.u = u
        self.v = v

    def step(self,):
        self.u.requires_grad = True
        self.v.requires_grad = True
        loss = self.lip_constant_estimate(self.u, self.v)
        loss_sum = torch.sum(loss)
        loss_sum.backward()

        for z in [self.u, self.v]:
            z_grad = z.grad.detach()
            z_tmp = z.data + self.lr * z_grad
            z.grad.zero_()
            z.data = z_tmp

        loss_tmp = self.lip_constant_estimate(self.u, self.v)
        # mask =  (loss_tmp > loss_best).type(torch.int).view(-1, 1, 1, 1)
        # u.data = u_tmp * mask + u.data * (1 - mask)
        # v.data = v_tmp * mask + v.data * (1 - mask)
        #self.lr = (self.lr * self.lr_decay) # * (1 - mask)) + (lr * conf.reg_decay * mask)
        # u.data = torch.clamp(u.data, 0., 1.)
        # v.data = torch.clamp(v.data, 0., 1.)
        return loss_tmp
        # loss_best = loss_best * (1 - mask).squeeze() + loss_tmp.detach() * mask.squeeze()

class adverserial_nesterov_accelerated:
    def __init__(self,
        model,
        u, v,
        lr = 0.1,
        lr_decay = .95,
        reg_init = "partial_random"):

        self.lr = lr
        self.lr_decay = lr_decay
        self.lip_constant_estimate = lip_constant_estimate(model)
        self.reg_init = reg_init
        self.u = u
        self.v = v

        self.alpha = torch.zeros(len(u)).to(u.device)
        self.updated_grad_uv = torch.cat((u, v)).detach() # this needs to be the initial value of the uv

    def step(self,):
        self.u.requires_grad = True
        self.v.requires_grad = True
        loss = self.lip_constant_estimate(self.u, self.v)
        loss_sum = torch.sum(loss)
        loss_sum.backward()

        u_grad = self.u.grad.detach()
        u_futur = self.u.data + self.lr * u_grad
        self.u.grad.zero_()
        v_grad = self.v.grad.detach()
        v_futur = self.v.data + self.lr * v_grad
        self.v.grad.zero_()
        alpha_next = (1 + torch.sqrt(1 + 4 * self.alpha ** 2)) / 2
        gamma = (1 - self.alpha) / (alpha_next)


        u_tmp = (1 - gamma) * u_futur + gamma * self.updated_grad_uv[:u_futur.shape[0]]
        v_tmp = (1 - gamma) * v_futur + gamma * self.updated_grad_uv[u_futur.shape[0]:]

        loss_tmp = self.lip_constant_estimate(u_tmp, v_tmp)
        # mask =  (loss_tmp > loss_best).type(torch.int).view(-1, 1, 1, 1)
        self.u.data = u_tmp #* mask + self.u.data * (1 - mask)
        self.v.data = v_tmp #* mask + self.v.data * (1 - mask)

        self.lr = (self.lr * self.lr_decay) # * mask + (lr / self.lr_decay * (1 - mask))
        self.alpha = alpha_next #* mask + self.alpha * (1 - mask)
        self.updated_grad_uv = torch.cat((self.u, self.v)).detach() #* torch.cat((mask, mask)) + self.updated_grad_uv * (1 - torch.cat((mask, mask)))
        #u.data = torch.clamp(u.data, 0., 1.)
        #v.data = torch.clamp(v.data, 0., 1.)
        loss_best = loss_tmp.detach() #* mask.squeeze() + loss_best * (1 - mask).squeeze()
        return loss_best
    
class plotting_adversarial_update:
    def __init__(self, model, x_min, x_max, adv_iters=10, type="gradient_ascent"):
        self.model = model
        self.space = torch.linspace(x_min, x_max, 1000).to(model.device)
        self.adv_iters = adv_iters
        self.type = type

    def set_updater(self, u, v):
        if self.type == "gradient_ascent":
            adv = adversarial_gradient_ascent(self.model, u, v, lr=1)
        elif self.type == "nesterov_accelerated":
            adv = adverserial_nesterov_accelerated(self.model, u, v, lr=1)
        elif self.type == "torch_gradient_ascent":
            adv = adversarial_update(self.model, u, v, opt_kwargs={'name':'SGD', 'lr':1})
        elif self.type == "torch_nesterov":
            adv = adversarial_update(self.model, u, v, opt_kwargs={'name':'Nesterov', 'lr':1})
        elif self.type == "torch_adam":
            adv = adversarial_update(self.model, u, v, opt_kwargs={'name':'Adam', 'lr':0.01})
        else:
            raise ValueError("adversarial_update" + self.type + " is unknown.")
        return adv

    def print_uv(self, u, v):
        adv = self.set_updater(u, v)
        for _ in range(self.adv_iters):
            adv.step()
        print(adv.u.cpu().detach(), adv.v.cpu().detach())
    
    def first_plot(self):
        plt.plot(self.space.cpu(), self.model(self.space.unsqueeze(1)).squeeze(1).cpu().detach())
        plt.show()
    
    def second_plot(self, u, v):
        plt.plot(self.space.cpu(), self.model(self.space.unsqueeze(1)).squeeze(1).cpu().detach())
        plt.plot(u.cpu().detach(), self.model(u.unsqueeze(1)).squeeze(1).cpu().detach(), 'ro', markersize=10)
        plt.plot(v.cpu().detach(), self.model(v.unsqueeze(1)).squeeze(1).cpu().detach(), 'ro', markersize=10)
        adv = self.set_updater(u, v)
        for _ in range(self.adv_iters):
            adv.step()
            u_new = adv.u
            v_new = adv.v
            plt.plot(u_new.cpu().detach(), self.model(u_new.unsqueeze(1)).squeeze(1).cpu().detach(), 'go')
            plt.plot(v_new.cpu().detach(), self.model(v_new.unsqueeze(1)).squeeze(1).cpu().detach(), 'go')
            plt.pause(.2,)
        plt.show()

class fully_connected(nn.Module):
    def __init__(self, sizes, act_fun, mean = 0.0, std = 1.0):
        super(fully_connected, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act_fn = get_activation_function(act_fun)
        self.mean = mean
        self.std = std
        layer_list = [Flatten()]
        for i in range(len(sizes)-2):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(self.act_fn())
            
        layer_list.append(nn.Linear(sizes[-2], sizes[-1]))
            
        self.layers = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        x = (x - self.mean)/self.std
        return self.layers(x)
    
def get_activation_function(activation_function):
    af = None
    if activation_function == "ReLU":
        af = nn.ReLU
    elif activation_function == "sigmoid":
        af = nn.Sigmoid
    else:
        af = nn.ReLU
    return af


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fully_connected([1, 20, 1], "sigmoid")
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            layer.weight.data = torch.randn(layer.weight.data.size())
            print(layer.weight.data)
    model = model.to(device)
    plotting = plotting_adversarial_update(model, -1, 1, type="nesterov_accelerated", adv_iters=1000)
    #plotting.first_plot()
    plotting.second_plot(torch.tensor([-0.5]).to(device), torch.tensor([0.5]).to(device))