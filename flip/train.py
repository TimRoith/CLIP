import torch
import torch.nn as nn
import torch.optim as optim
from .adversarial_update import adversarial_update, lip_constant_estimate

class Trainer:
    def __init__(
            self,
            model, 
            train_loader, val_loader=None,
            loss = None,
            opt_kwargs = None,
            verbosity = 0,
            epochs = 100,
            ):
        
        self.model = model
        self.train_loader = train_loader
        self.num_data = len(train_loader.dataset)
        self.val_loader = val_loader
        self.verbosity = verbosity
        self.epochs = epochs
        self.lamda = None
        self.hist= {'acc':[], 'loss':[]}
        
    
        self.loss = nn.CrossEntropyLoss() if loss is None else loss
        opt_kwargs = {'lr':0.1} if opt_kwargs is None else opt_kwargs
        self.opt_kwargs = {k:v for k,v in opt_kwargs.items() if not k=='type'}
        self.opt_cls = opt_kwargs.get('type', optim.SGD)
        
    def update(self, x, y):
        pass
        
    def train_step(self,):
        self.running_acc = 0
        self.running_loss = 0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            self.update(x, y)
            
        self.hist['loss'].append(self.running_loss/self.num_data)
        self.hist['acc'].append(self.running_acc/self.num_data)
        
        self.print_step()
        
    
    def print_step(self,):
        if self.verbosity > 0:
            for k,v in self.hist.items():
                print(str(k)+': ' + str(v[-1]))
    
    def lamda_schedule(self,):
        pass
    
    def train(self,):
        self.model.train()
        self.opt = self.opt_cls(self.model.parameters(), **self.opt_kwargs)
        self.init_hist()
        
        for e in range(self.epochs):
            self.train_step()
            self.lamda_schedule()
            
    def init_hist(self):
        self.hist= {'acc':[], 'loss':[]}
        
            
            
class StandardTrainer(Trainer):
    def __init__(
            self,
            model, 
            train_loader, val_loader=None,
            loss = None,
            opt_kwargs = None,
            **kwargs
            ):
        super().__init__(model, train_loader, 
                         val_loader=val_loader, 
                         loss=loss,
                         opt_kwargs=opt_kwargs,
                         **kwargs)
        
    def update(self, x, y):
        self.opt.zero_grad() # reset gradients
        logits = self.model(x)
        loss = self.loss(logits, y)
        loss.backward()
        self.opt.step()
        
        #%%
        self.running_acc += torch.sum(logits.topk(1)[1][:,0]==y)
        self.running_loss += loss.item()
        

class AdversarialTrainer(Trainer):
    def __init__(
            self,
            model, 
            train_loader, val_loader=None,
            loss = None,
            opt_kwargs = None,
            num_iters = 5,
            **kwargs
            ):
        super().__init__(model, train_loader, 
                         val_loader=val_loader, 
                         loss=loss,
                         opt_kwargs=opt_kwargs,
                         **kwargs)
        self.num_iters = num_iters
        
    def update(self, x, y):
        self.opt.zero_grad() # reset gradients
        x_adv = x.clone().detach().requires_grad_(True)
        for _ in range(self.num_iters):
            logits = self.model(x_adv)
            c_loss = self.loss(logits, y)
            c_loss.backward()
            x_adv = x_adv + self.lr * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, min=0, max=1)
            x_adv = x_adv.detach().requires_grad_(True)
        logits = self.model(x_adv)
        loss = self.loss(logits, y)
        loss.backward()
        self.opt.step()
        
        self.running_acc += torch.sum(logits.topk(1)[1][:,0]==y)
        self.running_loss += loss.item()  


class FLIPTrainer(Trainer):
    def __init__(
            self,
            model, 
            train_loader, val_loader=None,
            loss = None,
            opt_kwargs = None,
            lamda = 0.1,
            adv_kwargs = None,
            num_iters = 5,
            estimation = "max",
            min_acc = 0.9,
            **kwargs
            ):
        super().__init__(model, train_loader, 
                         val_loader=val_loader, 
                         loss=loss,
                         opt_kwargs=opt_kwargs,
                         **kwargs)
        self.num_iters = num_iters
        self.lipschitz = lambda u, v: lip_constant_estimate(self.model, estimation = estimation)(u, v)
        self.adversarial = lambda u: adversarial_update(self.model, u, u+torch.rand_like(u)*0.1, adv_kwargs=adv_kwargs, estimation = estimation)
        self.lamda = lamda
        self.min_acc = min_acc
        self.dlamda = lamda*0.1
        self.lamda_bound = [lamda*(1/4),lamda*4]
        
    def lamda_schedule(self):
        acc = self.hist['acc'][-1].item()
        if acc > self.min_acc:
            self.lamda = max(self.lamda + self.dlamda, self.lamda_bound[1])
            self.dlamda = self.dlamda*0.99
        else:
            self.lamda = min(self.lamda - self.dlamda, self.lamda_bound[0])
            self.dlamda = self.dlamda*0.99
        
        
    def update(self, x, y):
        self.opt.zero_grad() # reset gradients
        x_adv = x.clone().detach().requires_grad_(True)
        adv = self.adversarial(x_adv)
        for _ in range(self.num_iters):
            adv.step()
        u, v = adv.u, adv.v
        c_reg_loss = self.lipschitz(u, v)
        logits = self.model(x_adv)
        loss = self.loss(logits, y)
        loss = loss + self.lamda * c_reg_loss
        loss.backward()
        self.opt.step()
        
        self.running_acc += torch.sum(logits.topk(1)[1][:,0]==y)
        self.running_loss += loss.item()




# class Trainer:
#     def __init__(self, model, train_loader, lip_reg_max,loss = F.mse_loss , lamda=0.1, percent_of_lamda=0.1, min_accuracy=None, lr=0.1, adversarial_name="SGD", num_iters=1, epsilon=1e-2, backtracking=None, in_norm=None, out_norm=None, CLIP_estimation = "sum", iter_warm_up=None, lamda_stuck = None, change_lamda_in = True):
#         self.device = device
#         self.model = model.to(self.device)
#         self.train_loader = train_loader
#         self.lr = lr
#         self.backtracking = backtracking
#         self.reg_max = lip_reg_max
#         self.num_iters = num_iters
#         self.loss = loss
#         self.lipschitz = lambda u, v: lip_constant_estimate(self.model, estimation = CLIP_estimation)(u, v)
#         self.adversarial = lambda u: adversarial_update(self.model, u, u+torch.rand_like(u)*0.1, opt_kwargs={'name':adversarial_name, 'lr':self.lr}, in_norm = in_norm, out_norm = out_norm)
#         #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         self.lamda = lamda
#         self.lamda_stuck = lamda_stuck
#         self.maximal_lamda = lamda*4
#         self.minimal_lamda = lamda*0.25
#         self.dlamda = lamda*percent_of_lamda
#         # self.saved_min_acc = min_accuracy
#         self.min_acc = min_accuracy
#         self.change_lamda_in = change_lamda_in
#         self.epsilon = epsilon
#         self.train_acc = 0.0
#         self.train_loss = 0.0
#         self.tot_steps = 0
#         self.train_lip_loss = 0.0
#         self.saved_basic_loss = 0.0
#         self.random_lip_constant = 0.0
#         self.warm_up = True
#         if iter_warm_up is not None:
#             self.iter_warm_up = iter_warm_up
#         else:
#             print("No warm up iteration defined, using default value of 10.")
#             self.iter_warm_up = 10

#     def set_learning_rate(self, optimizer, new_lr):
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = new_lr

#     def train_step(self):
#         # train phase
#         self.model.train()
#         opt = self.optimizer
#         #scheduler = ReduceLROnPlateau(opt, 'min')
#         # initialize values for train accuracy and train loss
#         self.train_acc = 0.0
#         self.train_loss = 0.0
#         self.tot_steps = 0
#         self.train_lip_loss = 0.0

#         #
#         u = None
#         v = None

#         # -------------------------------------------------------------------------
#         # loop over all batches
#         for batch_idx, (x, y) in enumerate(self.train_loader):
#             # get batch data
#             x, y = x.to(self.device), y.to(self.device)
#             # ---------------------------------------------------------------------
#             # Adversarial update
#             # ---------------------------------------------------------------------
#             # adversarial update on the Lipschitz set
#             #ux = torch.linspace(-3, 3, 40).to(device)
#             if not self.warm_up:
#                 if sum(x.shape)-x.shape[0] > 1:
#                     t_weight = torch.logspace(-3, 3, x.shape[0]).to(device)
#                     ux = x[:, None]
#                     ux.requires_grad = True
#                     vx = x + torch.rand_like(x)*1000
#                     lip_ux = self.lipschitz(ux, vx)
#                     grad_lip_ux = torch.autograd.grad(lip_ux, ux, create_graph=False)[0]
#                     max_grad = 0
#                     index = 0
#                     for i in range(x.shape[0]):
#                         if torch.norm(grad_lip_ux[i]) > max_grad:
#                             max_grad = torch.norm(grad_lip_ux[i])
#                             index = i
#                     best_ux_value = ux[index]
#                     ux = best_ux_value
#                     ux_grad = grad_lip_ux[index]*t_weight[0]
#                     #print("x shape", x.shape)
#                     for i in range(x.shape[0]-1):
#                         ux = torch.cat([ux, best_ux_value], dim=0)
#                         ux_grad = torch.cat([ux_grad, grad_lip_ux[index]*t_weight[i+1]], dim=0)
#                     #print("grad shape", t_weight[:,None].shape)
#                     ux = ux + ux_grad
#                 else:
#                     ux = torch.linspace(-3, 3, x.shape[0]).to(device)
#                 #ux = x[:, None]
#                 #print("x shape", x.shape)
#                 adv = self.adversarial(ux)
#                 for _ in range(self.num_iters):
#                     adv.step()
#                 u, v = adv.u, adv.v
#                 # print("u or v are nan", torch.isnan(u).any() or torch.isnan(v).any())
#                 # ---------------------------------------------------------------------

#                 # Compute the Lipschitz constant
#                 c_reg_loss = self.lipschitz(u, v)
            
#             # reset gradients
#             opt.zero_grad()

#             # evaluate model on batch
#             logits = self.model(x)
#             #print("logits are nan", torch.isnan(logits).any())
#             if torch.isnan(logits).any() and torch.isnan(x).any():
#                 print("logits and x are nan")
#             if self.loss == F.mse_loss:
#                 logits = logits.reshape(logits.shape[0])

#             # Get classification loss
#             c_loss = self.loss(logits, y)
#             #c_loss = torch.sum((logits-y)**2)
#             self.saved_basic_loss = c_loss.detach().item()
#             # Use regularization parameter lamda
#             lamda = self.lamda
#             # check if Lipschitz term is too large. Note that regularization with
#             # large Lipschitz terms yields instabilities!
#             if not self.warm_up:
#                 if not (c_reg_loss.item() > self.reg_max):
#                     c_loss = c_loss + lamda * c_reg_loss
#                     pass
#                 else:
#                     print('The Lipschitz constant was too big:', c_reg_loss.item(), ". No Lip Regularization for this batch!")

#             # Backtracking line search
#             if self.backtracking is not None:
#                 initial_lr = self.lr
#                 beta = 0.5
#                 tau = self.backtracking

#                 # Compute the objective function with the current parameters
#                 f_up = c_loss.item()
#                 success = False
                
#                 while True:
#                     # Backup current model parameters
#                     backup_params = [p.clone() for p in self.model.parameters()]

#                     # Perform a gradient descent step with the reduced learning rate
#                     opt.step()

#                     # Compute the new objective function
#                     with torch.no_grad():
#                         new_logits = self.model(x)
#                         if self.loss == F.mse_loss:
#                             new_logits = new_logits.reshape(new_logits.shape[0])
#                         new_c_loss = self.loss(new_logits, y)
#                         if not self.warm_up:
#                             new_c_reg_loss = self.lipschitz(u, v)
#                             if not (new_c_reg_loss.item() > self.reg_max):
#                                 new_c_loss = new_c_loss + lamda * new_c_reg_loss

#                     f_down = new_c_loss.item()

#                     if f_down <= f_up - tau * initial_lr:
#                         success = True
#                         break
#                     else:
#                         # Restore model parameters and reduce the learning rate
#                         for original, backup in zip(self.model.parameters(), backup_params):
#                             original.data.copy_(backup.data)
#                         initial_lr *= beta
#                         self.set_learning_rate(opt, initial_lr)

#                 if not success:
#                     print("Backtracking line search failed to reduce the objective function.")

#             # Update model parameters
#             c_loss.backward()
#             opt.step()
#             #scheduler.step(c_loss)
            
#             # update accuracy and loss
#             self.train_loss += c_loss.item()
#             if not self.warm_up:
#                 self.train_lip_loss = max(c_reg_loss.item(), self.train_lip_loss)
#             if self.loss == F.mse_loss:
#                 for i in range(y.shape[0]):
#                     self.tot_steps += 1
#                     equal = torch.isclose(logits[i], y[i], atol=self.epsilon)
#                     self.train_acc += equal.item()*1.0
#             else:
#                 self.train_acc += (logits.max(1)[1] == y).sum().item()
#                 self.tot_steps += y.shape[0]
#             if self.min_acc is not None and self.change_lamda_in: 
#                 if self.train_acc > self.min_acc*self.tot_steps:
#                     self.warm_up = False
#                     self.lamda = min(self.lamda + self.dlamda, self.maximal_lamda)
#                     self.dlamda = self.dlamda*0.99
#                     # weight = 0.6
#                     # self.min_acc = (self.train_acc*weight + self.saved_min_acc*(2-weight))/2
#                 elif not self.warm_up:
#                      self.lamda = max(self.lamda - self.dlamda, self.minimal_lamda)
#                      self.dlamda = self.dlamda*0.99
#                     #  if self.lamda == self.minimal_lamda:
#                     #      self.min_acc = self.saved_min_acc
#             elif self.iter_warm_up == 0:
#                 self.warm_up = False
#                 if self.min_acc is None:
#                     self.min_acc = self.train_acc
#                 # self.saved_min_acc = (self.train_acc+0.5)/2
#             else:
#                 self.iter_warm_up -= 1
#             self.lamda = self.lamda_stuck if self.lamda_stuck is not None else self.lamda
#         u_rand = torch.rand_like(x)*torch.rand(1).item()*10
#         v_rand = torch.rand_like(x)*torch.rand(1).item()*10
#         self.random_lip_constant = self.lipschitz(u_rand, v_rand).item()
#         self.train_acc /= self.tot_steps
#         if not self.change_lamda_in and self.train_acc > self.min_acc:
#             self.lamda = min(self.lamda + self.dlamda, self.maximal_lamda)
#             self.dlamda = self.dlamda*0.95
#         elif not self.change_lamda_in :
#             self.lamda = max(self.lamda - self.dlamda, self.minimal_lamda)
#             self.dlamda = self.dlamda*0.95