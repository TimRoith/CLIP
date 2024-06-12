import torch
import torch.nn as nn
import torch.optim as optim


class attack:
    def __init__(
            self, 
            x_range = None,
            epsilon = 0.1
            ):
        
        self.x_range = (0,1) if x_range is None else x_range
        self.delta = None
        self.epsilon = epsilon
        
        
    def ensure_range(self, x):
        self.delta.data = torch.clamp(self.delta.data, self.x_range[0] - x, self.x_range[1] - x)
        
    def init_delta(self, x, uniform = True):
        self.delta = nn.Parameter(torch.zeros_like(x))
        if uniform:
            self.delta.data.uniform_(-self.epsilon, self.epsilon)
        self.ensure_range(x)
        

def proj_l2(delta, eps, dim=(-2,-1)):
    return (eps/torch.clamp(torch.linalg.vector_norm(delta, dim=dim, keepdim=True), min=eps)) * delta
    
def proj_linf(delta, eps, dim=(-2,-1)):
    torch.clamp(delta.data, min=-eps, max=eps, out=delta.data)
    
proj_dict = {'l2': proj_l2, 'linf': proj_linf}

def select_proj(proj):
    if proj in proj_dict.keys():
        return proj_dict[proj]
    else:
        raise ValueError('Unkwon projection: ' + str(proj) + ' Please choose from:' + str(proj_dict.keys()))
        
class pgd(attack):
    def __init__(
            self, 
            x_range = None,
            epsilon = 0.1,
            proj = 'l2',
            max_iters = 10,
            loss = None,
            opt_kwargs = None,
            targeted = False
            ):
        super().__init__(x_range=x_range, epsilon=epsilon)
        
        self.max_iters = max_iters
        self.loss = nn.CrossEntropyLoss(reduction='sum') if loss is None else loss
        opt_kwargs = {'lr':0.1} if opt_kwargs is None else opt_kwargs
        self.opt_cls = opt_kwargs.get('type', optim.SGD)
        self.opt_kwargs = {k:v for k,v in opt_kwargs.items() if not k=='type'}
        self.sched_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.sched_kwargs = {'factor':0.75, 'patience':10, 'min_lr': 1e-6} 
        self.sgn = -1 + 2 * targeted

        
        if isinstance(proj, str):
            self.project = select_proj(proj)
        else:
            self.project = proj
        
        
    def __call__(self, model, x, y):
        self.init_summary()
        self.init_delta(x)
        self.project(self.delta, self.epsilon)
        self.opt = self.opt_cls([self.delta], **self.opt_kwargs)
        self.sched = self.sched_cls(self.opt, **self.sched_kwargs)
        
        for it in range(self.max_iters):
            self.opt.zero_grad()
            inp = torch.clamp(x + self.delta, *self.x_range)
            pred = model(inp)
            loss = self.sgn * self.loss(pred, y)
            loss.backward()
            self.opt.step()
            self.project(self.delta, self.epsilon)
            self.ensure_range(x)
            self.sched.step(loss)
            self.summary['losses'].append(loss.item())
            
            
    def init_summary(self):
        self.summary = {'losses':[]}
        

# class pgd(attack):
#     def __init__(self, loss, epsilon=None, x_min=0.0, x_max=1.0, restarts=1, 
#                  attack_iters=7, alpha=None, alpha_mul=1.0, norm_type="l2",
#                  classes = None, verbosity= 1):
#         super(pgd, self).__init__()
#         self.loss = loss
#         self.x_min = x_min
#         self.x_max = x_max
#         self.restarts=restarts
#         self.attack_iters = attack_iters
#         self.alpha_mul=alpha_mul
#         self.classes=classes
#         self.losses = []
#         self.verboisty = verbosity
        
#         # set norm and set epsilon and alpha accordingly
#         self.norm_type = norm_type
#         if self.norm_type not in ["l2","linf"]:
#             raise ValueError("Unknown norm specified for pdg attack")
#         if epsilon is None:
#             if norm_type == "l2":
#                 self.epsilon = 2.0
#             else:
#                 self.epsilon = 1.0
#         else:
#             self.epsilon = epsilon
#         # alpha
#         if alpha is None:
#             if norm_type == "l2":
#                 self.alpha_init = 0.5
#             else:
#                 self.alpha_init = 0.3/4
#         else:
#             self.alpha_init = alpha
            
          
#     def __call__(self, model, x, y):
#         self.alpha = self.alpha_init
#         self.losses = []
#         prev_g = None
#         # initilaize delta
#         delta = get_delta(x, self.epsilon, uniform=True)
#         if self.norm_type == "l2":
#             delta = delta / torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).view(delta.shape[0], 1, 1, 1) * self.epsilon
        
#         index = torch.arange(0, x.shape[0], dtype=torch.long)
#         # Restarting the attack to prevent getting stuck
#         for i in range(self.restarts):
#             delta.requires_grad = True
            
#             # restart get new delta
#             if i > 0:
#                 delta = get_delta(x, self.epsilon, uniform=True)
#                 if self.norm_type == "l2":
#                     delta = delta / torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).view(delta.shape[0], 1, 1, 1) * self.epsilon
#                 delta.data[index] = delta[index]
#                 delta.requires_grad = True

#             for j in range(self.attack_iters):
#                 inp = torch.clamp(x + delta, self.x_min, self.x_max)
#                 pred = model(inp)
                
#                 # indexes are used to determine which samples needs to be updated
#                 index = torch.where(pred.max(1)[1] != y)[0]
#                 if len(index) == 0:
#                     break
#                 # get loss and step backward
#                 loss = self.loss(pred, y)
#                 loss.backward()
#                 grad = delta.grad.detach()
#                 if prev_g is not None:
#                     grad = 0.9 * prev_g + 0.1 * grad

#                 prev_g = grad.clone()
                
#                 if self.verboisty > 0:
#                     print(20*'-')
#                     print('Iteration: ' + str(j))
#                     print('Prediction: ' +str(pred.argmax(dim=1)))
#                     print('Loss: ' + str(loss.item()))
                
#                 if self.norm_type == "linf":
#                     d = torch.clamp(delta - self.alpha * grad, -self.epsilon, self.epsilon)
#                 else:
#                     d = delta - self.alpha * torch.sign(grad)
#                     d = clamp(d, self.x_min - x, self.x_max - x)
#                     d = d / torch.norm(d.view(d.shape[0], -1), p=2, dim=1).view(d.shape[0], 1, 1, 1) * self.epsilon
#                 #
#                 delta.data[index] = d[index]
#                 delta.grad.zero_()
                
#                 self.losses.append(loss.item())
#                 self.losses = self.losses[-5:]
#                 if self.losses[-1] > self.losses[0]:
#                     alpha = self.alpha * 0.75
#                     self.alpha = max(alpha, 1e-6)
#         summary = {'num_its':j}
#         return torch.clamp(x + delta.detach(), self.x_min, self.x_max), summary
