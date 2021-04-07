import torch

class attack:
    pass

# No attack     
class no_attack(attack):
    #
    def __call__(self, model, x, y):
        return x



# get gauss noise augmentation. Magnitude is dependent on noise level
class gauss_attack(attack):
    def __init__(self, nl=1.0, x_min=0.0, x_max=1.0):
        super(gauss_attack, self).__init__()
        self.nl = nl
        self.x_min = x_min
        self.x_max = x_max
    
    #
    def __call__(self, model, x, y):
        return torch.clamp(x+torch.randn_like(x) * self.nl, min=self.x_min, max=self.x_max)

    
# fgsm attack
class fgsm(attack):
    def __init__(self, loss, epsilon=0.3, x_min=0.0, x_max=1.0):
        super(fgsm, self).__init__()
        self.epsilon = epsilon
        self.loss = loss
        self.x_min = x_min
        self.x_max = x_max
        
    def __call__(self, model, x, y):
        #get delta
        delta = get_delta(x, self.epsilon, x_min=self.x_min, x_max=self.x_max)
        delta.requires_grad = True
        # get loss
        pred = model(x + delta)
        loss = self.loss(pred, y)
        loss.backward()
        # get example
        grad = delta.grad.detach()
        delta.data = delta + self.epsilon * torch.sign(grad)
        return torch.clamp(x + delta.detach(), min=self.x_min, max=self.x_max)
    
    

class pgd(attack):
    '''pgd attack where the attack is not updated for samples where it was already successful
    (this gives a better lower bound on the robustness)'''
    def __init__(self, loss, epsilon=None, x_min=0.0, x_max=1.0, restarts=1, 
                 attack_iters=7, alpha=None, alpha_mul=1.0, norm_type="l2"):
        super(pgd, self).__init__()
        self.loss = loss
        self.x_min = x_min
        self.x_max = x_max
        self.restarts=restarts
        self.attack_iters = attack_iters
        self.alpha_mul=alpha_mul
        
        # set norm and set epsilon and alpha accordingly
        self.norm_type = norm_type
        if self.norm_type not in ["l2","linf"]:
            raise ValueError("Unknown norm specified for pdg attack")
        if epsilon is None:
            if norm_type == "l2":
                self.epsilon = 2.0
            else:
                self.epsilon = 1.0
        else:
            self.epsilon = epsilon
        # alpha
        if alpha is None:
            if norm_type == "l2":
                self.alpha = 0.5
            else:
                self.epsilon = 0.3/4
        else:
            self.alpha = alpha
            
          
    def __call__(self, model, x, y):
        # initilaize delta
        delta = get_delta(x, self.epsilon, uniform=True)
        if self.norm_type == "l2":
            delta = delta / torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).view(delta.shape[0], 1, 1, 1) * self.epsilon
        
        index = torch.arange(0, x.shape[0], dtype=torch.long)
        # Restarting the attack to prevent getting stuck
        for i in range(self.restarts):
            delta.requires_grad = True
            
            # restart get new delta
            if i > 0:
                delta = get_delta(x, self.epsilon, uniform=True)
                if self.norm_type == "l2":
                    delta = delta / torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).view(delta.shape[0], 1, 1, 1) * self.epsilon
                delta.data[index] = delta[index]

            for _ in range(self.attack_iters):
                pred = model(x + delta)
                # indexes are used to determine which samples needs to be updated
                index = torch.where(pred.max(1)[1] == y)[0]
                if len(index) == 0:
                    break
                # get loss and step backward
                loss = self.loss(pred, y)
                loss.backward()
                grad = delta.grad.detach()
                if self.norm_type == "linf":
                    d = torch.clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
                else:
                    d = delta + self.alpha * torch.sign(grad)
                    d = clamp(d, self.x_min - x, self.x_max - x)
                    d = d / torch.norm(d.view(d.shape[0], -1), p=2, dim=1).view(d.shape[0], 1, 1, 1) * self.epsilon
                #
                delta.data[index] = d[index]
                delta.grad.zero_()
        return torch.clamp(x + delta.detach(), self.x_min, self.x_max)
                    


                    
def clamp(x, x_min, x_max):
    return torch.max(torch.min(x, x_max), x_min)
                    
def get_delta(x, eps=1.0, uniform=False, x_min=0.0, x_max=1.0):
    delta = torch.zeros_like(x)
    if uniform:
        delta.uniform_(-eps, eps)
    return clamp(delta, x_min - x, x_max - x)
