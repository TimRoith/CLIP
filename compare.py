import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update, lip_constant_estimate, Flatten

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Comparator:
    def __init__(self, model, lip_constant, u, v, list_name, lr =0.05, num_iters =1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.true_lipschitz = lip_constant
        self.u = u
        self.v = v
        self.list_name = list_name
        self.lr = lr
        self.num_iters = num_iters
        self.lip_constant_estimate = lip_constant_estimate(model)
        self.adv = [adversarial_update(self.model, self.u, self.v, opt_kwargs={'name':name, 'lr':self.lr}) for name in self.list_name]
        self.values = []

    def compare(self):
        for adv in self.adv:
            adv_values = torch.zeros(self.num_iters).to(self.device)
            for i in range(self.num_iters):
                adv.step()
                adv_values[i] = torch.sqrt(torch.square(self.true_lipschitz - self.lip_constant_estimate(adv.u, adv.v)))
            self.values.append(adv_values)

    def plot(self):
        for i in range(len(self.list_name)):
            plt.plot(self.values[i].cpu().detach(), label=self.list_name[i])
        plt.legend()
        plt.show()

class GaussianModel(nn.Module):
    def __init__(self, sizes, m, sd):
        super(GaussianModel, self).__init__()
        self.m = m
        self.sd = sd
        self.flatten = Flatten()
        self.linear = nn.Linear(sizes[0], 1)
        
    def forward(self, x):
        x = self.flatten(x)            # Step 1: Flatten the input
        Ax = self.linear(x)            # Step 2: Compute Ax using a linear layer
        diff = Ax - self.m             # Step 3: Subtract the mean m
        diff_squared = diff ** 2       # Step 4: Square the result
        scaled_diff = diff_squared / self.sd # Step 5: Divide by the standard deviation sd
        exp_result = torch.exp(-scaled_diff) # Step 6: Apply the negative sign and the exponential function
        return exp_result
    

sizes = [1]  # Example input size
m = torch.tensor(0.0)  # Example mean value
sd = torch.tensor(1.0)  # Example standard deviation

model = GaussianModel(sizes, m, sd).to(device)

u = torch.tensor([-2.5]*sizes[0]).to(device)
v = torch.tensor([2.5]*sizes[0]).to(device)
true_lipschitz = 1.0
list_name = ['Adam', 'SGD'] # , 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'Adamax', 'ASGD', 'LBFGS
comparator = Comparator(model, true_lipschitz, u, v, list_name)
comparator.compare()
comparator.plot()