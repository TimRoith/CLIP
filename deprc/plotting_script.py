import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from compare import Comparator, GaussianModel, BasicGaussianModel

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fully_connected([1, 20, 1], "relu")
for layer in model.layers:
    if isinstance(layer, nn.Linear):
        layer.weight.data = torch.randn(layer.weight.data.size())
model = BasicGaussianModel().to(device)

num_xlip = 1
u, v = [torch.zeros(num_xlip).uniform_(-3,3).to(device) for _ in [0,1]]
u, v = torch.tensor([-2.5]).to(device), torch.tensor([2.5]).to(device)
#adv = adversarial_gradient_ascent(model, u, v, lr=10)
adv = adversarial_update(model, u, v, opt_kwargs={'name':'Adam', 'lr':.005})
#adv = adverserial_nesterov_accelerated(model, u, v, lr=10)
#%%
plt.close('all')
fig, ax = plt.subplots(1,)

x_min, x_max = (-3,3)
s = torch.linspace(x_min, x_max, 1000).to(device)
ax.plot(s.cpu(),model(s).cpu().detach())

uv = torch.cat([u,v])
sc = ax.scatter(uv.cpu().detach().squeeze(),
                model(uv).cpu().detach().squeeze(),
                color= num_xlip * ['red'] + num_xlip * ['blue'])

num_iters = 1000

for i in range(num_iters):
    adv.step()
    ax.set_title('Iteration: ' + str(i))
    uv = torch.cat([adv.u, adv.v])
    oup = model(uv).detach().squeeze()
    scdata = torch.stack([uv.detach().squeeze(), oup]).T
    sc.set_offsets(scdata.cpu())
    plt.pause(0.1)
    
