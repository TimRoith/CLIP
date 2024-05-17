import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fully_connected([1, 20, 1], "sigmoid")
for layer in model.layers:
    if isinstance(layer, nn.Linear):
        layer.weight.data = torch.randn(layer.weight.data.size())
        print(layer.weight.data)
model = model.to(device)

u, v = (torch.tensor([-0.5]).to(device), torch.tensor([0.5]).to(device))

adv = adversarial_gradient_ascent(model, u, v, lr=10)
adv = adverserial_nesterov_accelerated(model, u, v, lr=10)
#%%
plt.close('all')
fig, ax = plt.subplots(1,)

x_min, x_max = (-3,3)
s = torch.linspace(x_min, x_max, 1000).to(model.device)
ax.plot(s,model(s).detach())

sc = ax.scatter(torch.stack([u,v]).detach().squeeze(),
                model(torch.stack([u,v])).detach().squeeze(),
                color= ['red', 'blue'])

num_iters = 100

for i in range(num_iters):
    adv.step()
    ax.set_title('Iteration: ' + str(i))
    uv = torch.stack([adv.u, adv.v])
    oup = model(uv).detach().squeeze()
    scdata = torch.stack([uv.detach().squeeze(), oup]).T
    sc.set_offsets(scdata)
    plt.pause(0.1)
    
