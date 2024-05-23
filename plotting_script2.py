# import torch
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from matplotlib.animation import FuncAnimation
# import matplotlib.animation as animation

# from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = fully_connected([1, 20, 1], "sigmoid")
# for layer in model.layers:
#     if isinstance(layer, nn.Linear):
#         layer.weight.data = torch.randn(layer.weight.data.size())
# model = model.to(device)

# num_xlip = 1
# u, v = [torch.zeros(num_xlip).uniform_(-3,3).to(device) for _ in [0,1]]
# u,v = torch.tensor([-2.5]).to(device), torch.tensor([2.5]).to(device)
# adv = adversarial_update(model, u, v, opt_kwargs={'name':'Adam', 'lr':.005})

# # Close any existing plots
# plt.close('all')

# # Create a figure and axis for the plot
# fig, ax = plt.subplots(1,)

# # Define plot limits and initial data
# x_min, x_max = (-3,3)
# s = torch.linspace(x_min, x_max, 1000).to(model.device)
# ax.plot(s.cpu(), model(s).cpu().detach())

# uv = torch.cat([u, v])
# sc = ax.scatter(uv.cpu().detach().squeeze(),
#                 model(uv).cpu().detach().squeeze(),
#                 color= num_xlip * ['red'] + num_xlip * ['blue'])

# num_iters = 1000

# # Function to update the plot
# def update(frame):
#     adv.step()
#     ax.set_title(f'Iteration: {frame}')
#     uv = torch.cat([adv.u, adv.v])
#     oup = model(uv).detach().squeeze()
#     scdata = torch.stack([uv.detach().squeeze(), oup]).T
#     sc.set_offsets(scdata.cpu())

# # Create animation
# ani = FuncAnimation(fig, update, frames=num_iters, interval=100)

# # Save animation
# ani.save('adversarial_optimization.mp4', writer='ffmpeg')

# plt.show()

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fully_connected([1, 20, 1], "sigmoid")
for layer in model.layers:
    if isinstance(layer, nn.Linear):
        layer.weight.data = torch.randn(layer.weight.data.size())
model = model.to(device)

num_xlip = 1
u, v = [torch.zeros(num_xlip).uniform_(-3,3).to(device) for _ in [0,1]]
u, v = torch.tensor([-2.5]).to(device), torch.tensor([2.5]).to(device)
adv = adversarial_update(model, u, v, opt_kwargs={'name':'Adam', 'lr':.005})

# Close any existing plots
plt.close('all')

# Create a figure and axis for the plot
fig, ax = plt.subplots(1,)

# Define plot limits and initial data
x_min, x_max = (-3, 3)
s = torch.linspace(x_min, x_max, 1000).to(model.device)
ax.plot(s.cpu(), model(s).cpu().detach())

uv = torch.cat([u, v])
sc = ax.scatter(uv.cpu().detach().squeeze(),
                model(uv).cpu().detach().squeeze(),
                color=num_xlip * ['red'] + num_xlip * ['blue'])

num_iters = 1000

# Function to compute the interval based on the frame number
def compute_interval(frame):
    initial_interval = 1000  # Start with 1000 ms
    final_interval = 100  # End with 100 ms
    alpha = frame / num_iters
    return initial_interval * (1 - alpha) + final_interval * alpha

# Function to update the plot
def update(frame):
    adv.step()
    ax.set_title(f'Iteration: {frame}')
    uv = torch.cat([adv.u, adv.v])
    oup = model(uv).detach().squeeze()
    scdata = torch.stack([uv.detach().squeeze(), oup]).T
    sc.set_offsets(scdata.cpu())
    # Adjust the interval for the next frame
    interval = compute_interval(frame)
    plt.pause(interval / 1000)

# Create animation
ani = FuncAnimation(fig, update, frames=num_iters, interval=compute_interval(0))

# Save animation
ani.save('adversarial_optimization.mp4', writer='ffmpeg')

plt.show()