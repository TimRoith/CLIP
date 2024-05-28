import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from adversarial_opt import fully_connected, adversarial_gradient_ascent, adverserial_nesterov_accelerated, adversarial_update, lip_constant_estimate, Flatten
from train_regularizer import Trainer, Polynom, scattered_points
from sklearn.decomposition import PCA
device ="cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

#coeffs = torch.tensor([0.5,0.01,-0.,0.,0.,0.,1]).to(device)
coeffs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])
polynom = Polynom(coeffs, scaling=0.00005)
xmin,xmax=(-3,3)
n_dim = 3
x = scattered_points(num_pts=50, xmin=xmin, xmax=xmax, percent_loss=0.75, random=False, dim=n_dim).to(device)
xy_loader = polynom.createdata(x,sigma=0.0)[0]
#XY = torch.stack([xy_loader.dataset.tensors[i] for i in [0,1]])


model = fully_connected([n_dim, 50, 100, 50, 1], "ReLU")
model = model.to(device)

trainer = Trainer(model, xy_loader, 100, lamda=.7, lr=0.001, adversarial_name="SGD", num_iters=50)#, backtracking=0.9)
num_total_iters = 100
for i in range(num_total_iters):
    trainer.train_step()
    if i % 1 == 0:
        print(i)
        print("train accuracy : ", trainer.train_acc)
        print("train loss : ", trainer.train_loss)
        print("train lip loss : ", trainer.train_lip_loss)

# df = (torch.rand(100)*6 - 3).unsqueeze(1).to(device)
# df.requires_grad = True
# model.zero_grad()  # Set previous gradient to zero
# output = model(df)
# output.backward(torch.ones_like(output))
# grad = df.grad
# df_near = (df+torch.rand_like(df)*0.1).to(device)
# output_near = model(df_near)
# lipschitz_estimation = torch.norm((output_near-output), p=2, dim=1)/torch.norm((df_near-df), p=2, dim=1)
# t = torch.rand(1).to(device)
# df_lin = df + t*df
# output_lin = model(df_lin)
# lipschitz_estimation_lin = torch.norm((output_lin-output), p=2, dim=1)/torch.norm((df_lin-df), p=2, dim=1)

# XX = xy = torch.stack((df.reshape(df.shape[0]), output.detach().reshape(df.shape[0]), lipschitz_estimation.detach(), lipschitz_estimation_lin.detach(), grad.detach().reshape(df.shape[0])), dim=1)
# # Run PCA on the matrix XX
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(XX.cpu().detach().numpy())

# # Plot the individual cloud for the best direction
# plt.scatter(pca_result[:, 0], pca_result[:, 1])
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('Individual Cloud for the Best Direction')
# plt.show()

# pca_values = pca.components_

# plt.figure(figsize=(10,10))
# plt.rcParams.update({'font.size': 14})
# #Plot circle
# #Create a list of 500 points with equal spacing between -1 and 1
# import numpy as np
# x=np.linspace(start=-1,stop=1,num=500)
# #Find y1 and y2 for these points
# y_positive=lambda x: np.sqrt(1-x**2) 
# y_negative=lambda x: -np.sqrt(1-x**2)
# plt.plot(x,list(map(y_positive, x)), color='maroon')
# plt.plot(x,list(map(y_negative, x)),color='maroon')
# #Plot smaller circle
# x=np.linspace(start=-0.5,stop=0.5,num=500)
# y_positive=lambda x: np.sqrt(0.5**2-x**2) 
# y_negative=lambda x: -np.sqrt(0.5**2-x**2)
# plt.plot(x,list(map(y_positive, x)), color='maroon')
# plt.plot(x,list(map(y_negative, x)),color='maroon')
# #Create broken lines
# x=np.linspace(start=-1,stop=1,num=30)
# plt.scatter(x,[0]*len(x), marker='_',color='maroon')
# plt.scatter([0]*len(x), x, marker='|',color='maroon')

# #Define color list
# colors = ['blue', 'red', 'green', 'black', 'purple', 'brown']
# if len(pca_values[0]) > 6:
#     colors=colors*(int(len(pca_values[0])/6)+1)

# columns = ['x', 'y', 'lipschitz_estimation', 'lipschitz_estimation_lin', 'grad']
# add_string=""
# for i in range(len(pca_values[0])):
#     xi=pca_values[0][i]
#     yi=pca_values[1][i]
#     plt.arrow(0,0, 
#               dx=xi, dy=yi, 
#               head_width=0.03, head_length=0.03, 
#               color=colors[i], length_includes_head=True)
#     add_string=f" ({round(xi,2)} {round(yi,2)})"
#     plt.text(pca_values[0, i], 
#              pca_values[1, i] , 
#              s=columns[i]) #+ add_string )
    
# plt.xlabel(f"Component 1 ({round(pca.explained_variance_ratio_[0]*100,2)}%)")
# plt.ylabel(f"Component 2 ({round(pca.explained_variance_ratio_[1]*100,2)}%)")
# plt.title('Variable factor map (PCA)')
# plt.show()