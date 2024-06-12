import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        if sum(x.shape)>2:
            return x.view(x.size(0), -1)
        else:
            return x

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