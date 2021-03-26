import torch.nn as nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_model(conf):
    model = None
    if conf.model.lower() == "fc":
        model = fully_connected(conf)
    else:
        raise NameError("Modelname: {} does not exist!".format(conf.model))
    model = model.to(conf.device)
    return model


def get_activation_function(activation_function):
    af = None
    if activation_function == "ReLU":
        af = nn.ReLU
    elif activation_function == "sigmoid":
        af = nn.Sigmoid
    else:
        af = nn.ReLU
    return af
    
class fully_connected(nn.Module):
    def __init__(self, sizes, act_fun):
        super(fully_connected, self).__init__()
        
        self.act_fn = get_activation_function(act_fun)
        
        layer_list = [Flatten()]
        for i in range(len(sizes)-1):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(self.act_fn())
            
        self.layers = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        return self.layers(x)

