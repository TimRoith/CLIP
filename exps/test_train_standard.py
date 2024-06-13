import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer
from flip.load_data import load_MNIST_test, load_MNIST
from flip.utils.config import cfg, dataset, model_attributes
import matplotlib.pyplot as plt

#%%
CFG = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_max_clip.pth', #'model_sum_clip.pth'
              )
          )

model = load_model.load(CFG)
dataloader= load_MNIST(CFG)
trainer = StandardTrainer(model, dataloader, 
                          opt_kwargs={'type': torch.optim.Adam},
                          verbosity=1)
#%%
trainer.train()




#%%
attack = pgd(proj='linf', max_iters=500, epsilon=0.3)