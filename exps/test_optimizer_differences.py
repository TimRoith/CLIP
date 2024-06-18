import torch
from flip.models import load_model
from flip.attacks import pgd
from flip.train import StandardTrainer, FLIPTrainer, AdversarialTrainer
from flip.load_data import load_MNIST_test, load_MNIST
from flip.utils.config import cfg, dataset, model_attributes
import matplotlib.pyplot as plt
import time 

estimation = 'max'
#%%
CFG = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_diff_optimizer.pth',
              )
          )

model_SGD = load_model.load(CFG)
dataloader= load_MNIST(CFG)
#%%
def eval_acc(model, x, y):
    return torch.sum(model(x).topk(1)[1][:,0]==y)

#%%
attack = pgd(proj='linf', max_iters=500, epsilon=0.8)
epochs = 10

# #%%
# model_ADV = load_model.load(CFG)
# trainer = AdversarialTrainer(model_ADV, dataloader,
#                           opt_kwargs={'type': torch.optim.Adam },
#                           verbosity=1,
#                           epochs=epochs,)

# start_time = time.time()
# trainer.train()
# elapsed_time_ADV = time.time() - start_time

# x,y = next(iter(dataloader))
# attack(model_ADV, x, y)
# delta = attack.delta
# eval_acc(model_ADV, x+delta, y)
# acc_ADV = eval_acc(model_ADV, x+delta, y)/len(y)

# hist_ADV = trainer.hist.copy()

#%%
CFG_adv = cfg(data=dataset(), 
          model = model_attributes(
              name = 'FC', 
              sizes=[784, 200, 80, 10],
              act_fun = 'ReLU',
              file_name = 'model_adv_training.pth', #'model_sum_clip.pth'
              )
          )

model_adv = load_model.load(CFG_adv)
x,y = next(iter(dataloader))
attack(model_adv, x, y)
delta = attack.delta
eval_acc(model_adv, x+delta, y)
acc_adv = eval_acc(model_adv, x+delta, y)/len(y)
#%%

trainer = FLIPTrainer(model_SGD, dataloader,
                          lamda=0.1,
                          num_iters=2,
                          estimation=estimation,
                          opt_kwargs={'type': torch.optim.Adam },
                          adv_kwargs={'name' : 'SGD', 'lr' : 0.07},
                          verbosity=1,
                          epochs=epochs,
                          min_acc=1.,)

start_time = time.time()
trainer.train()
elapsed_time_SGD = time.time() - start_time

x,y = next(iter(dataloader))
attack(model_SGD, x, y)
delta = attack.delta
eval_acc(model_SGD, x+delta, y)
acc_SGD = eval_acc(model_SGD, x+delta, y)/len(y)

hist_SGD = trainer.hist.copy()

#%%
model_NAG = load_model.load(CFG)
trainer = FLIPTrainer(model_NAG, dataloader,
                          lamda=0.1,
                          num_iters=2,
                          estimation=estimation,
                          opt_kwargs={'type': torch.optim.Adam },
                          adv_kwargs={'name' : 'Nesterov', 'lr' : 0.1},
                          verbosity=1,
                          epochs=epochs,
                          min_acc=1.,)

start_time = time.time()
trainer.train()
elapsed_time_ADAM = time.time() - start_time

x,y = next(iter(dataloader))
attack(model_NAG, x, y)
delta = attack.delta
eval_acc(model_NAG, x+delta, y)
acc_NAG = eval_acc(model_NAG, x+delta, y)/len(y)

hist_NAG = trainer.hist.copy()

#%%
model_ADAM = load_model.load(CFG)
trainer = FLIPTrainer(model_ADAM, dataloader,
                          lamda=0.1,
                          num_iters=2,
                          estimation=estimation,
                          opt_kwargs={'type': torch.optim.Adam },
                          adv_kwargs={'name' : 'Adam', 'lr' : 0.01},
                          verbosity=1,
                          epochs=epochs,
                          min_acc=1.,)

start_time = time.time()
trainer.train()
elapsed_time_NAG = time.time() - start_time

x,y = next(iter(dataloader))
attack(model_ADAM, x, y)
delta = attack.delta
eval_acc(model_ADAM, x+delta, y)
acc_ADAM = eval_acc(model_ADAM, x+delta, y)/len(y)

hist_ADAM = trainer.hist.copy()


#%%
model_STA = load_model.load(CFG)
trainer = StandardTrainer(model_STA, dataloader,
                          opt_kwargs={'type': torch.optim.Adam },
                          verbosity=1,
                          epochs=epochs,)

start_time = time.time()
trainer.train()
elapsed_time_STA = time.time() - start_time

x,y = next(iter(dataloader))
attack(model_STA, x, y)
delta = attack.delta
eval_acc(model_STA, x+delta, y)
acc_STA = eval_acc(model_STA, x+delta, y)/len(y)

hist_STA = trainer.hist.copy()

#%%
plt.figure(figsize=(10, 5))

# Plot acc
plt.subplot(1, 2, 1)
plt.plot(hist_SGD['acc'], label='SGD')
plt.plot(hist_NAG['acc'], label='Nesterov')
plt.plot(hist_ADAM['acc'], label='Adam')
# plt.plot(hist_ADV['acc'], label='Adversarial')
plt.plot(hist_STA['acc'], label='Standard')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(hist_SGD['loss'], label='SGD')
plt.plot(hist_NAG['loss'], label='Nesterov')
plt.plot(hist_ADAM['loss'], label='Adam')
# plt.plot(hist_ADV['loss'], label='Adversarial')
plt.plot(hist_STA['loss'], label='Standard')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#%%
print('time SGD: ', elapsed_time_SGD)
print('time NAG: ', elapsed_time_NAG)
print('time ADAM: ', elapsed_time_ADAM)
# print('time ADV: ', elapsed_time_ADV)
print('time STA: ', elapsed_time_STA)
print('adv acc SGD: ', acc_SGD)
print('adv acc NAG: ', acc_NAG)
print('adv acc ADAM: ', acc_ADAM)
# print('adv acc ADV: ', acc_ADV)
print('adv acc ADV: ', acc_adv)
print('adv acc STA: ', acc_STA)