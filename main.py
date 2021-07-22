import torch
import utils.configuration as cf
from utils.datasets import get_data_set
import models
import train

# -----------------------------------------------------------------------------------
# Set up variable and data for an example
# -----------------------------------------------------------------------------------
# specify the path of your data
data_file = "/"

# load up configuration from examples
# conf = cf.plain_example(data_file, use_cuda=False, download=False)
conf = cf.clip_example(data_file, use_cuda=False, download=False)

# get train, validation and test loader
train_loader, valid_loader, test_loader = get_data_set(conf)


# -----------------------------------------------------------------------------------
# define the model and an instance of the best model class
# -----------------------------------------------------------------------------------
sizes = [784, 200, 80, 10]
model = models.fully_connected(sizes, conf.activation_function).to(conf.device)
best_model = train.best_model(models.fully_connected(sizes, conf.activation_function).to(conf.device), goal_acc = conf.goal_acc)

# -----------------------------------------------------------------------------------
# Initialize optimizer and lamda scheduler
# -----------------------------------------------------------------------------------
opt = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
lamda_scheduler = train.lamda_scheduler(conf, warmup = 5, warmup_lamda = 0.0, cooldown = 1)
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# initalize history
# -----------------------------------------------------------------------------------
tracked = ['train_loss', 'train_acc', 'train_lip_loss', 'val_loss', 'val_acc']
history = {key: [] for key in tracked}
# -----------------------------------------------------------------------------------
# cache for the lipschitz update
cache = {'counter':0}



print("Train model: {}".format(conf.model))
for i in range(conf.epochs):
    print(25*"<>")
    print(50*"|")
    print(25*"<>")
    print('Epoch', i)
    
    # train_step
    train_data = train.train_step(conf, model, opt, train_loader, valid_loader, cache)
    
    # ------------------------------------------------------------------------
    # validation step
    val_data = train.validation_step(conf, model, valid_loader)
    
    # ------------------------------------------------------------------------
    # update history
    for key in tracked:
        if key in val_data:
            history[key].append(val_data[key])
        if key in train_data:
            history[key].append(train_data[key])
    
    # ------------------------------------------------------------------------
    lamda_scheduler(conf, train_data['train_acc'])
    best_model(train_data['train_acc'], val_data['val_acc'], model=model)
    

# -----------------------------------------------------------------------------------
# Test the model afterwards
# -----------------------------------------------------------------------------------
conf.attack.attack_iters=100
test_data = train.test_step(conf, best_model.best_model, test_loader, attack=conf.attack)
