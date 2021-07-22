import torch
import torch.nn.functional as F
import adversarial_attacks as at
import models


class Conf:
    def __init__(self, **kwargs):
        # model
        self.model = kwargs.get('model', "fc")
        self.activation_function = kwargs.get('activation_function', "ReLU")
        
        # dataset
        self.data_set = kwargs.get('data_set', "MNIST")
        self.data_set_mean = 0.0
        self.data_set_std = 1.0
        self.data_file = kwargs.get('data_file', "data")
        self.train_split = kwargs.get('train_split', 0.9)
        self.download = False
        self.im_shape = None
        self.x_min = 0.0
        self.x_max = 1.0
        
        # CUDA settings
        self.use_cuda = kwargs.get('use_cuda', False)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_workers = kwargs.get('num_workers', 1)
        
        # Loss function and norm
        def l2_norm(x):
            return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)
        self.loss = kwargs.get('loss', F.cross_entropy)
        self.in_norm = kwargs.get('in_norm', l2_norm)
        self.out_norm = kwargs.get('out_norm', l2_norm)
        
        # specification for regularizer
        # -----------------------------
        self.regularization = kwargs.get('regularization', "global_lipschitz")
        if not self.regularization in ["global_lipschitz", "none"]:
            raise ValueError("Unknown Regularization specified.")
        # -----------------------------
        self.reg_iters = kwargs.get('reg_iters', 1)
        self.reg_lr = kwargs.get('reg_lr', 1.0)
        self.reg_decay = kwargs.get('reg_decay', 1.3)
        self.reg_interval = kwargs.get('reg_interval', 1)
        self.reg_max = kwargs.get('reg_max', 5e3)
        # -----------------------------
        self.reg_init = kwargs.get('reg_init', "plain")
        if not self.reg_init in ["partial_random", "plain", "noise"]:
            raise ValueError("Unknown regularization initialization specified.")
        # -----------------------------
        self.reg_all = kwargs.get('reg_all', False)
        self.reg_incremental = kwargs.get('reg_incremental', 2000)
        # -----------------------------
        self.reg_update = kwargs.get('reg_incremental', "adverserial_update")
        if not self.reg_update  in ["adverserial_update"]:
            raise ValueError("Unknown regularization update specified.")
        # -----------------------------
        self.lamda = kwargs.get('lamda', 0.0)
        self.lamda_increment = kwargs.get('lamda_increment', 0.0)
        self.goal_acc = kwargs.get('goal_accuracy', 0.9)
 
        # specification for Training
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 128)
        self.lr = kwargs.get('lr', 0.1)
        
        # adverserial_attack
        self.attack = kwargs.get('attack', None)
        if self.attack == None:
            self.attack = at.no_attack()
            
            
# -----------------------------
# Examples
# -----------------------------

# -----------------------------
# no regularization
# -----------------------------
def plain_example(data_file, use_cuda=False, num_workers=None, download=False):
    if use_cuda and num_workers is None:
        num_workers = 4
    else:
        num_workers = 0
    
    conf_args = {'lamda':0.0,\
                 'data_file':data_file, 'download':download, 'train_split':0.9,\
                 'use_cuda':use_cuda, 'num_workers':num_workers,\
                 'regularization':"none",\
                 'activation_function':"sigmoid"}

    # get configuration
    conf = Conf(**conf_args)
    
    # set attack
    conf.attack = at.pgd(conf.loss, epsilon=2.0, x_min=conf.x_min,x_max=conf.x_max)
    return conf

# -----------------------------
# PDG L2 Attack
# -----------------------------
def clip_example(data_file, use_cuda=False, num_workers=None, download=False):
    if use_cuda and num_workers is None:
        num_workers = 4
    else:
        num_workers = 0
    
    conf_args = {'lamda':0.1,\
                 'data_file':data_file, 'download':download, 'train_split':0.9,\
                 'use_cuda':use_cuda, 'num_workers':num_workers,\
                 'regularization':"global_lipschitz",\
                 'reg_init': "partial_random",\
                 'reg_lr':10,\
                 'activation_function':"sigmoid",\
                 'goal_acc': 0.95}

    # get configuration
    conf = Conf(**conf_args)
    
    # set attack
    conf.attack = at.pgd(conf.loss, epsilon=2.0, x_min=conf.x_min,x_max=conf.x_max)
    return conf
            
