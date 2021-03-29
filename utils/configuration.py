import torch
import torch.nn.functional as F
import adversarial_attacks as at


class Conf:
    def __init__(self, **kwargs):
        #self.__dict__.update((key, value) for key, value in kwargs.items())
        
        # model
        self.model = kwargs.get('model', "fc")
        self.activation_function = kwargs.get('activation_function', "ReLU")
        
        # dataset
        self.data_set = kwargs.get('data_set', "MNIST")
        self.data_file = kwargs.get('data_file', "data")
        self.train_split = kwargs.get('train_split', 0.9)
        self.im_shape = None
        
        # CUDA settings
        self.use_cuda = kwargs.get('use_cuda', False)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_workers = kwargs.get('num_workers', 1)
        
        # Loss function and norm
        def l2_norm(X):
            return torch.norm(X.view(X.shape[0], -1), p=2, dim=1)
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
        self.reg_incremental = kwargs.get('reg_incremental', 100000)
        # -----------------------------
        self.reg_update = kwargs.get('reg_incremental', "adverserial_update")
        if not self.reg_update  in ["adverserial_update"]:
            raise ValueError("Unknown regularization update specified.")
        # -----------------------------
        self.alpha = kwargs.get('alpha', 0.0)
        self.goal_accuracy = kwargs.get('goal_accuracy', 0.9)
        self.alpha_incremental = kwargs.get('alpha_incremental', 0.0001)
 
        # specification for Training
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 128)
        self.lr = kwargs.get('lr', 100)
        
        # adverserial_attack
        self.attack = kwargs.get('attack', None)
        if self.attack == None:
            self.attack = at.no_attack()
