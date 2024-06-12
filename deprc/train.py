import torch
import regularizer as reg
from itertools import cycle

def train_step(conf, model, opt, train_loader, lip_loader, cache, verbosity = 1):
    # train phase
    model.train()
    
    # initalize values for train accuracy and train loss
    train_acc = 0.0
    train_loss = 0.0
    tot_steps = 0
    train_lip_loss = 0.0
    
    # 
    u = None
    v = None
    
    # define the set XLip as cycle
    if  not lip_loader is None:
        lip_cycle = cycle(lip_loader)
    
    # -------------------------------------------------------------------------
    # loop over all batches
    for batch_idx, (x, y) in enumerate(train_loader):
        # get batch data
        x, y = x.to(conf.device), y.to(conf.device)
        
        # if the global lipschitz regularization is wanted we need to update the adverserial pair
        if conf.regularization == "global_lipschitz":
            # ---------------------------------------------------------------------
            # Adverserial update
            # ---------------------------------------------------------------------
            # get initialization for Lipschitz Training set      
            if ((cache['counter'] % conf.reg_incremental) == 0) or (not ('init' in cache)):
                if verbosity > 0:
                    print('The Lipschitz set was reset')
                cache['init'] = reg.u_v_init(conf, lip_cycle, cache)
                cache['counter'] = 1
            else:
                cache['counter'] += 1

            # adverserial update on the Lipschitz set
            u, v = reg.search_u_v(conf, model, cache)
            # ---------------------------------------------------------------------

            # Use either all tuples or only one tuple for regularization
            if conf.reg_all:
                u_reg, v_reg = u, v
            else:
                # Use idx:idx+1 to keep shape
                u_reg = u[cache["idx"]:cache["idx"] + 1].detach()
                v_reg = v[cache["idx"]:cache["idx"] + 1].detach()
                
            # Compute the Lipschitz constant
            c_reg_loss = reg.lip_constant(conf, model, u_reg, v_reg, mean=conf.reg_all)
        
        
        # reset gradients
        opt.zero_grad()
        
        # evaluate model on batch
        logits = model(x)
        
        # Get classification loss
        c_loss = conf.loss(logits, y)
        
        if conf.regularization == "global_lipschitz":
            # Change regularization parameter lamda
            lamda = conf.lamda
            # check if Lipschitz term is to large. Note that regularization with 
            # large Lipschitz terms yields instabilities!
            if not (c_reg_loss.item() > conf.reg_max):
                c_loss = c_loss + lamda * c_reg_loss
            else:
                if verbosity > 1:
                    print('The Lipschitz constant was too big:', c_reg_loss.item(), ". No Lip Regularization for this batch!")

        # Update model parameters
        c_loss.backward()
        opt.step()
        
        # update accurcy and loss
        train_acc += (logits.max(1)[1] == y).sum().item()
        train_loss += c_loss.item()
        if conf.regularization == "global_lipschitz":
            train_lip_loss = max(c_reg_loss.item(), train_lip_loss)
        tot_steps += y.shape[0]
    
    # print accuracy and loss
    if verbosity > 0: 
        print(50*"-")
        print('Train Accuracy:', train_acc/tot_steps)
        print('Train Loss:', train_loss)
        print('Lipschitz Constant', train_lip_loss)
    return {'train_loss':train_loss, 'train_acc':train_acc/tot_steps, 'u':u,'v':v, 'train_lip_loss': train_lip_loss,
            'cache':cache}



def validation_step(conf, model, validation_loader, verbosity = 1):
    val_acc = 0.0
    val_loss = 0.0
    tot_steps = 0
    
    # -------------------------------------------------------------------------
    # loop over all batches
    for batch_idx, (x, y) in enumerate(validation_loader):
        # get batch data
        x, y = x.to(conf.device), y.to(conf.device)

        # update x to a adverserial example
        x = conf.attack(model, x, y)
        
         # evaluate model on batch
        logits = model(x)
        
        # Get classification loss
        c_loss = conf.loss(logits, y)
        
        val_acc += (logits.max(1)[1] == y).sum().item()
        val_loss += c_loss.item()
        tot_steps += y.shape[0]
        
    # print accuracy
    if verbosity > 0: 
        print(50*"-")
        print('Validation Accuracy:', val_acc/tot_steps)
    return {'val_loss':val_loss, 'val_acc':val_acc/tot_steps}

def test_step(conf, model, test_loader, attack = None, verbosity = 1):
    model.eval()
    
    test_acc = 0.0
    test_loss = 0.0
    tot_steps = 0
    
    if attack is None:
        attack = conf.attack
    # -------------------------------------------------------------------------
    # loop over all batches
    for batch_idx, (x, y) in enumerate(test_loader):
        # get batch data
        x, y = x.to(conf.device), y.to(conf.device)

        # update x to a adverserial example
        x = attack(model, x, y)
        
         # evaluate model on batch
        logits = model(x)
        
        # Get classification loss
        c_loss = conf.loss(logits, y)
        
        test_acc += (logits.max(1)[1] == y).sum().item()
        test_loss += c_loss.item()
        tot_steps += y.shape[0]
        
    # print accuracy
    if verbosity > 0: 
        print(50*"-")
        print('Test Accuracy:', test_acc/tot_steps)
    return {'test_loss':test_loss, 'test_acc':test_acc/tot_steps}


class best_model:
    '''saves the best model'''
    def __init__(self, best_model=None, gamma = 0.0, goal_acc = 0.0):
        # stores best seen score and model
        self.best_score = 0.0
        
        # if specified, a copy of the model gets saved into this variable
        self.best_model = best_model

        # score function
        def score_fun(train_acc, test_acc):
            return gamma * train_acc + (1-gamma) * test_acc + (train_acc > goal_acc)
        self.score_fun = score_fun
        
    
    def __call__(self, train_acc, val_acc, model=None):
        # evaluate score
        score = self.score_fun(train_acc, val_acc)
        if score >= self.best_score:
            self.best_score = score
            # store model
            if self.best_model is not None:
                self.best_model.load_state_dict(model.state_dict())
                
                

class lamda_scheduler:
    '''scheduler for the regularization parameter'''
    def __init__(self, conf, warmup = 5, warmup_lamda = 0.0, cooldown=0):
        self.conf = conf
        
        # warm up
        self.warmup = warmup
        self.warmup_lamda = warmup_lamda
        
        # save real lamda
        self.lamda = conf.lamda        
        conf.lamda = warmup_lamda
        
        # cooldown
        self.cooldown_val = cooldown
        self.cooldown = cooldown
         
    def __call__(self, conf, train_acc, verbosity = 1):
        # check if we are still in the warm up phase
        if self.warmup > 0:
            self.warmup -= 1
            conf.lamda = self.warmup_lamda
        elif self.warmup == 0:
            self.warmup = -1
            conf.lamda = self.lamda
        else:
            # cooldown 
            if self.cooldown_val > 0:
                self.cooldown_val -= 1 
            else: # cooldown is over, time to update and reset
                self.cooldown_val = self.cooldown

                # discrepancy principle for lamda
                if train_acc > conf.goal_acc:
                    conf.lamda += conf.lamda_increment
                else:
                    conf.lamda = max(conf.lamda - conf.lamda_increment,0.0)
                    
        if verbosity > 0:
            print('Lamda was set to:', conf.lamda)
