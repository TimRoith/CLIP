import torch
import regularizer as reg
from itertools import cycle

def train_step(conf, model, opt, train_loader, lip_loader, verbosity = 1):
    # train phase
    model.train()
    
    # initalize values for train accuracy and train loss
    train_acc = 0.0
    train_loss = 0.0
    tot_steps = 0
    train_lip_loss = 0.0
    
    # define the set XLip as cycle
    lip_cycle = cycle(lip_loader)
    
    im_shape = train_loader.dataset.dataset.data.shape[1:]
    # Initialization for adverserial pairs
    cache = {}
    u = torch.tensor((1, *im_shape)).to(conf.device)
    v = torch.tensor((1, *im_shape)).to(conf.device)
    
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
            # if i is a muliple of the reg_increment or         
            if ((batch_idx % conf.reg_incremental) == 0) or (not ("init" in cache)):
                cache["init"] = reg.u_v_init(conf, lip_cycle, cache)
            else: # reset the initial u,v to the new pair found in the step before
                cache["init"] = torch.cat((u, v)).detach()

            # adverserial update on the Lipschitz set
            u, v, cache = reg.search_u_v(conf, model, cache=cache)
            cache["idx"]=0
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
            # Change regularization parameter alpha
            alpha = conf.alpha
            c_loss = c_loss + alpha * c_reg_loss

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
    return {'train_loss':train_loss, 'train_acc':train_acc/tot_steps, 'u':u,'v':v, 'train_lip_loss': train_lip_loss,
            'highest_loss_idx': cache["idx"]}



def validation_step(conf, model, validation_loader, u_reg, v_reg, verbosity = 1):
    val_acc = 0.0
    val_loss = 0.0
    tot_steps = 0
    val_lip_loss = 0.0
    
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
        if conf.regularization == "global_lipschitz":
            c_reg_loss = reg.lip_constant(conf, model, u_reg, v_reg, mean=conf.reg_all)
            c_loss = c_loss + conf.alpha * c_reg_loss
        
        val_acc += (logits.max(1)[1] == y).sum().item()
        val_loss += c_loss.item()
        if conf.regularization == "global_lipschitz":
            val_lip_loss = max(c_reg_loss.item(), val_lip_loss)
        tot_steps += y.shape[0]
        
    # print accuracy
    if verbosity > 0: 
        print(50*"-")
        print('Validation Accuracy:', val_acc/tot_steps)
        print('Validation Lipschitz constant', val_lip_loss)
    return {'val_loss':val_loss, 'val_acc':val_acc/tot_steps, 'val_lip_loss': val_lip_loss}