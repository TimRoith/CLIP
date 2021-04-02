import torch
#from adversarial import get_adversarial_attack

def search_u_v(conf, model, cache):
    init = cache['init']
    num = init.shape[0] // 2
    u = init[:num]
    v = init[num:2*num]
    if "lr" not in cache or conf.reg_incremental == 1:
        loss_best = torch.zeros((init.shape[0] // 2)).to(conf.device)
        lr = torch.tensor(conf.reg_lr).repeat(num).view(-1, 1, 1, 1).to(conf.device)
    else:
        loss_best = lip_constant(conf, model, u, v)
        lr = cache["lr"]
    if conf.reg_update == "adverserial_update":
        u = u.requires_grad_()
        v = v.requires_grad_()
        u, v, loss = adverserial_update(conf, model, loss_best, lr, u, v)
    else:
        raise ValueError(conf.reg_update + "as reg_update is unknown!")

    # ------------------------------------------------------
    # find index of pair with highest Lipschitz constant
    highest_loss_idx = torch.argmax(loss)

    cache["idx"] = highest_loss_idx
    cache["lr"] = lr
    cache['init'] = torch.cat((u, v)).detach()
    return u, v



def adverserial_update(conf, model, loss_best, lr, u, v):
    iters = conf.reg_iters
    for i in range(iters):
        loss = lip_constant(conf, model, u, v)
        loss_sum = torch.sum(loss)
        loss_sum.backward()

        u_grad = u.grad.detach()
        u_tmp = u.data + lr * u_grad
        u.grad.zero_()
        v_grad = v.grad.detach()
        v_tmp = v.data + lr * v_grad
        v.grad.zero_()

        loss_tmp = lip_constant(conf, model, u_tmp, v_tmp)
        mask =  (loss_tmp > loss_best).type(torch.int).view(-1, 1, 1, 1)
        u.data = u_tmp * mask + u.data * (1 - mask)
        v.data = v_tmp * mask + v.data * (1 - mask)

        lr = (lr / conf.reg_decay * (1 - mask)) + (lr * conf.reg_decay * mask)

        #u.data = torch.clamp(u.data, 0., 1.)
        #v.data = torch.clamp(v.data, 0., 1.)

        loss_best = loss_best * (1 - mask).squeeze() + loss_tmp.detach() * mask.squeeze()
    return u, v, loss_best


def lip_constant(conf,model, u, v, mean=False):
    uv = torch.cat((u, v), 0)
    num = uv.shape[0] // 2
    output = model(uv)
    u_out = output[:num]
    v_out = output[num:2*num]
    loss = conf.out_norm(u_out - v_out) / conf.in_norm(u - v)
    if mean:
        return torch.mean(torch.square(loss))
    else:
        return torch.square(loss)


def u_v_init(conf, reg_loader, cache):
    '''
    :param loaders: train and validation loaders
    :param reg_loader: Loader used to get VALID tuples
    :param u: current VALID tuple u
    :param v: current VALID tuple v
    :param cache: cache of valid parameters
    :param i: batch
    :return: new init for u and v and current chache
    '''
    X_reg, y_reg = get_reg_batch(conf, reg_loader)
    num = X_reg.shape[0] // 2
    init = torch.zeros_like(X_reg)
    # if conf.reg_init == "adverserial":
    #     init[num:] = X_reg[num:] + get_adversarial_attack(conf, AdversarialAttacks.fgsm, conf.loss_function, model, X_reg, y_reg)[1][num:]
    #     init[:num] = X_reg[num:]
    if conf.reg_init == "partial_random":
        init[num:] = X_reg[num:] + torch.rand_like(X_reg)[num:] * 1e-1
        init[:num] = X_reg[num:]
    elif conf.reg_init == "noise":
        init = torch.rand_like(X_reg) * 1e-1
    elif conf.reg_init == "plain":
        init = X_reg
    else:
        raise Exception("Regularization init not defined:", conf.reg_init)
    
    return init

def get_reg_batch(conf, iterator):
    data, target = next(iterator)
    return data.to(conf.device), target.to(conf.device)
