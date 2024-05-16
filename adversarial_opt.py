import torch

def l2_norm(x):
    return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)

def get_reg_batch(conf, iterator):
    data, target = next(iterator)
    return data.to(conf.device), target.to(conf.device)

def default_initializer():
    pass

class lip_constant_estimate:
    def __init__(
        self, model, 
        out_norm = None, 
        in_norm=None, 
        mean=False
    ):

        self.model = model
        self.out_norm = out_norm if out_norm is not None else l2_norm
        self.in_norm = in_norm if in_norm is not None else l2_norm
        self.mean = mean

    def __call__(self, u, v)
        uv = torch.cat((u, v), 0)
        num = uv.shape[0] // 2
        output = self.model(uv)
        u_out = output[:num]
        v_out = output[num:2*num]
        loss = self.out_norm(u_out - v_out) / self.in_norm(u - v)
        if self.mean:
            return torch.mean(torch.square(loss))
        else:
            return torch.square(loss)

class adversarial_gradient_ascent:
    def __init__(self,
        model,
        u, v,
        lr = 0.1,
        lr_decay = .95,
        reg_init = "partial_random"):

        self.lr = lr
        self.lip_constant_estimate = lip_constant_estimate(model)
        self.reg_init = reg_init

        self.u = u
        self.v = v

    def step(self,):
        loss = self.lip_constant_estimate(self.u, self.v)
        loss_sum = torch.sum(loss)
        loss_sum.backward()

        for z in [self.u, self.v]:
            z_grad = z.grad.detach()
            z_tmp = z.data + self.lr * z_grad
            z.grad.zero_()
            z.data = z_tmp

        # loss_tmp = lip_constant(conf, model, u_tmp, v_tmp)
        # mask =  (loss_tmp > loss_best).type(torch.int).view(-1, 1, 1, 1)
        # u.data = u_tmp * mask + u.data * (1 - mask)
        # v.data = v_tmp * mask + v.data * (1 - mask)
        self.lr = (self.lr / self.lr_decay) # * (1 - mask)) + (lr * conf.reg_decay * mask)
        # u.data = torch.clamp(u.data, 0., 1.)
        # v.data = torch.clamp(v.data, 0., 1.)

        # loss_best = loss_best * (1 - mask).squeeze() + loss_tmp.detach() * mask.squeeze()