import torch
import mathplotlib.pyplot as plt

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

    def __call__(self, u, v):
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
        self.lr_decay = lr_decay
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

class adverserial_nesterov_accelerated:
    def __init__(self,
        model,
        u, v,
        lr = 0.1,
        lr_decay = .95,
        reg_init = "partial_random"):

        self.lr = lr
        self.lr_decay = lr_decay
        self.lip_constant_estimate = lip_constant_estimate(model)
        self.reg_init = reg_init
        self.u = u
        self.v = v

        self.alpha = torch.zeros(len(u)).to(u.device)
        self.updated_grad_uv = torch.cat((u, v)).detach() # this needs to be the initial value of the uv

    def step(self,):
        loss = self.lip_constant_estimate(self.u, self.v)
        loss_sum = torch.sum(loss)
        loss_sum.backward()

        u_grad = self.u.grad.detach()
        u_futur = self.u.data + self.lr * u_grad
        self.u.grad.zero_()
        v_grad = self.v.grad.detach()
        v_futur = self.v.data + self.lr * v_grad
        self.v.grad.zero_()
        alpha_next = (1 + torch.sqrt(1 + 4 * self.alpha ** 2)) / 2
        gamma = (1 - self.alpha) / (alpha_next)


        u_tmp = (1 - gamma) * u_futur + gamma * self.updated_grad_uv[:u_futur.shape[0]]
        v_tmp = (1 - gamma) * v_futur + gamma * self.updated_grad_uv[u_futur.shape[0]:]

        loss_tmp = self.lip_constant_estimate(u_tmp, v_tmp)
        mask =  (loss_tmp > loss_best).type(torch.int).view(-1, 1, 1, 1)
        self.u.data = u_tmp * mask + self.u.data * (1 - mask)
        self.v.data = v_tmp * mask + self.v.data * (1 - mask)

        lr = (lr / self.reg_decay * (1 - mask)) + (lr * self.reg_decay * mask)
        self.alpha = alpha_next * mask + self.alpha * (1 - mask)
        self.updated_grad_uv = torch.cat((self.u, self.v)).detach() * torch.cat((mask, mask)) + self.updated_grad_uv * (1 - torch.cat((mask, mask)))
        #u.data = torch.clamp(u.data, 0., 1.)
        #v.data = torch.clamp(v.data, 0., 1.)
        loss_best = loss_best * (1 - mask).squeeze() + loss_tmp.detach() * mask.squeeze()
        return loss_best
    
class plotting_adversarial_update:
    def __init__(self, model, x_min, x_max, adv_iters=10, type="gradient_ascent"):
        self.model = model
        self.space = torch.linspace(x_min, x_max, 1000).to(model.device)
        self.adv_iters = adv_iters
        self.type = type

    def set_uv(self, u, v):
        if self.type == "gradient_ascent":
            adv = adversarial_gradient_ascent(self.model, u, v)
        elif self.type == "nesterov_accelerated":
            adv = adverserial_nesterov_accelerated(self.model, u, v)
        else:
            raise ValueError("type should be either gradient_ascent or nesterov_accelerated")
        for _ in range(self.adv_iters):
            adv.step()
        return adv.u, adv.v
    
    def first_plot(self):
        plt.plot(self.space.cpu(), self.model(self.space.unsqueeze(1)).squeeze(1).cpu().detach())
        plt.show()
    