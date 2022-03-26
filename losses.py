import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vae_loss(x, recon, mu, log_var, kld_w):
    
    recons_loss = F.mse_loss(x, recon)
    
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    loss = recons_loss +  kld_w * kld_loss
    return loss, recons_loss, kld_loss


def recon_loss(recon_x, input_x):
    mse_loss = F.binary_cross_entropy(recon_x, input_x)
    return mse_loss

def sim_loss(input_z, label):
    
    batch_size = label.shape[0]
    z_dis = torch.abs(input_z.reshape(batch_size, -1, 1) - input_z.T.reshape(1, -1, batch_size))
    z_dis = torch.sum(z_dis, dim=1) # [N, N]

    same_label_mask = label.reshape(batch_size, 1) == label.reshape(1, batch_size)
    same_label_mask[list(range(batch_size)), list(range(batch_size))] = False

    except_self_mask = torch.ones_like(same_label_mask).to(device)
    except_self_mask[list(range(batch_size)), list(range(batch_size))] = False
    loss = torch.sum(z_dis[same_label_mask]) / torch.sum(z_dis[except_self_mask])
    return loss



def energy_ranking(la, label, bits, beta=1., log_space=False):
    # la : batch x length
    # label : batch x 1
    # c : batch x length x length

    # normalize
    b, k = la.shape
    assert k % bits == 0
    dim = k // bits
    la1 = (la / torch.sum(la, dim=1).unsqueeze(-1)).unsqueeze(0)  # 1 x batch x k
    la2 = (la / torch.sum(la, dim=1).unsqueeze(-1)).unsqueeze(1)  # batch x 1 x k
    la1 = la1.repeat((b, 1, 1)).reshape(b * b, k)
    la2 = la2.repeat((1, b, 1)).reshape(b * b, k)
    tensor_list = [torch.ones((bits, bits)) for _ in range(dim)]
    c = (2 * torch.ones((k, k)) - torch.block_diag(*tensor_list)).repeat((b * b, 1, 1))
    l = sinkhorn(c, la1, la2, log_space=log_space)
    # 0 for same categories, 1 for different categories
    target = (label == label.t()).float().reshape(1, -1)
    pair_potential = torch.exp(beta * l).reshape(1, -1)

    # margin ranking
    energy_diff = pair_potential - pair_potential.t()  # b x b
    label_diff = torch.sign(target - target.t())  # b x b
    objective = -energy_diff * label_diff
    loss_value = torch.sum((objective + torch.abs(objective)) / 2)
    loss_num = torch.sum(torch.sign(objective + torch.abs(objective)))
    loss = loss_value / (loss_num + 1e-10)
    return loss


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def sinkhorn_iterations(Z: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, iters: int) -> torch.Tensor:
    u, v = torch.ones_like(mu), torch.ones_like(nu)
    for _ in range(iters):
        u = mu / torch.einsum('bjk,bk->bj', [Z, v])
        v = nu / torch.einsum('bkj,bk->bj', [Z, u])
    return torch.einsum('bk,bkj,bj->bjk', [u, Z, v])


def sinkhorn(C, a, b, eps=2e-1, n_iter=10, log_space=True):
    """
    Args:
        a: tensor, normalized, note: no zero elements
        b: tensor, normalized, note: no zero elements
        C: cost Matrix [batch, n_dim, n_dim], note: no zero elements
    """
    P = torch.exp(-C/eps)
    if log_space:
        log_a = a.log()
        log_b = b.log()
        log_P = P.log()
    
        # solve the P
        log_P = log_sinkhorn_iterations(log_P, log_a, log_b, n_iter)
        P = torch.exp(log_P)
    else:
        P = sinkhorn_iterations(P, a, b, n_iter)
    return torch.sum(C * P, dim=[1, 2])


if __name__ == '__main__':
    # la : batch_size x coding_dim
    # label : batch_size x 1
    # bits : quantization bits
    # Assertion : coding_dim // bits = latent_dim, coding_dim % bits = 0

    # la = torch.tensor([[0, 1], [1, 0], [0, 1]]).reshape((3, 2))
    # label = torch.tensor([0, 1, 1]).reshape((3, 1))
    batch_size = 64
    latent_dim = 24
    bits = 2

    la = torch.randint(1, 3, size=(batch_size, latent_dim * bits)).float()
    la.requires_grad = True
    label = torch.abs(2 * torch.randn((batch_size, 1))).long()
    while True:
        loss = energy_ranking(la, label, bits, beta=1., log_space=False)
        loss.backward()
        la = torch.autograd.Variable(la - la.grad.data, requires_grad=True)
        print(loss)


