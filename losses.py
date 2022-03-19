import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def recon_loss(input_x, recon_x):
    mse_loss = F.mse_loss(input_x, recon_x)
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
    