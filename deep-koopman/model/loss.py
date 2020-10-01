import torch
import torch.nn.functional as F
import numpy as np

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target, reduction='none'):
    return F.mse_loss(output, target, reduction=reduction)

def l1_loss(output, target, reduction='none'):
    return F.l1_loss(output, target, reduction=reduction)

def local_geo(g, states, scaling_factor = 10):
    bs, T = states.shape[:2]
    permu = np.random.permutation(bs * (T))
    split_0 = permu[:bs * (T) // 2]
    split_1 = permu[bs * (T) // 2: 2*(bs * (T) // 2)]

    g = g.reshape(bs*T, 1, -1) # dim 1 stands for N_obj
    states = states.reshape(bs*T, 1, -1)

    dist_g = torch.mean((g[split_0] - g[split_1]) ** 2, dim=(1, 2))
    dist_s = torch.mean((states[split_0] - states[split_1]) ** 2, dim=(1, 2))

    return torch.abs(dist_g * scaling_factor - dist_s).mean()

def embedding_loss(output, target, lambd=0.3):
    assert len(output) == 3 or len(output) == 5

    if len(output) == 5:
        rec, pred, g, mu, logvar = output
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    else:
        rec, pred, g = output
        kl_loss = torch.zeros(1)
        print('size = 3')

    free_pred = 1
    rec_loss = 10 * mse_loss(rec, target[:, -rec.shape[1]:])\
        .view(rec.size(0)*(rec.size(1)), -1).sum(dim=-1).mean()

    pred_loss = 10 * mse_loss(pred[:, :-free_pred], target[:, -pred.shape[1]:-free_pred])\
        .view(pred.size(0)*(pred.size(1)-free_pred), -1).sum(dim=-1).mean()

    local_geo_loss = local_geo(g, target[:, -g.shape[1]:])

    loss = rec_loss + pred_loss + lambd * local_geo_loss + 2 * kl_loss

    return loss, {'Rec Loss':rec_loss, 'Pred Loss':pred_loss, 'Local Geometry Loss':local_geo_loss, 'KL Loss':kl_loss}

# permu = np.random.permutation(bs * (T + 1))
# split_0 = permu[:bs * (T + 1) // 2]
# split_1 = permu[bs * (T + 1) // 2:]
# dist_g = torch.mean((g[split_0] - g[split_1]) ** 2, dim=(1, 2))
# dist_s = torch.mean((states_flat[split_0] - states_flat[split_1]) ** 2, dim=(1, 2))
# scaling_factor = 10
# loss_metric = torch.abs(dist_g * scaling_factor - dist_s).mean()
# loss_auto_encode = F.l1_loss(decode_s_for_ae, states[:, :T + 1].reshape(decode_s_for_ae.shape))
# loss_prediction = F.l1_loss(decode_s_for_pred, states[:, 1:].reshape(decode_s_for_pred.shape))
#
# loss = loss_auto_encode + loss_prediction + loss_metric * args.lambda_loss_metric